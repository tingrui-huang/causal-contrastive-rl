"""
Actor-driven forced-U evaluation (NO planner, NO waypoints).

The actor receives only the final goal state and outputs actions directly.
This cleanly tests the learned policy without privileged planner assistance.

Usage:
  python experiments/evaluate_actor_forced_u.py \
      --checkpoint checkpoints/actor_critic_confounded_seed0_....npz \
      --eval-env-id CausalContrastive-WindyCorridor-15x15-Lethal-v0 \
      --episodes-per-regime 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.core.actions import Actions

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs  # noqa: F401

from agents.goal_conditioned_actor_numpy import GoalConditionedActorNumpy
from utils.preprocess import extract_state, extract_state_oracle

VALID_ACTIONS = [int(Actions.left), int(Actions.right), int(Actions.forward)]
CSV_FIELDS = [
    "checkpoint", "eval_env_id",
    "success_u0", "success_u1", "mean_success", "worst_case_success", "regime_gap",
]


def _load_checkpoint(path: Path) -> tuple[GoalConditionedActorNumpy, dict[str, Any]]:
    ckpt = np.load(path, allow_pickle=True)
    cfg = json.loads(str(ckpt["train_config_json"]))
    state_dim = int(ckpt["state_dim"])
    n_actions = int(ckpt["n_actions"])
    actor_hidden = int(ckpt["actor_hidden"])

    actor = GoalConditionedActorNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=actor_hidden,
        seed=0,
    )
    actor.W1[...] = ckpt["actor_W1"]
    actor.b1[...] = ckpt["actor_b1"]
    actor.W2[...] = ckpt["actor_W2"]
    actor.b2[...] = ckpt["actor_b2"]
    return actor, cfg


def _goal_pos(raw) -> tuple[int, int]:
    goal_attr = raw.goal_pos
    if callable(goal_attr):
        return tuple(int(v) for v in goal_attr())
    return tuple(int(v) for v in goal_attr)


def _shortest_path_action(env) -> int:
    """BFS safe-topology action for generating goal state reference."""
    raw = env.unwrapped
    pos = tuple(int(v) for v in raw.agent_pos)
    goal = _goal_pos(raw)
    if pos == goal:
        return int(Actions.done)

    walkable = raw.walkable_cells()
    q: deque[tuple[int, int]] = deque([pos])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {pos: None}
    while q:
        cell = q.popleft()
        if cell == goal:
            break
        x, y = cell
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in walkable and nxt not in parent:
                parent[nxt] = cell
                q.append(nxt)

    if goal not in parent:
        return int(Actions.forward)

    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()

    if len(path) < 2:
        return int(Actions.done)

    dx = path[1][0] - path[0][0]
    dy = path[1][1] - path[0][1]
    target_dir = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[(dx, dy)]
    cur_dir = int(raw.agent_dir)
    if cur_dir == target_dir:
        return int(Actions.forward)
    if (target_dir - cur_dir) % 4 == 1:
        return int(Actions.right)
    return int(Actions.left)


def _get_goal_state(
    *,
    env_id: str,
    fixed_u: int,
    seed: int,
    state_fn,
) -> np.ndarray:
    """Run a deterministic reference to get the goal state vector."""
    env = gym.make(env_id, render_mode="rgb_array", fixed_u=fixed_u, wind_strength=0.0)
    obs, info = env.reset(seed=seed)
    goal = _goal_pos(env.unwrapped)

    for _ in range(env.unwrapped.max_steps):
        action = _shortest_path_action(env)
        if action == int(Actions.done):
            break
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()

    pos = tuple(int(v) for v in env.unwrapped.agent_pos)
    if pos != goal:
        raise RuntimeError(f"Reference did not reach goal: ended at {pos}, goal is {goal}")

    if "confounder" not in info:
        info["confounder"] = fixed_u
    return state_fn(obs, info).astype(np.float64)


def _run_regime(
    *,
    actor: GoalConditionedActorNumpy,
    eval_env_id: str,
    fixed_u: int,
    episodes: int,
    goal_state: np.ndarray,
    state_fn,
    max_steps: int = 200,
    temperature: float = 1.0,
    verbose: bool = False,
) -> dict[str, Any]:
    successes = 0
    lava_deaths = 0
    timeouts = 0

    for ep in range(episodes):
        ep_seed = 100 + fixed_u * 1000 + ep
        rng = np.random.default_rng(ep_seed)
        env = gym.make(eval_env_id, render_mode="rgb_array", fixed_u=fixed_u)
        obs, info = env.reset(seed=ep_seed)
        if "confounder" not in info:
            info["confounder"] = int(getattr(env.unwrapped, "hidden_u", 0))

        success = False
        died = False
        for step in range(max_steps):
            state = state_fn(obs, info).astype(np.float64)
            action = actor.sample_action(
                state, goal_state,
                valid_actions=VALID_ACTIONS,
                rng=rng,
                temperature=temperature,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            if "confounder" not in info:
                info["confounder"] = int(getattr(env.unwrapped, "hidden_u", 0))

            if reward > 0:
                success = True
                break
            if terminated:
                died = True
                break
            if truncated:
                break

        if success:
            successes += 1
        elif died:
            lava_deaths += 1
        else:
            timeouts += 1

        env.close()

    rate = successes / float(episodes)
    if verbose:
        print(f"  u={fixed_u}: success={successes}/{episodes}={rate:.2f}  "
              f"lava={lava_deaths}  timeout={timeouts}")
    return {"success_rate": rate, "successes": successes,
            "lava_deaths": lava_deaths, "timeouts": timeouts}


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    eval_env_id: str,
    episodes_per_regime: int,
    temperature: float = 1.0,
    verbose: bool = True,
) -> dict[str, Any]:
    actor, cfg = _load_checkpoint(checkpoint)
    is_oracle = cfg.get("oracle_state", False)
    state_fn = extract_state_oracle if is_oracle else extract_state
    seed = int(cfg.get("seed", 0))

    if verbose:
        print(f"[Eval] checkpoint: {checkpoint.name}")
        print(f"[Eval] oracle_state={is_oracle}  env={eval_env_id}  "
              f"episodes={episodes_per_regime}  temp={temperature}")

    goal_u0 = _get_goal_state(env_id=eval_env_id, fixed_u=0, seed=seed, state_fn=state_fn)
    goal_u1 = _get_goal_state(env_id=eval_env_id, fixed_u=1, seed=seed, state_fn=state_fn)

    if verbose:
        print(f"[Eval] goal_u0 dim={goal_u0.shape}  goal_u1 dim={goal_u1.shape}")

    r0 = _run_regime(
        actor=actor, eval_env_id=eval_env_id, fixed_u=0,
        episodes=episodes_per_regime, goal_state=goal_u0,
        state_fn=state_fn, temperature=temperature, verbose=verbose,
    )
    r1 = _run_regime(
        actor=actor, eval_env_id=eval_env_id, fixed_u=1,
        episodes=episodes_per_regime, goal_state=goal_u1,
        state_fn=state_fn, temperature=temperature, verbose=verbose,
    )

    s0 = r0["success_rate"]
    s1 = r1["success_rate"]
    mean_s = 0.5 * (s0 + s1)
    worst = min(s0, s1)
    gap = abs(s0 - s1)

    if verbose:
        print(f"[Eval] success_u0 = {s0:.3f}")
        print(f"[Eval] success_u1 = {s1:.3f}")
        print(f"[Eval] mean_success = {mean_s:.3f}")
        print(f"[Eval] worst_case_success = {worst:.3f}")
        print(f"[Eval] regime_gap = {gap:.3f}")

    return {
        "checkpoint": str(checkpoint),
        "eval_env_id": eval_env_id,
        "success_u0": s0,
        "success_u1": s1,
        "mean_success": mean_s,
        "worst_case_success": worst,
        "regime_gap": gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Actor-driven forced-U evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-env-id", type=str,
                        default="CausalContrastive-WindyCorridor-15x15-Lethal-v0")
    parser.add_argument("--episodes-per-regime", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for action sampling (lower=more greedy)")
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "actor_forced_u_eval.csv")
    args = parser.parse_args()

    row = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        eval_env_id=args.eval_env_id,
        episodes_per_regime=args.episodes_per_regime,
        temperature=args.temperature,
    )

    if not args.stdout_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row)

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)


if __name__ == "__main__":
    main()
