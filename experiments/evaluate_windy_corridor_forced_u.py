"""
Forced-U regime evaluation for WindyCorridor NumPy checkpoints.

This script measures *regime robustness* under explicit `fixed_u=0/1` overrides.
It is NOT the same as ordinary rollout success on the deployment environment.

Interpretation:
- use this script for `success_u0`, `success_u1`, `worst_case_success`, `regime_gap`
- do NOT use it as the sole metric for "clean env success rate"
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
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

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from utils.preprocess import extract_state, extract_state_oracle

ACTION_IDS = [int(Actions.left), int(Actions.right), int(Actions.forward)]
CSV_FIELDS = [
    "checkpoint",
    "train_env_id",
    "eval_env_id",
    "success_u0",
    "success_u1",
    "mean_success",
    "worst_case_success",
    "regime_gap",
]


def _load_numpy_checkpoint(path: Path) -> tuple[ContrastiveCriticNumpy, dict[str, Any]]:
    ckpt = np.load(path, allow_pickle=True)
    state_dim = int(ckpt["state_dim"])
    n_actions = int(ckpt["n_actions"])
    tau = float(ckpt["tau"])
    hidden = int(ckpt["W1"].shape[1])
    emb_dim = int(ckpt["W2"].shape[1])
    model = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=hidden,
        emb_dim=emb_dim,
        tau=tau,
        seed=0,
    )
    model.W1[...] = ckpt["W1"]
    model.b1[...] = ckpt["b1"]
    model.W2[...] = ckpt["W2"]
    model.b2[...] = ckpt["b2"]
    cfg = json.loads(str(ckpt["train_config_json"]))
    return model, cfg


def _goal_pos(raw) -> tuple[int, int]:
    goal_attr = raw.goal_pos
    if callable(goal_attr):
        return tuple(int(v) for v in goal_attr())
    return tuple(int(v) for v in goal_attr)


def _neighbors(
    cell: tuple[int, int],
    walkable: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    x, y = cell
    out: list[tuple[int, int]] = []
    for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if nxt in walkable:
            out.append(nxt)
    return out


def _shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    walkable: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        cell = q.popleft()
        if cell == goal:
            break
        for nxt in _neighbors(cell, walkable):
            if nxt not in parent:
                parent[nxt] = cell
                q.append(nxt)

    if goal not in parent:
        raise RuntimeError("Safe reference path could not reach goal.")

    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _desired_direction(src: tuple[int, int], dst: tuple[int, int]) -> int:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx == 1 and dy == 0:
        return 0
    if dx == 0 and dy == 1:
        return 1
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == -1:
        return 3
    raise ValueError(f"Non-adjacent move from {src} to {dst}")


def _safe_reference_action(env) -> int:
    raw = env.unwrapped
    if not hasattr(raw, "walkable_cells") or not hasattr(raw, "goal_pos"):
        raise RuntimeError("Safe topology reference requires walkable_cells() and goal_pos().")

    pos = tuple(int(v) for v in raw.agent_pos)
    goal = _goal_pos(raw)
    if pos == goal:
        return int(Actions.done)

    path = _shortest_path(pos, goal, raw.walkable_cells())
    if len(path) < 2:
        return int(Actions.done)

    target_dir = _desired_direction(path[0], path[1])
    if int(raw.agent_dir) == target_dir:
        return int(Actions.forward)
    if (target_dir - int(raw.agent_dir)) % 4 == 1:
        return int(Actions.right)
    return int(Actions.left)


def _reference_trajectory_from_fixed_u(
    *,
    env_id: str,
    fixed_u: int,
    seed: int,
    state_fn=None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    fn = state_fn or extract_state
    env = gym.make(env_id, render_mode="rgb_array", fixed_u=fixed_u, wind_strength=0.0)
    goal = _goal_pos(env.unwrapped)

    obs, info = env.reset(seed=seed)
    states = [fn(obs, info)]
    positions = [tuple(int(v) for v in env.unwrapped.agent_pos)]
    for _ in range(env.unwrapped.max_steps):
        action = _safe_reference_action(env)
        if action == int(Actions.done):
            break
        next_obs, _, terminated, truncated, info = env.step(action)
        states.append(fn(next_obs, info))
        positions.append(tuple(int(v) for v in env.unwrapped.agent_pos))
        obs = next_obs
        if terminated or truncated:
            break
    env.close()

    if positions[-1] != goal:
        raise RuntimeError("Safe reference trajectory did not reach the goal.")

    return np.stack(states, axis=0).astype(np.float64, copy=False), positions


def _score_action(
    model: ContrastiveCriticNumpy,
    state: np.ndarray,
    action_id: int,
    goal_bank: np.ndarray,
) -> float:
    s_batch = np.repeat(state[None, :], goal_bank.shape[0], axis=0)
    a_batch = np.full(goal_bank.shape[0], action_id, dtype=np.int64)
    h_sa, _ = model._embed(s_batch, a_batch)
    h_goal, _ = model._embed(goal_bank, a_batch)
    return float(np.max(np.sum(h_sa * h_goal, axis=1)))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _future_window_bank(
    ref_states: np.ndarray,
    *,
    ref_idx: int,
    lookahead_k: int,
    future_window: int,
) -> np.ndarray:
    start = min(ref_idx + lookahead_k, len(ref_states) - 1)
    stop = min(start + future_window + 1, len(ref_states))
    return ref_states[start:stop]


def _goal_bank(
    ref_states: np.ndarray,
    *,
    ref_idx: int,
    lookahead_k: int,
    future_window: int,
    goal_bank_mode: str,
) -> np.ndarray:
    if goal_bank_mode == "waypoint":
        return _future_window_bank(
            ref_states,
            ref_idx=ref_idx,
            lookahead_k=lookahead_k,
            future_window=future_window,
        )
    if goal_bank_mode == "final_only":
        return ref_states[-1:]
    raise ValueError(f"Unknown goal_bank_mode: {goal_bank_mode!r}")


def _reference_index(
    *,
    step_idx: int,
    state: np.ndarray,
    ref_states: np.ndarray,
    alignment_mode: str,
) -> int:
    if alignment_mode == "time":
        return min(step_idx, len(ref_states) - 1)
    if alignment_mode == "nearest":
        d2 = np.sum((ref_states - state[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))
    raise ValueError(f"Unknown alignment_mode: {alignment_mode!r}")


def _is_fatal_transition(*, env, pos_after: tuple[int, int], reward: float, terminated: bool) -> bool:
    if reward > 0:
        return False
    cell = env.unwrapped.grid.get(*pos_after)
    if cell is not None and cell.type == "lava":
        return True
    return bool(terminated)


def _plan_action(
    *,
    env,
    obs,
    model: ContrastiveCriticNumpy,
    ref_states: np.ndarray,
    ref_positions: list[tuple[int, int]],
    step_idx: int,
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_bank_mode: str,
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
    fatal_penalty: float,
    state_fn=None,
) -> int:
    fn = state_fn or extract_state
    info = {"confounder": int(getattr(env.unwrapped, "hidden_u", 0))}
    state = fn(obs, info).astype(np.float64, copy=False)
    best_value = -np.inf
    best_action = ACTION_IDS[0]

    for action_id in ACTION_IDS:
        for suffix in itertools.product(ACTION_IDS, repeat=max(0, plan_depth - 1)):
            seq = (action_id, *suffix)
            sim_env = copy.deepcopy(env)
            sim_obs = obs
            seq_value = 0.0
            for depth_idx, seq_action in enumerate(seq):
                sim_raw_before = sim_env.unwrapped
                pos_before = tuple(int(v) for v in sim_raw_before.agent_pos)
                sim_info = {"confounder": int(getattr(sim_raw_before, "hidden_u", 0))}
                sim_state = fn(sim_obs, sim_info).astype(np.float64, copy=False)
                sim_ref_idx = _reference_index(
                    step_idx=step_idx + depth_idx,
                    state=sim_state,
                    ref_states=ref_states,
                    alignment_mode=alignment_mode,
                )
                future_bank = _goal_bank(
                    ref_states,
                    ref_idx=sim_ref_idx,
                    lookahead_k=lookahead_k,
                    future_window=future_window,
                    goal_bank_mode=goal_bank_mode,
                )
                if goal_bank_mode == "waypoint":
                    progress_target_pos = ref_positions[
                        min(sim_ref_idx + lookahead_k, len(ref_positions) - 1)
                    ]
                elif goal_bank_mode == "final_only":
                    progress_target_pos = ref_positions[-1]
                else:
                    raise ValueError(f"Unknown goal_bank_mode: {goal_bank_mode!r}")
                seq_value += _score_action(model, sim_state, seq_action, future_bank)

                sim_next_obs, reward, sim_terminated, sim_truncated, _ = sim_env.step(seq_action)
                sim_raw_after = sim_env.unwrapped
                pos_after = tuple(int(v) for v in sim_raw_after.agent_pos)

                if _is_fatal_transition(
                    env=sim_env,
                    pos_after=pos_after,
                    reward=float(reward),
                    terminated=bool(sim_terminated),
                ):
                    seq_value -= fatal_penalty
                    break

                seq_value += progress_bonus * float(
                    _manhattan(pos_before, progress_target_pos)
                    - _manhattan(pos_after, progress_target_pos)
                )
                if seq_action == int(Actions.forward) and pos_after == pos_before:
                    seq_value -= collision_penalty
                elif seq_action in (int(Actions.left), int(Actions.right)):
                    seq_value -= turn_penalty
                if sim_terminated and reward > 0:
                    seq_value += success_bonus

                sim_obs = sim_next_obs
                if sim_terminated or sim_truncated:
                    break

            if seq_value > best_value:
                best_value = seq_value
                best_action = action_id

    return best_action


def _run_regime(
    *,
    model: ContrastiveCriticNumpy,
    eval_env_id: str,
    fixed_u: int,
    episodes: int,
    ref_states: np.ndarray,
    ref_positions: list[tuple[int, int]],
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_bank_mode: str,
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
    fatal_penalty: float,
    state_fn=None,
) -> float:
    successes = 0
    for ep in range(episodes):
        seed = 100 + fixed_u * 1000 + ep
        env = gym.make(eval_env_id, render_mode="rgb_array", fixed_u=fixed_u)
        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        while not done and steps < 120:
            best_action = _plan_action(
                env=env,
                obs=obs,
                model=model,
                ref_states=ref_states,
                ref_positions=ref_positions,
                step_idx=steps,
                lookahead_k=lookahead_k,
                future_window=future_window,
                alignment_mode=alignment_mode,
                goal_bank_mode=goal_bank_mode,
                plan_depth=plan_depth,
                collision_penalty=collision_penalty,
                turn_penalty=turn_penalty,
                progress_bonus=progress_bonus,
                success_bonus=success_bonus,
                fatal_penalty=fatal_penalty,
                state_fn=state_fn,
            )
            obs, reward, terminated, truncated, _ = env.step(best_action)
            done = terminated or truncated
            if reward > 0:
                successes += 1
            steps += 1
        env.close()
    return successes / float(episodes)


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    eval_env_id: str,
    episodes_per_regime: int,
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_bank_mode: str,
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
    fatal_penalty: float,
) -> dict[str, Any]:
    model, cfg = _load_numpy_checkpoint(checkpoint)
    is_oracle = cfg.get("oracle_state", False)
    state_fn = extract_state_oracle if is_oracle else None
    ref_states_u0, ref_positions_u0 = _reference_trajectory_from_fixed_u(
        env_id=eval_env_id,
        fixed_u=0,
        seed=int(cfg.get("seed", 0)),
        state_fn=state_fn,
    )
    ref_states_u1, ref_positions_u1 = _reference_trajectory_from_fixed_u(
        env_id=eval_env_id,
        fixed_u=1,
        seed=int(cfg.get("seed", 0)),
        state_fn=state_fn,
    )

    success_u0 = _run_regime(
        model=model,
        eval_env_id=eval_env_id,
        fixed_u=0,
        episodes=episodes_per_regime,
        ref_states=ref_states_u0,
        ref_positions=ref_positions_u0,
        lookahead_k=lookahead_k,
        future_window=future_window,
        alignment_mode=alignment_mode,
        goal_bank_mode=goal_bank_mode,
        plan_depth=plan_depth,
        collision_penalty=collision_penalty,
        turn_penalty=turn_penalty,
        progress_bonus=progress_bonus,
        success_bonus=success_bonus,
        fatal_penalty=fatal_penalty,
        state_fn=state_fn,
    )
    success_u1 = _run_regime(
        model=model,
        eval_env_id=eval_env_id,
        fixed_u=1,
        episodes=episodes_per_regime,
        ref_states=ref_states_u1,
        ref_positions=ref_positions_u1,
        lookahead_k=lookahead_k,
        future_window=future_window,
        alignment_mode=alignment_mode,
        goal_bank_mode=goal_bank_mode,
        plan_depth=plan_depth,
        collision_penalty=collision_penalty,
        turn_penalty=turn_penalty,
        progress_bonus=progress_bonus,
        success_bonus=success_bonus,
        fatal_penalty=fatal_penalty,
        state_fn=state_fn,
    )
    mean_success = 0.5 * (success_u0 + success_u1)
    worst_case_success = min(success_u0, success_u1)
    regime_gap = abs(success_u0 - success_u1)

    print(f"[Eval] success_u0 = {success_u0:.3f}")
    print(f"[Eval] success_u1 = {success_u1:.3f}")
    print(f"[Eval] mean_success = {mean_success:.3f}")
    print(f"[Eval] worst_case_success = {worst_case_success:.3f}")
    print(f"[Eval] regime_gap = {regime_gap:.3f}")

    return {
        "checkpoint": str(checkpoint),
        "train_env_id": cfg.get("env_id"),
        "eval_env_id": eval_env_id,
        "success_u0": success_u0,
        "success_u1": success_u1,
        "mean_success": mean_success,
        "worst_case_success": worst_case_success,
        "regime_gap": regime_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forced-U evaluation for WindyCorridor checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-env-id", type=str, default="CausalContrastive-WindyCorridor-15x15-v0")
    parser.add_argument("--episodes-per-regime", type=int, default=20)
    parser.add_argument("--lookahead-k", type=int, default=4)
    parser.add_argument("--future-window", type=int, default=2)
    parser.add_argument(
        "--goal-bank-mode",
        type=str,
        default="waypoint",
        choices=["waypoint", "final_only"],
    )
    parser.add_argument(
        "--alignment-mode",
        type=str,
        default="time",
        choices=["time", "nearest"],
    )
    parser.add_argument("--plan-depth", type=int, default=2)
    parser.add_argument("--collision-penalty", type=float, default=2.0)
    parser.add_argument("--turn-penalty", type=float, default=0.05)
    parser.add_argument("--progress-bonus", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--fatal-penalty", type=float, default=999.0)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "windy_corridor_forced_u_eval.csv",
    )
    args = parser.parse_args()

    row = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        eval_env_id=args.eval_env_id,
        episodes_per_regime=args.episodes_per_regime,
        lookahead_k=args.lookahead_k,
        future_window=args.future_window,
        alignment_mode=args.alignment_mode,
        goal_bank_mode=args.goal_bank_mode,
        plan_depth=args.plan_depth,
        collision_penalty=args.collision_penalty,
        turn_penalty=args.turn_penalty,
        progress_bonus=args.progress_bonus,
        success_bonus=args.success_bonus,
        fatal_penalty=args.fatal_penalty,
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
