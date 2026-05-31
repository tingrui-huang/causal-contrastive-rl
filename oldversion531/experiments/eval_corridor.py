"""
Evaluate a trained corridor agent on WindyCorridor.

Loads a checkpoint produced by train_corridor.py, runs N episodes against the
SCM-wrapped env using the actor's greedy / sampled policy, and reports:
  * success rate (reached goal at (13, 1))
  * lava death rate
  * timeout rate
  * mean reward, mean steps
  * route classification (NEAR if traj never visited y=13, else FAR)

Optionally sweeps **forced wind regimes** to test robustness under shifted
confounder distributions::

    --forced-wind south       always south wind (lethal pressure on NEAR)
    --forced-wind still       always still (no wind, NEAR trivially short)
    --forced-wind notebook    default (.1,.1,.1,.1,.6) — same as training

Usage::

    python experiments/eval_corridor.py \\
        --checkpoint checkpoints/corridor_baseline_seed0.npz \\
        --num-episodes 100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from minigrid.core.actions import Actions

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from agents.goal_conditioned_actor_numpy import GoalConditionedActorNumpy
from envs import make_windy_corridor_scm
from envs.windy_corridor import GOAL_POS


VALID_ACTIONS = [
    int(Actions.left),
    int(Actions.right),
    int(Actions.forward),
    int(Actions.done),
]

FORCED_WIND_PRESETS: dict[str, tuple[float, float, float, float, float]] = {
    "south":    (0.0, 1.0, 0.0, 0.0, 0.0),
    "still":    (0.0, 0.0, 0.0, 0.0, 1.0),
    "notebook": (0.1, 0.1, 0.1, 0.1, 0.6),
}


def _load_checkpoint(path: Path) -> tuple[ContrastiveCriticNumpy, GoalConditionedActorNumpy, dict, np.ndarray]:
    ckpt = np.load(path, allow_pickle=True)
    cfg = json.loads(str(ckpt["config_json"]))
    state_dim = int(cfg["state_dim"])
    n_actions = int(cfg["n_actions"])

    critic = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=int(cfg["hidden"]),
        emb_dim=int(cfg["emb_dim"]),
        tau=float(cfg["tau"]),
        seed=int(cfg["seed"]),
    )
    critic.sa_W1[...] = ckpt["sa_W1"]; critic.sa_b1[...] = ckpt["sa_b1"]
    critic.sa_W2[...] = ckpt["sa_W2"]; critic.sa_b2[...] = ckpt["sa_b2"]
    critic.g_W1[...] = ckpt["g_W1"]; critic.g_b1[...] = ckpt["g_b1"]
    critic.g_W2[...] = ckpt["g_W2"]; critic.g_b2[...] = ckpt["g_b2"]

    actor = GoalConditionedActorNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=int(cfg["hidden"]),
        seed=int(cfg["seed"]) + 1,
    )
    actor.W1[...] = ckpt["actor_W1"]; actor.b1[...] = ckpt["actor_b1"]
    actor.W2[...] = ckpt["actor_W2"]; actor.b2[...] = ckpt["actor_b2"]

    goal_state = np.asarray(ckpt["goal_state"], dtype=np.float64)
    return critic, actor, cfg, goal_state


def _state_from_env(obs, env) -> np.ndarray:
    pos = tuple(int(v) for v in obs)
    return np.array([pos[0], pos[1], int(env.agent_dir)], dtype=np.float64)


def _classify_route(visited_ys: set[int]) -> str:
    return "far" if 13 in visited_ys else "near"


def _run_episodes(
    *,
    actor: GoalConditionedActorNumpy,
    goal_state: np.ndarray,
    num_episodes: int,
    max_steps: int,
    seed: int,
    temperature: float,
    forced_wind: str | None,
    verbose: bool,
) -> dict[str, Any]:
    wind_dist = FORCED_WIND_PRESETS[forced_wind] if forced_wind else None
    if verbose:
        wind_label = forced_wind if forced_wind else "default (.1,.1,.1,.1,.6)"
        print(f"  wind regime: {wind_label}")

    outcomes: Counter[str] = Counter()
    route_outcome: dict[str, Counter[str]] = {"near": Counter(), "far": Counter()}
    rewards: list[float] = []
    steps_list: list[int] = []

    for ep in range(num_episodes):
        env = make_windy_corridor_scm(wind_dist=wind_dist) if wind_dist else make_windy_corridor_scm()
        obs, info = env.reset(seed=seed + ep)
        rng = np.random.default_rng(seed + ep + 999_999)

        total_reward = 0.0
        visited_ys: set[int] = set()
        terminated = False
        truncated = False
        n_steps = 0

        for t in range(max_steps):
            state = _state_from_env(obs, env)
            visited_ys.add(int(state[1]))
            action = actor.sample_action(
                state, goal_state,
                valid_actions=VALID_ACTIONS,
                rng=rng,
                temperature=temperature,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            n_steps += 1
            if terminated or truncated:
                break

        final_pos = tuple(int(v) for v in obs)
        if final_pos == GOAL_POS:
            outcome = "goal"
        elif terminated:
            outcome = "lava"
        elif truncated:
            outcome = "timeout"
        else:
            outcome = "incomplete"

        outcomes[outcome] += 1
        route = _classify_route(visited_ys)
        route_outcome[route][outcome] += 1
        rewards.append(total_reward)
        steps_list.append(n_steps)

    return {
        "outcomes": outcomes,
        "route_outcome": route_outcome,
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "success_rate": outcomes["goal"] / num_episodes,
        "lava_rate": outcomes["lava"] / num_episodes,
        "timeout_rate": outcomes["timeout"] / num_episodes,
    }


def evaluate(
    *,
    checkpoint: Path,
    num_episodes: int = 100,
    max_steps: int = 200,
    seed: int = 0,
    temperature: float = 1.0,
    forced_wind: str | None = None,
    sweep: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    _, actor, cfg, goal_state = _load_checkpoint(checkpoint)
    if verbose:
        print(f"[Eval] checkpoint={checkpoint.name}  episodes={num_episodes}  temp={temperature}")
        print(f"[Eval] training γ={cfg['gamma']}  λ={cfg['lam']}  steps={cfg['num_steps']}")
        print(f"[Eval] goal_state={goal_state.tolist()}")

    regimes = ["notebook", "south", "still"] if sweep else [forced_wind]

    results: dict[str, dict] = {}
    for regime in regimes:
        if verbose:
            print(f"\n[Eval] regime={regime}")
        res = _run_episodes(
            actor=actor,
            goal_state=goal_state,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
            temperature=temperature,
            forced_wind=regime if regime != "notebook" else None,
            verbose=verbose,
        )
        results[regime or "default"] = res

        if verbose:
            print(f"  success={res['success_rate']:.2%}  lava={res['lava_rate']:.2%}  "
                  f"timeout={res['timeout_rate']:.2%}")
            print(f"  mean_reward={res['mean_reward']:+.2f}  mean_steps={res['mean_steps']:.1f}")
            for r, c in res["route_outcome"].items():
                if sum(c.values()) > 0:
                    print(f"  {r:<4} ({sum(c.values()):3d} eps): {dict(c)}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained corridor agent.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--forced-wind",
        choices=list(FORCED_WIND_PRESETS.keys()),
        default=None,
        help="Override wind to a fixed preset; omit for default notebook distribution.",
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Run notebook + south + still in sequence (robustness sweep).")
    args = parser.parse_args()

    evaluate(
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        temperature=args.temperature,
        forced_wind=args.forced_wind,
        sweep=args.sweep,
    )


if __name__ == "__main__":
    main()
