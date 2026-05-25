"""Coverage diagnostic: compare (cell, action) coverage across collector modes.

For each collector_mode, run N episodes and report:
- # unique cells visited
- # unique (cell, action) pairs
- # unique (cell, action, U) triples
- per-regime trajectory length and reach rate (did the agent step on the env goal?)

This is the evidence that ``multigoal_oracle`` widens the (s, a) distribution to
something closer to D4RL AntMaze-play, while ``oracle_eps`` stays on two narrow
corridor bands.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

import envs  # noqa: F401
from utils.collector import rollout_episode
from utils.offline_data import build_policy

GAMMA = 0.95
MOVE_ACTIONS = {0, 1, 2}  # MiniGrid: left, right, forward


def _geometric_offset(rng: np.random.Generator, max_val: int) -> int:
    off = int(rng.geometric(1.0 - GAMMA))
    return min(max(off, 1), max_val)


def _entropy(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def collect_coverage(
    env_id: str,
    collector_mode: str,
    *,
    num_episodes: int,
    oracle_epsilon: float,
    max_steps: int,
    seed: int,
    hindsight_samples_per_anchor: int = 4,
) -> dict:
    cell_visits: dict[tuple[int, int], int] = {}
    cell_move_action: set[tuple[int, int, int]] = set()
    lengths: list[int] = []
    reached: list[int] = []
    regimes: list[int] = []
    episode_cells: list[list[tuple[int, int]]] = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = gym.make(env_id, render_mode="rgb_array")
        env.action_space.seed(ep_seed)
        policy = build_policy(
            env,
            collector_mode=collector_mode,
            oracle_epsilon=oracle_epsilon,
            seed=ep_seed,
        )
        traj = rollout_episode(env, policy, max_steps=max_steps, seed=ep_seed)

        raw = env.unwrapped
        u = int(getattr(raw, "hidden_u", 0))
        regimes.append(u)
        lengths.append(len(traj))

        goal_pos = raw.goal_pos() if callable(raw.goal_pos) else tuple(int(v) for v in raw.goal_pos)
        ep_cell_list: list[tuple[int, int]] = []
        for tr in traj:
            obs = tr["obs"]
            cell = (int(obs["agent_pos"][0]), int(obs["agent_pos"][1]))
            cell_visits[cell] = cell_visits.get(cell, 0) + 1
            ep_cell_list.append(cell)
            a = int(tr["action"])
            if a in MOVE_ACTIONS:
                cell_move_action.add((cell[0], cell[1], a))
        episode_cells.append(ep_cell_list)
        last = traj[-1]
        nx, ny = last["next_obs"]["agent_pos"]
        reached.append(1 if (int(nx), int(ny)) == goal_pos else 0)

    # Hindsight s_future sampling — replays the geometric-offset relabeling
    # used by _build_batch to show what cells actually become "goals" for the critic.
    rng = np.random.default_rng(seed + 9999)
    hindsight_visits: dict[tuple[int, int], int] = {}
    for cells in episode_cells:
        T = len(cells)
        if T < 2:
            continue
        for t in range(T - 1):
            for _ in range(hindsight_samples_per_anchor):
                max_off = T - t - 1
                off = _geometric_offset(rng, max_off)
                c = cells[t + off]
                hindsight_visits[c] = hindsight_visits.get(c, 0) + 1

    n_walkable = len(env.unwrapped.walkable_cells())
    visit_counts = np.array(list(cell_visits.values()), dtype=np.int64)
    hindsight_counts = np.array(list(hindsight_visits.values()), dtype=np.int64)
    move_max = n_walkable * 3
    return {
        "mode": collector_mode,
        "n_episodes": num_episodes,
        "total_steps": int(sum(lengths)),
        "mean_len": float(np.mean(lengths)),
        "unique_cells": len(cell_visits),
        "cell_cov_pct": round(100.0 * len(cell_visits) / n_walkable, 1),
        "move_cell_act_cov_pct": round(100.0 * len(cell_move_action) / move_max, 1),
        "visit_entropy_nats": round(_entropy(visit_counts), 3),
        "max_visit_share_pct": round(100.0 * visit_counts.max() / visit_counts.sum(), 1),
        "hindsight_cells": len(hindsight_visits),
        "hindsight_cov_pct": round(100.0 * len(hindsight_visits) / n_walkable, 1),
        "hindsight_entropy_nats": round(_entropy(hindsight_counts), 3),
        "hindsight_max_share_pct": round(100.0 * hindsight_counts.max() / hindsight_counts.sum(), 1),
        "u0_eps": sum(1 for u in regimes if u == 0),
        "u1_eps": sum(1 for u in regimes if u == 1),
        "reach_u0": round(float(np.mean([r for r, u in zip(reached, regimes) if u == 0]) if any(u == 0 for u in regimes) else float("nan")), 2),
        "reach_u1": round(float(np.mean([r for r, u in zip(reached, regimes) if u == 1]) if any(u == 1 for u in regimes) else float("nan")), 2),
    }


def main() -> None:
    env_id = "CausalContrastive-WindyCorridor-15x15-Lethal-v0"
    num_episodes = 50
    max_steps = 500
    seed = 0

    rows = []
    for mode, eps in [("oracle_eps", 0.2), ("multigoal_oracle", 0.0), ("random", 0.0)]:
        r = collect_coverage(
            env_id,
            mode,
            num_episodes=num_episodes,
            oracle_epsilon=eps,
            max_steps=max_steps,
            seed=seed,
        )
        rows.append(r)

    keys = [
        "mode", "n_episodes", "total_steps", "mean_len",
        "cell_cov_pct", "move_cell_act_cov_pct",
        "visit_entropy_nats", "max_visit_share_pct",
        "hindsight_cov_pct", "hindsight_entropy_nats", "hindsight_max_share_pct",
        "u0_eps", "u1_eps", "reach_u0", "reach_u1",
    ]
    widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in keys}
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in keys))


if __name__ == "__main__":
    main()
