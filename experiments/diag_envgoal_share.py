"""Targeted diagnostic: how much hindsight-goal mass does the env goal (13,5) get?

Runs multigoal_oracle in both `exclude_env_goal` modes and reports for each:
- top-5 cells by hindsight share
- the env goal's hindsight share specifically

This answers: "did flipping exclude_env_goal actually change the (13,5) share,
or did it just shuffle which cell happens to be the most-visited?"
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
from utils.multigoal_oracle import build_multigoal_oracle_policy

GAMMA = 0.95


def _geometric_offset(rng: np.random.Generator, max_val: int) -> int:
    off = int(rng.geometric(1.0 - GAMMA))
    return min(max(off, 1), max_val)


def run(exclude_env_goal: bool, *, num_episodes: int = 50, max_steps: int = 500, seed: int = 0):
    episode_cells: list[list[tuple[int, int]]] = []
    reach = 0
    env_goal = None

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = gym.make("CausalContrastive-WindyCorridor-15x15-Lethal-v0", render_mode="rgb_array")
        env.action_space.seed(ep_seed)
        policy = build_multigoal_oracle_policy(
            env, epsilon=0.0, seed=ep_seed, exclude_env_goal=exclude_env_goal
        )
        traj = rollout_episode(env, policy, max_steps=max_steps, seed=ep_seed)
        raw = env.unwrapped
        if env_goal is None:
            env_goal = raw.goal_pos() if callable(raw.goal_pos) else tuple(int(v) for v in raw.goal_pos)
        cells = [(int(tr["obs"]["agent_pos"][0]), int(tr["obs"]["agent_pos"][1])) for tr in traj]
        # Mirror the collect_episodes fix: include the terminal state (next_obs
        # of the final transition) so hindsight relabel can sample it.
        if traj:
            last_next = traj[-1]["next_obs"]
            cells.append((int(last_next["agent_pos"][0]), int(last_next["agent_pos"][1])))
        episode_cells.append(cells)
        last = traj[-1]
        nx, ny = last["next_obs"]["agent_pos"]
        if (int(nx), int(ny)) == env_goal:
            reach += 1

    # Replay hindsight relabel
    rng = np.random.default_rng(seed + 9999)
    hindsight_counts: dict[tuple[int, int], int] = {}
    samples_per = 4
    for cells in episode_cells:
        T = len(cells)
        if T < 2:
            continue
        for t in range(T - 1):
            for _ in range(samples_per):
                max_off = T - t - 1
                off = _geometric_offset(rng, max_off)
                c = cells[t + off]
                hindsight_counts[c] = hindsight_counts.get(c, 0) + 1

    total = sum(hindsight_counts.values())
    sorted_cells = sorted(hindsight_counts.items(), key=lambda kv: -kv[1])
    env_goal_share = hindsight_counts.get(env_goal, 0) / total * 100

    print(f"\n=== exclude_env_goal={exclude_env_goal} ===")
    print(f"reach_rate = {reach}/{num_episodes} = {reach/num_episodes:.2f}")
    print(f"mean_episode_len = {np.mean([len(c) for c in episode_cells]):.1f}")
    print(f"total_hindsight_samples = {total}")
    print(f"env_goal {env_goal} hindsight share = {env_goal_share:.2f}%")
    print(f"top 5 cells by hindsight share:")
    for cell, cnt in sorted_cells[:5]:
        pct = cnt / total * 100
        marker = "  ← env goal" if cell == env_goal else ""
        print(f"  {cell}: {cnt:>5d} samples = {pct:.2f}%{marker}")


def main() -> None:
    run(exclude_env_goal=True)
    run(exclude_env_goal=False)


if __name__ == "__main__":
    main()
