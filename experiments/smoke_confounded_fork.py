"""Smoke test for the ConfoundedFork env: env creation, oracle rollout, propensity."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

import envs  # noqa: F401

from envs.confounded_fork import ConfoundedForkEnv
from utils.collector import rollout_episode
from utils.offline_data import build_policy


def _summarize(env_id: str, fixed_u: int | None, num_episodes: int, seed: int = 0) -> None:
    actions_at_fork: Counter = Counter()
    reaches = 0
    lava_deaths = 0
    timeouts = 0
    lengths: list[int] = []
    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = gym.make(env_id)
        env.action_space.seed(ep_seed)
        policy = build_policy(env, collector_mode="oracle_eps",
                              oracle_epsilon=0.2, seed=ep_seed)
        traj = rollout_episode(env, policy, max_steps=200, seed=ep_seed)
        raw = env.unwrapped
        for tr in traj:
            obs = tr["obs"]
            cell = (int(obs["agent_pos"][0]), int(obs["agent_pos"][1]))
            if cell == (1, 8):  # fork cell
                actions_at_fork[int(tr["action"])] += 1

        last = traj[-1]
        nx, ny = last["next_obs"]["agent_pos"]
        if (int(nx), int(ny)) == raw.goal_pos():
            reaches += 1
        if last["reward"] < 0 or "lava" in str(last.get("info", {})).lower():
            pass
        # crude: short trajectories that didn't reach goal → lava death
        if (int(nx), int(ny)) != raw.goal_pos() and len(traj) < 200:
            lava_deaths += 1
        if len(traj) >= 200:
            timeouts += 1
        lengths.append(len(traj))

    print(f"[{env_id} fixed_u={fixed_u} eps={num_episodes}]")
    print(f"  reach_rate = {reaches}/{num_episodes} = {reaches/num_episodes:.2f}")
    print(f"  lava_or_other_failure = {lava_deaths}/{num_episodes}")
    print(f"  timeouts = {timeouts}/{num_episodes}")
    print(f"  mean_len = {np.mean(lengths):.1f}")
    print(f"  actions_at_fork (1,8): {dict(actions_at_fork)}")


def main() -> None:
    print("=== ConfoundedFork: oracle properties ===")
    print(f"walkable cells: {len(ConfoundedForkEnv.walkable_cells())}")
    print(f"hazard cells: {ConfoundedForkEnv.hazard_cells()}")
    print(f"walkable_for_regime(0) size: {len(ConfoundedForkEnv.walkable_for_regime(0))}")
    print(f"walkable_for_regime(1) size: {len(ConfoundedForkEnv.walkable_for_regime(1))}")
    print()

    # Confounded version
    print("=== Confounded (random U) with lethal lava ===")
    _summarize("CausalContrastive-ConfoundedFork-11x11-Lethal-v0", None, 30, seed=0)
    print()

    # Forced U=0
    print("=== fixed_u=0 (calm) with lethal lava ===")
    env_id_u0 = "CausalContrastive-ConfoundedFork-11x11-Lethal-Clean-v0"
    _summarize(env_id_u0, 0, 30, seed=0)
    print()

    print("[smoke] PASS")


if __name__ == "__main__":
    main()
