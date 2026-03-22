"""
Phase 5 smoke: build contrastive tuples and print required DEV_SPEC log lines.

Settings: ``K=2``, negative = buffer ``state``, anchors uniform on valid ``t``.
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

from buffers.replay_buffer import ReplayBuffer
from utils.collector import rollout_episode
from utils.contrastive_sampling import DEFAULT_K, build_contrastive_batch
from utils.preprocess import extract_state


def build_random_policy(env):
    def policy(_obs):
        return env.action_space.sample()

    return policy


def processed_transition(trans: dict) -> dict:
    return {
        "state": extract_state(trans["obs"]),
        "next_state": extract_state(trans["next_obs"]),
        "action": trans["action"],
        "reward": trans["reward"],
        "done": trans["done"],
    }


def main() -> None:
    rng = np.random.default_rng(0)

    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    policy = build_random_policy(env)
    trajectory = rollout_episode(env, policy, max_steps=500, seed=0)

    buffer = ReplayBuffer(capacity=10000)
    for trans in trajectory:
        buffer.add(processed_transition(trans))

    T = len(trajectory)
    k = DEFAULT_K
    if T < k + 1:
        raise RuntimeError(
            f"Need trajectory length >= k+1={k+1}, got T={T}. Increase max_steps or change seed."
        )

    batch_size = min(16, T - k)
    batch = build_contrastive_batch(
        trajectory,
        buffer,
        k=k,
        batch_size=batch_size,
        rng=rng,
    )

    # Shape / finiteness checks
    assert batch["s"].shape == batch["s_pos"].shape == batch["s_neg"].shape
    assert np.isfinite(batch["s"]).all() and np.isfinite(batch["s_pos"]).all()
    assert np.isfinite(batch["s_neg"]).all()

    # DEV_SPEC: exact lines (verbatim)
    print("[Contrastive] positive sampled from t+k")
    print("[Contrastive] negative sampled randomly")


if __name__ == "__main__":
    main()
