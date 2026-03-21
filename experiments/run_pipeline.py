"""
Phase 4 — integrated pipeline: env → collector → preprocess → replay buffer.

Documented behavior:
- One rollout: until the environment reports ``done`` or ``max_steps`` is reached
  (whichever comes first), random policy. Default ``max_steps=500``.
- Each stored item is a **processed** transition: ``state`` / ``next_state`` are
  NumPy vectors from ``utils.preprocess.extract_state`` applied to ``obs`` /
  ``next_obs`` (not raw dict observations).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401 — register MiniGrid envs
import numpy as np

from buffers.replay_buffer import ReplayBuffer
from utils.collector import rollout_episode
from utils.preprocess import extract_state


def build_random_policy(env):
    def policy(_obs):
        return env.action_space.sample()

    return policy


def processed_transition(trans: dict) -> dict:
    """Map one raw transition to processed vectors for the buffer."""
    return {
        "state": extract_state(trans["obs"]),
        "next_state": extract_state(trans["next_obs"]),
        "action": trans["action"],
        "reward": trans["reward"],
        "done": trans["done"],
    }


def main() -> None:
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    policy = build_random_policy(env)

    trajectory = rollout_episode(env, policy, max_steps=500, seed=0)

    buffer: ReplayBuffer = ReplayBuffer(capacity=10000)
    first_state: np.ndarray | None = None
    for trans in trajectory:
        item = processed_transition(trans)
        if first_state is None:
            first_state = item["state"]
        buffer.add(item)

    if first_state is None or len(buffer) == 0:
        raise RuntimeError("Empty trajectory or buffer; cannot run Phase 4 checks.")

    state_shape = first_state.shape
    sample_vals = first_state[:5].astype(np.float64)
    sample_str = " ".join(f"{float(v):.6f}" for v in sample_vals)

    requested_batch = 8
    batch_size = min(requested_batch, len(buffer))
    batch = buffer.sample(batch_size)

    for row in batch:
        if row["state"].shape != state_shape or row["next_state"].shape != state_shape:
            raise ValueError("Inconsistent processed state shapes inside batch.")
        if not np.isfinite(row["state"]).all() or not np.isfinite(row["next_state"]).all():
            raise ValueError("Non-finite values in buffered processed states.")

    traj_len = len(trajectory)

    print(f"[Pipeline] trajectory length = {traj_len}")
    print(f"[Pipeline] processed state shape = {tuple(int(x) for x in state_shape)}")
    print(f"[Pipeline] first processed state sample = {sample_str}")
    print(f"[Pipeline] buffer size = {len(buffer)}")
    print(f"[Pipeline] sample batch size = {len(batch)}")


if __name__ == "__main__":
    main()
