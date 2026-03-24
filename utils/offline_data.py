from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions

from buffers.replay_buffer import ReplayBuffer
from configs.training_defaults import TRAIN_MAX_EPISODE_STEPS, TRAIN_REPLAY_CAPACITY
from utils.collector import rollout_episode
from utils.contrastive_sampling import DEFAULT_K
from utils.preprocess import extract_state


def build_random_policy(env: gym.Env):
    def policy(_obs):
        return env.action_space.sample()

    return policy


def _desired_dir(src: tuple[int, int], dst: tuple[int, int]) -> int:
    sx, sy = src
    dx, dy = dst
    if dx == sx + 1 and dy == sy:
        return 0  # right
    if dx == sx and dy == sy + 1:
        return 1  # down
    if dx == sx - 1 and dy == sy:
        return 2  # left
    if dx == sx and dy == sy - 1:
        return 3  # up
    raise ValueError(f"Non-adjacent move requested: {src} -> {dst}")


def _turn_toward(cur_dir: int, target_dir: int) -> int:
    if cur_dir == target_dir:
        return int(Actions.forward)
    if (cur_dir - 1) % 4 == target_dir:
        return int(Actions.left)
    if (cur_dir + 1) % 4 == target_dir:
        return int(Actions.right)
    return int(Actions.left)


def _oracle_target_cell(env: gym.Env) -> tuple[int, int]:
    raw = env.unwrapped
    ax, ay = raw.agent_pos
    cx = raw.width // 2
    fork_row = int(raw._fork_row)
    u = int(raw.hidden_u)
    branch_x = cx - 1 if u == 0 else cx + 1

    # Stay on the center trunk until the cell just below the blocked fork.
    if ay > fork_row + 1:
        return (cx, ay - 1)

    # Bypass the blocked center fork cell via the regime-selected side corridor.
    if ay == fork_row + 1 and ax == cx:
        return (branch_x, ay)
    if ax == branch_x and ay > fork_row - 1:
        return (ax, ay - 1)

    # Rejoin the center spine above the blocked fork.
    if ay == fork_row - 1 and ax == branch_x:
        return (cx, ay)

    if ay > 1:
        return (cx, ay - 1)
    return (ax, ay)


def build_oracle_eps_policy(
    env: gym.Env,
    *,
    epsilon: float,
    seed: int,
):
    rng = np.random.default_rng(seed)

    def policy(_obs):
        if rng.random() < epsilon:
            return env.action_space.sample()

        raw = env.unwrapped
        src = tuple(raw.agent_pos)
        dst = _oracle_target_cell(env)
        if src == dst:
            return env.action_space.sample()
        target_dir = _desired_dir(src, dst)
        return _turn_toward(int(raw.agent_dir), target_dir)

    return policy


def build_policy(
    env: gym.Env,
    *,
    collector_mode: str,
    oracle_epsilon: float,
    seed: int,
):
    if collector_mode == "random":
        return build_random_policy(env)
    if collector_mode == "oracle_eps":
        return build_oracle_eps_policy(
            env,
            epsilon=oracle_epsilon,
            seed=seed,
        )
    raise ValueError(f"Unknown collector_mode: {collector_mode!r}")


def processed_transition(trans: dict) -> dict:
    return {
        "state": extract_state(trans["obs"]),
        "next_state": extract_state(trans["next_obs"]),
        "action": trans["action"],
        "reward": trans["reward"],
        "done": trans["done"],
    }


def collect_episodes(
    seed: int,
    env_id: str,
    num_episodes: int,
    *,
    positive_window: int,
    collector_mode: str,
    oracle_epsilon: float,
    max_episode_steps: int = TRAIN_MAX_EPISODE_STEPS,
    replay_capacity: int = TRAIN_REPLAY_CAPACITY,
):
    """Roll out multiple episodes; positives stay within-episode, negatives use a global buffer."""
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity=replay_capacity, seed=seed)
    k = DEFAULT_K
    trajectories: list[list[dict]] = []
    episode_states: list[np.ndarray] = []
    episode_actions: list[np.ndarray] = []
    valid_anchors: list[tuple[int, int]] = []
    n_actions: int | None = None
    state_dim: int | None = None

    for ep_idx in range(num_episodes):
        ep_seed = seed + ep_idx
        env = gym.make(env_id, render_mode="rgb_array")
        env.action_space.seed(ep_seed)
        if n_actions is None:
            n_actions = int(env.action_space.n)
        policy = build_policy(
            env,
            collector_mode=collector_mode,
            oracle_epsilon=oracle_epsilon,
            seed=ep_seed,
        )
        trajectory = rollout_episode(
            env,
            policy,
            max_steps=max_episode_steps,
            seed=ep_seed,
        )
        trajectories.append(trajectory)

        for trans in trajectory:
            buffer.add(processed_transition(trans))

        states = np.stack([extract_state(tr["obs"]) for tr in trajectory], axis=0)
        actions = np.array([tr["action"] for tr in trajectory], dtype=np.int64)
        episode_states.append(states)
        episode_actions.append(actions)

        if state_dim is None:
            state_dim = int(states.shape[1])

        T = len(trajectory)
        num_valid = T - k - positive_window
        for t in range(max(0, num_valid)):
            valid_anchors.append((ep_idx, t))

    if not valid_anchors:
        raise RuntimeError(
            f"No valid anchors across {num_episodes} episode(s). Need some episode with "
            f"T >= k+positive_window+1={k+positive_window+1}. Change seed/env or max steps."
        )

    total_steps = int(sum(len(traj) for traj in trajectories))
    return (
        trajectories,
        episode_states,
        episode_actions,
        valid_anchors,
        buffer,
        int(state_dim),
        int(n_actions),
        total_steps,
        k,
        rng,
    )
