from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions
from collections import deque

from buffers.replay_buffer import ReplayBuffer
from configs.training_defaults import TRAIN_MAX_EPISODE_STEPS, TRAIN_REPLAY_CAPACITY
from utils.collector import rollout_episode
from utils.contrastive_sampling import DEFAULT_K
from utils.preprocess import extract_state, extract_state_oracle, get_u_from_info


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


def _safe_walkable_for_regime(
    walkable: set[tuple[int, int]],
    hidden_u: int,
    env,
) -> set[tuple[int, int]]:
    """Return a possibly restricted walkable set based on the hidden regime.

    u=0 (calm): full walkable set — oracle takes the short risky route via x=11.
    u=1 (windy): remove the x=11 vertical passage (y=6..13) so BFS is forced
                  through the safe x=3 -> x=7 route.  The final approach cells
                  (11,5), (12,5), (13,5) stay walkable so the goal is reachable.
    """
    if hidden_u == 0:
        return walkable

    danger_zone = {(11, y) for y in range(6, 14)}
    restricted = walkable - danger_zone
    return restricted


def _oracle_target_cell(env: gym.Env) -> tuple[int, int]:
    raw = env.unwrapped
    if hasattr(raw, "walkable_cells") and hasattr(raw, "goal_pos"):
        walkable = raw.walkable_cells()
        start = tuple(int(v) for v in raw.agent_pos)
        goal_attr = raw.goal_pos
        goal = goal_attr() if callable(goal_attr) else tuple(int(v) for v in goal_attr)
        if start == goal:
            return start
        hidden_u = int(getattr(raw, "hidden_u", 0))
        effective_walkable = _safe_walkable_for_regime(walkable, hidden_u, raw)
        return _shortest_path_next_cell(start, goal, effective_walkable)

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


def _shortest_path_next_cell(
    start: tuple[int, int],
    goal: tuple[int, int],
    walkable: set[tuple[int, int]],
) -> tuple[int, int]:
    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

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
        return start

    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    if len(path) < 2:
        return start
    return path[1]


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
    if collector_mode == "multigoal_oracle":
        from utils.multigoal_oracle import build_multigoal_oracle_policy
        return build_multigoal_oracle_policy(
            env,
            epsilon=oracle_epsilon,
            seed=seed,
        )
    raise ValueError(f"Unknown collector_mode: {collector_mode!r}")


def processed_transition(trans: dict, *, state_fn=None) -> dict:
    fn = state_fn or extract_state
    return {
        "state": fn(trans["obs"], trans.get("info")),
        "next_state": fn(trans["next_obs"], trans.get("next_info")),
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
    state_fn=None,
):
    """Roll out multiple episodes; positives stay within-episode, negatives use a global buffer."""
    fn = state_fn or extract_state
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity=replay_capacity, seed=seed)
    regime_buffers: dict[int, ReplayBuffer] = {}
    k = DEFAULT_K
    trajectories: list[list[dict]] = []
    episode_states: list[np.ndarray] = []
    episode_actions: list[np.ndarray] = []
    episode_regimes: list[int] = []
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

        ep_regime = int(getattr(env.unwrapped, "hidden_u", 0))
        episode_regimes.append(ep_regime)

        if ep_regime not in regime_buffers:
            regime_buffers[ep_regime] = ReplayBuffer(capacity=replay_capacity, seed=seed + ep_regime)

        for trans in trajectory:
            pt = processed_transition(trans, state_fn=fn)
            buffer.add(pt)
            regime_buffers[ep_regime].add(pt)

        states = np.stack(
            [fn(tr["obs"], tr.get("info")) for tr in trajectory], axis=0
        )
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
        episode_regimes,
        regime_buffers,
    )
