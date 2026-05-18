"""Multi-goal oracle policy for data collection (D4RL AntMaze-play analog).

Instead of always navigating to the env's fixed goal, this policy picks random
sub-goals from the walkable cells and navigates between them, re-sampling a new
sub-goal whenever the current one is reached or becomes unreachable. This widens
(s, a, s_future) coverage to match the original contrastive RL paper's offline
data assumption (D4RL AntMaze-*-play datasets).

U-awareness is preserved: under u=1, the sub-goal sampler and BFS both use the
regime-restricted walkable set, so each action still depends on U (this is the
confounding mechanism we want the critic to see).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from utils.offline_data import (
    _desired_dir,
    _safe_walkable_for_regime,
    _shortest_path_next_cell,
    _turn_toward,
)


def build_multigoal_oracle_policy(
    env: gym.Env,
    *,
    epsilon: float,
    seed: int,
    exclude_env_goal: bool = True,
    min_subgoal_manhattan: int = 2,
):
    """Oracle that walks between random sub-goals (D4RL-play style).

    Parameters
    ----------
    epsilon
        Probability of taking a uniformly random action at each step. Multi-goal
        sampling already provides coverage diversity, so ``epsilon=0.0`` is a
        reasonable default; raise it only if extra perturbation is desired.
    exclude_env_goal
        If True, the env's true goal is never picked as a sub-goal. Without this,
        episodes terminate as soon as the oracle reaches the goal and trajectory
        length collapses.
    min_subgoal_manhattan
        Minimum Manhattan distance from current position when sampling a sub-goal,
        to avoid trivial 1-cell sub-goals.
    """
    rng = np.random.default_rng(seed)
    state: dict[str, tuple[int, int] | None] = {"subgoal": None}

    def _resample_subgoal(
        cur: tuple[int, int],
        effective_walkable: set[tuple[int, int]],
        forbidden: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        far = [
            c for c in effective_walkable
            if c not in forbidden
            and c != cur
            and abs(c[0] - cur[0]) + abs(c[1] - cur[1]) >= min_subgoal_manhattan
        ]
        if not far:
            far = [c for c in effective_walkable if c not in forbidden and c != cur]
        if not far:
            return None
        far.sort()
        return far[int(rng.integers(0, len(far)))]

    def policy(_obs):
        raw = env.unwrapped
        cur = tuple(int(v) for v in raw.agent_pos)

        if rng.random() < epsilon:
            return env.action_space.sample()

        hidden_u = int(getattr(raw, "hidden_u", 0))
        walkable = raw.walkable_cells()
        effective_walkable = _safe_walkable_for_regime(walkable, hidden_u, raw)

        forbidden: set[tuple[int, int]] = set()
        if exclude_env_goal and hasattr(raw, "goal_pos"):
            goal_attr = raw.goal_pos
            gp = goal_attr() if callable(goal_attr) else tuple(int(v) for v in goal_attr)
            forbidden.add(gp)

        need_resample = (
            state["subgoal"] is None
            or state["subgoal"] == cur
            or state["subgoal"] not in effective_walkable
        )
        if need_resample:
            state["subgoal"] = _resample_subgoal(cur, effective_walkable, forbidden)
            if state["subgoal"] is None:
                return env.action_space.sample()

        dst = _shortest_path_next_cell(cur, state["subgoal"], effective_walkable)
        if dst == cur:
            state["subgoal"] = None
            return env.action_space.sample()

        target_dir = _desired_dir(cur, dst)
        return _turn_toward(int(raw.agent_dir), target_dir)

    return policy
