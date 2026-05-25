"""Scalar value function V_f(c) derived from the contrastive critic.

C1 definition (per implementation plan):
    V_f(c) = max over (a, g) of min(f1(state_c, a, state_g), f2(state_c, a, state_g))

Used by the Thm 2 recursive sampler to pick the argmin pessimistic neighbor.

For our WindyCorridor with ~41 walkable cells and 7 actions, the table costs
41 cells * 41 candidate goals * 7 actions = ~12K critic forward passes per
refresh — fast enough to recompute every ~200 training steps.
"""
from __future__ import annotations

import numpy as np


def compute_v_f_table(
    critic1,
    critic2,
    walkable_cells: list[tuple[int, int]],
    pos_index: dict[tuple[int, int], np.ndarray],
) -> dict[tuple[int, int], float]:
    """Build V_f(c) for every walkable cell with at least one observed state.

    Cells absent from pos_index (never visited) are omitted from the table —
    callers should fall back to a default value.

    For each cell c, picks one representative state vector from pos_index[c]
    (the first one) to avoid blowing up the cost across multiple per-cell
    state instances. The position-aware state representation makes this safe.
    """
    cells_with_data = [c for c in walkable_cells if c in pos_index]
    if not cells_with_data:
        return {}

    goal_states = np.stack([pos_index[c][0] for c in cells_with_data], axis=0)
    n_goals = goal_states.shape[0]

    table: dict[tuple[int, int], float] = {}
    for c in cells_with_data:
        s_c = pos_index[c][0]
        s_batch = np.repeat(s_c[None, :], n_goals, axis=0)  # (n_goals, state_dim)

        scores1 = critic1.score_actions(s_batch, goal_states)  # (n_goals, n_actions)
        scores2 = critic2.score_actions(s_batch, goal_states)
        twin_min = np.minimum(scores1, scores2)                # twin Q clamp
        table[c] = float(np.max(twin_min))                      # max over (a, g)

    return table


def lookup_v_f(
    cell: tuple[int, int],
    table: dict[tuple[int, int], float],
    fallback: float = 0.0,
) -> float:
    return table.get(cell, fallback)


def v_f_table_summary(table: dict[tuple[int, int], float]) -> dict[str, float]:
    if not table:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    vals = np.array(list(table.values()), dtype=np.float64)
    return {
        "n": len(table),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
    }
