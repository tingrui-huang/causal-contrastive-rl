"""Empirical action propensity P(x | s) for the Thm 2 recursive sampler.

Aggregated by cell (POS_X_IDX, POS_Y_IDX) — fine-grained enough for the
WindyCorridor since the relevant state attribute is position.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from utils.domain_knowledge import POS_X_IDX, POS_Y_IDX


def _cell_of(s: np.ndarray) -> tuple[int, int]:
    return int(round(s[POS_X_IDX])), int(round(s[POS_Y_IDX]))


def build_propensity_table(
    episode_states: list[np.ndarray],
    episode_actions: list[np.ndarray],
    n_actions: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Return dict[cell] -> normalized action distribution.

    episode_actions[ep] has length T (one action per transition); episode_states[ep]
    has length T+1 after the terminal-state fix. We pair states[:T] with actions.
    """
    counts: dict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(n_actions, dtype=np.float64)
    )
    for states, actions in zip(episode_states, episode_actions):
        T = len(actions)
        for t in range(T):
            cell = _cell_of(states[t])
            counts[cell][int(actions[t])] += 1.0

    table: dict[tuple[int, int], np.ndarray] = {}
    for cell, c in counts.items():
        total = float(c.sum())
        if total > 0:
            table[cell] = c / total
    return table


def propensity(
    s: np.ndarray,
    a: int,
    table: dict[tuple[int, int], np.ndarray],
    n_actions: int,
    fallback_uniform: bool = True,
) -> float:
    cell = _cell_of(s)
    if cell not in table:
        return 1.0 / n_actions if fallback_uniform else 0.0
    return float(table[cell][int(a)])


def sample_action_from_propensity(
    s: np.ndarray,
    table: dict[tuple[int, int], np.ndarray],
    n_actions: int,
    rng: np.random.Generator,
) -> int:
    cell = _cell_of(s)
    if cell not in table:
        return int(rng.integers(0, n_actions))
    return int(rng.choice(n_actions, p=table[cell]))


def fallback_diagnostics(
    table: dict[tuple[int, int], np.ndarray],
    episode_states: list[np.ndarray],
) -> dict[str, float]:
    """How often would lookups against this table fall through to the uniform fallback?"""
    total = 0
    missing = 0
    for states in episode_states:
        for s in states:
            total += 1
            if _cell_of(s) not in table:
                missing += 1
    return {
        "lookups": total,
        "missing": missing,
        "fallback_rate": missing / max(total, 1),
        "unique_cells": len(table),
    }
