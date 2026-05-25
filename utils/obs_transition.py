"""Empirical observational transition table P_obs(s' | s, a) for the Thm 2 sampler.

For each observed (cell, action) pair, store the list of next_state vectors that
appeared in the dataset. Sampling = uniform draw from that list (= empirical p).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from utils.domain_knowledge import POS_X_IDX, POS_Y_IDX


def _cell_of(s: np.ndarray) -> tuple[int, int]:
    return int(round(s[POS_X_IDX])), int(round(s[POS_Y_IDX]))


def build_obs_transition_table(
    trajectories: list[list[dict]],
    state_fn,
) -> dict[tuple[tuple[int, int], int], list[np.ndarray]]:
    """For each (cell, action), collect observed next-state vectors.

    state_fn(obs, info) -> np.ndarray must match the state extractor used in
    collect_episodes (utils.preprocess.extract_state or extract_state_oracle).
    """
    table: dict[tuple[tuple[int, int], int], list[np.ndarray]] = defaultdict(list)
    for traj in trajectories:
        for tr in traj:
            s = state_fn(tr["obs"], tr.get("info"))
            cell = _cell_of(s)
            a = int(tr["action"])
            s_next = state_fn(tr["next_obs"], tr.get("next_info"))
            table[(cell, a)].append(np.asarray(s_next, dtype=np.float64))
    return dict(table)


def sample_obs_transition(
    s: np.ndarray,
    a: int,
    table: dict[tuple[tuple[int, int], int], list[np.ndarray]],
    rng: np.random.Generator,
    fallback_state: np.ndarray | None = None,
) -> np.ndarray:
    key = (_cell_of(s), int(a))
    candidates = table.get(key)
    if not candidates:
        return s.copy() if fallback_state is None else fallback_state.copy()
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def fallback_diagnostics(
    table: dict[tuple[tuple[int, int], int], list[np.ndarray]],
    trajectories: list[list[dict]],
    state_fn,
) -> dict[str, float]:
    """Coverage of (cell, action) lookups against the table."""
    total = 0
    missing = 0
    n_samples_per_key: list[int] = []
    for traj in trajectories:
        for tr in traj:
            s = state_fn(tr["obs"], tr.get("info"))
            key = (_cell_of(s), int(tr["action"]))
            total += 1
            if key not in table:
                missing += 1
    for samples in table.values():
        n_samples_per_key.append(len(samples))
    return {
        "lookups": total,
        "missing": missing,
        "fallback_rate": missing / max(total, 1),
        "unique_keys": len(table),
        "mean_samples_per_key": float(np.mean(n_samples_per_key)) if n_samples_per_key else 0.0,
        "min_samples_per_key": int(np.min(n_samples_per_key)) if n_samples_per_key else 0,
    }
