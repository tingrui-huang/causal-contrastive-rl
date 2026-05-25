"""
Domain knowledge for the WindyCorridor grid world.

Restricts the pessimistic worst-case to physically reachable neighbor states
instead of the entire state space. This resolves over-conservatism: near the
goal (no lava nearby), the worst case becomes the worst reachable corridor
cell — not a distant lava cell.

Max one-step displacement in the grid:
  - forward with tailwind:  2 cells in facing direction
  - forward with crosswind: 1 cell forward + 1 cell lateral
  - other cases:           ≤1 cell
  → Manhattan distance ≤ 2 covers all physically reachable positions.
"""
from __future__ import annotations

import numpy as np

from envs.windy_corridor import WindyCorridorEnv

_WALKABLE: set[tuple[int, int]] = WindyCorridorEnv.walkable_cells()

POS_X_IDX = 147
POS_Y_IDX = 148


def reachable_positions(
    x: int, y: int, max_manhattan: int = 2,
    walkable: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """All walkable cells within `max_manhattan` of (x, y).

    If `walkable` is not provided, falls back to the WindyCorridor walkable set
    (kept for backward compatibility with existing call sites).
    """
    cells = walkable if walkable is not None else _WALKABLE
    return [
        (wx, wy)
        for wx, wy in cells
        if abs(wx - x) + abs(wy - y) <= max_manhattan
    ]


def build_position_index(
    episode_states: list[np.ndarray],
) -> dict[tuple[int, int], list[np.ndarray]]:
    """Map (x, y) → list of observed state vectors at that position."""
    index: dict[tuple[int, int], list[np.ndarray]] = {}
    for states in episode_states:
        for s in states:
            key = (int(round(s[POS_X_IDX])), int(round(s[POS_Y_IDX])))
            if key not in index:
                index[key] = []
            index[key].append(s)
    return index


def deduplicate_position_index(
    index: dict[tuple[int, int], list[np.ndarray]],
    max_per_position: int = 8,
    rng: np.random.Generator | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Compact the index to at most `max_per_position` states per cell.

    Returns arrays of shape (K, state_dim) keyed by (x, y).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    compact: dict[tuple[int, int], np.ndarray] = {}
    for key, state_list in index.items():
        arr = np.stack(state_list, axis=0)
        if arr.shape[0] > max_per_position:
            chosen = rng.choice(arr.shape[0], size=max_per_position, replace=False)
            arr = arr[chosen]
        compact[key] = arr.astype(np.float64)
    return compact


def get_neighbor_states_for_batch(
    states: np.ndarray,
    pos_index: dict[tuple[int, int], np.ndarray],
    max_manhattan: int = 2,
    max_per_sample: int = 32,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """For each state in the batch, collect neighbor state vectors.

    Returns list of length B. Each element has shape (K_i, state_dim).
    """
    if rng is None:
        rng = np.random.default_rng()
    # Derive walkable set from the pos_index so this works for any env (not just
    # the one whose walkable was hardcoded into _WALKABLE at module load).
    walkable_from_index = set(pos_index.keys())
    result: list[np.ndarray] = []
    for i in range(states.shape[0]):
        x = int(round(states[i, POS_X_IDX]))
        y = int(round(states[i, POS_Y_IDX]))
        neighbor_pos = reachable_positions(x, y, max_manhattan, walkable=walkable_from_index)

        all_ns: list[np.ndarray] = []
        for nx, ny in neighbor_pos:
            if (nx, ny) in pos_index:
                all_ns.append(pos_index[(nx, ny)])

        if not all_ns:
            result.append(states[i : i + 1].astype(np.float64))
            continue

        combined = np.concatenate(all_ns, axis=0)
        if combined.shape[0] > max_per_sample:
            chosen = rng.choice(combined.shape[0], size=max_per_sample, replace=False)
            combined = combined[chosen]
        result.append(combined)
    return result
