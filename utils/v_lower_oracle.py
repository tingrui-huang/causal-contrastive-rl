"""
Oracle V_lower for WindyCorridor — BFS distance to goal + lava penalty.

Why this is sufficient (and not "cheating"):
* The env is a finite 15×15 grid with explicitly enumerated walkable cells and
  6 lava cells. V_lower is dir-agnostic — only position matters.
* In the (iii) branch of Theorem 2, the argmin selects from a tiny neighbor set
  (≤5 candidates per (s, x)). The dominant signal is "is there a lava neighbor?"
  — V_lower's only job for non-lava candidates is to prefer cells farther from
  the goal, which BFS captures exactly.
* Decoupling V_lower estimation from contrastive critic training removes the
  V_e bootstrap bias (V_e never sees walks into lava because the expert waits).
  Future work: replace with learned V_lower for envs without ground truth.

The module pre-computes BFS distances at import time over walkable cells.
"""
from __future__ import annotations

from collections import deque

from envs.windy_corridor import GOAL_POS, WindyCorridorEnv
from utils.wind_dynamics import LAVA_CELLS, WALKABLE_CELLS

LAVA_PENALTY: float = -1e9
UNREACHABLE_PENALTY: float = -1e6


def _compute_bfs_distances() -> dict[tuple[int, int], int]:
    """4-connected BFS from GOAL_POS over walkable cells (ignoring direction)."""
    dists: dict[tuple[int, int], int] = {GOAL_POS: 0}
    queue: deque[tuple[int, int]] = deque([GOAL_POS])
    while queue:
        x, y = queue.popleft()
        d = dists[(x, y)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (x + dx, y + dy)
            if nb in WALKABLE_CELLS and nb not in dists:
                dists[nb] = d + 1
                queue.append(nb)
    return dists


_BFS_DISTANCES: dict[tuple[int, int], int] = _compute_bfs_distances()


def v_lower(state) -> float:
    """V_lower((x, y, dir)) — direction is ignored."""
    pos = (int(state[0]), int(state[1]))
    if pos in LAVA_CELLS:
        return LAVA_PENALTY
    dist = _BFS_DISTANCES.get(pos)
    if dist is None:
        return UNREACHABLE_PENALTY
    return float(-dist)


def bfs_distance(pos: tuple[int, int]) -> int | None:
    """Raw BFS distance (None if unreachable). Useful for diagnostics."""
    return _BFS_DISTANCES.get(pos)
