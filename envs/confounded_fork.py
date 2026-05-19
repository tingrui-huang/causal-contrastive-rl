"""ConfoundedFork: a 15×15 confounding env with one visible fork early in the
path. Left branch is short (23 steps) but flanked by lava on BOTH sides; right
branch is long (41 steps) but lava-free, forcing a meaningful detour.

Wind model:
- U=0 (calm): no wind, everywhere safe.
- U=1 (windy): uniform 4-direction wind, each direction 20%, no wind 20%.
            Under U=1 the agent on the left column gets drifted east or west
            into adjacent lava with high probability (~40% per risky step).

Oracle U-awareness:
- U=0 picks left (short, no risk without wind).
- U=1 picks right (no lava, wind only stalls progress).
"""
from __future__ import annotations

from typing import Any

from envs.windy_corridor import WindyCorridorEnv


_GLOBAL_WIND_DIST: dict[int, tuple[float, ...]] = {
    0: (0.00, 0.00, 0.00, 0.00, 1.00),
    1: (0.10, 0.10, 0.10, 0.10, 0.60),
}


class ConfoundedForkEnv(WindyCorridorEnv):
    """15×15 single-fork env: short left flanked by 2-sided lava vs long
    detour right with no lava."""

    def __init__(
        self,
        size: int = 15,
        confound: bool = True,
        fixed_u: int | None = None,
        wind_dist: Any = None,
        wind_strength: float = 1.0,
        wind_per: str = "step",
        lethal_boundaries: bool = False,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        if size != 15:
            raise ValueError("ConfoundedForkEnv only supports size=15.")
        super().__init__(
            size=size,
            confound=confound,
            fixed_u=fixed_u,
            wind_dist=wind_dist if wind_dist is not None else _GLOBAL_WIND_DIST,
            wind_strength=wind_strength,
            wind_per=wind_per,
            lethal_boundaries=lethal_boundaries,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def start_pos() -> tuple[int, int]:
        return (2, 13)

    @staticmethod
    def goal_pos() -> tuple[int, int]:
        return (13, 1)

    @staticmethod
    def walkable_cells() -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        # Left vertical x=2, y=1..13 (short left path)
        cells.update((2, y) for y in range(1, 14))
        # Top horizontal y=1, x=2..13 (final approach to goal)
        cells.update((x, 1) for x in range(2, 14))
        # Fork horizontal y=12, x=2..13 (start of right detour)
        cells.update((x, 12) for x in range(2, 14))
        # Right vertical x=13, y=4..12 (right detour ascends partway)
        cells.update((13, y) for y in range(4, 13))
        # Middle bridge y=4, x=4..13 (force the detour to swing back left)
        cells.update((x, 4) for x in range(4, 14))
        # Short left column x=4, y=1..4 (connects bridge back to top)
        cells.update((4, y) for y in range(1, 5))
        return cells

    @staticmethod
    def hazard_cells() -> set[tuple[int, int]]:
        # 2-sided lava flanking the left vertical corridor at y=3..10
        west = {(1, y) for y in range(3, 11)}
        east = {(3, y) for y in range(3, 11)}
        return west | east

    @staticmethod
    def high_wind_cells() -> set[tuple[int, int]]:
        return ConfoundedForkEnv.walkable_cells()

    @staticmethod
    def walkable_for_regime(hidden_u: int) -> set[tuple[int, int]]:
        """Oracle U-awareness via regime-restricted BFS planning set.

        U=0: pick left (short, no risk without wind). Remove the right detour
             cells so BFS commits to the left column.
        U=1: pick right (long but no lava). Remove the left column's risky
             middle stretch (y=2..11) so BFS routes around through the right
             detour. Keep start (2,13), fork (2,12), and top approach (2,1)
             so the path stays connected.
        """
        walkable = ConfoundedForkEnv.walkable_cells()
        if hidden_u == 0:
            # Force left: drop the detour structure
            danger = set()
            danger.update((13, y) for y in range(4, 12))   # right vertical
            danger.update((x, 4) for x in range(4, 13))    # middle bridge
            danger.update((4, y) for y in range(2, 4))     # short detour stub
            danger.update((x, 12) for x in range(3, 14))   # fork horizontal interior
            return walkable - danger
        # Force right under U=1: drop the risky left middle stretch
        danger = {(2, y) for y in range(2, 12)}
        return walkable - danger
