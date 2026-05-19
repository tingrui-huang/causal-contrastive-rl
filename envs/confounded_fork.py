"""ConfoundedFork: a minimal 11x11 confounding env with a single visible fork.

Design (per planning discussion):
- Start at (1, 9), goal at (9, 1).
- Fork at (1, 8): one step north of start. Go N → short left path with lava risk.
                                            Go E → long right path, safe.
- Lava at (2, 3..6): hugs the east side of the left vertical corridor.
- Wind under U=1 is GLOBAL: every cell has the same 20% wind probability,
  uniformly distributed across 4 directions.
- Under U=0 (no wind) the left path is strictly safe and tied in length with the
  right path. Oracle picks left under U=0 by BFS tie-break order; under U=1 the
  oracle's regime-restricted walkable removes the left corridor, forcing right.

This env serves as a clean teaching example: a single confounded fork that
shows the U-aware expert switching paths based on the hidden wind regime.
"""
from __future__ import annotations

from typing import Any

from envs.windy_corridor import WindyCorridorEnv


# Directional wind per regime — each regime has wind blowing toward a single
# direction, making one path's adjacent lava reachable by drift. This breaks
# the "universally safe default" that previously masked confounding.
#
# Wind direction encoding (MiniGrid agent_dir convention):
#   0 = east, 1 = south, 2 = west, 3 = north, 4 = no wind.
#
# Tuple format: (east, south, west, north, none).
#
# U=0: east wind 40% → drifts agent east. On left corridor (x=1) drifts into
#      x=2 lava (y=3..6) → left path dangerous. Oracle picks right.
# U=1: west wind 40% → drifts agent west. On right corridor (x=9) drifts into
#      x=8 lava (y=4..5) → right path dangerous. Oracle picks left.
_GLOBAL_WIND_DIST: dict[int, tuple[float, ...]] = {
    0: (0.40, 0.00, 0.00, 0.00, 0.60),
    1: (0.00, 0.00, 0.40, 0.00, 0.60),
}


class ConfoundedForkEnv(WindyCorridorEnv):
    """11x11 single-fork confounded env with global wind under U=1."""

    def __init__(
        self,
        size: int = 11,
        confound: bool = True,
        fixed_u: int | None = None,
        wind_dist: Any = None,
        wind_strength: float = 1.0,
        wind_per: str = "step",
        lethal_boundaries: bool = False,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        if size != 11:
            raise ValueError("ConfoundedForkEnv only supports size=11.")
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
        return (1, 9)

    @staticmethod
    def goal_pos() -> tuple[int, int]:
        return (9, 1)

    @staticmethod
    def walkable_cells() -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        # Left vertical x=1, y=1..9
        cells.update((1, y) for y in range(1, 10))
        # Right vertical x=9, y=1..8
        cells.update((9, y) for y in range(1, 9))
        # Fork horizontal y=8, x=1..9
        cells.update((x, 8) for x in range(1, 10))
        # Top horizontal y=1, x=1..9
        cells.update((x, 1) for x in range(1, 10))
        return cells

    @staticmethod
    def hazard_cells() -> set[tuple[int, int]]:
        # Asymmetric lava: 4 cells east of the left corridor (x=2, y=3..6) and
        # 2 cells west of the right corridor (x=8, y=4..5). Under U=1 wind, the
        # left path has ~2x the lava exposure of the right path, breaking the
        # "always pick right" universal-safe-default that masked confounding in
        # the original symmetric layout.
        return {(2, y) for y in range(3, 7)} | {(8, y) for y in range(4, 6)}

    @staticmethod
    def high_wind_cells() -> set[tuple[int, int]]:
        # With global wind dist, every walkable cell is windy under U=1.
        return ConfoundedForkEnv.walkable_cells()

    @staticmethod
    def walkable_for_regime(hidden_u: int) -> set[tuple[int, int]]:
        """Oracle U-awareness for directional-wind regimes.

        U=0 (east wind): would drift the agent eastward off the left column
                         into the adjacent lava at x=2. Restrict the left
                         vertical, forcing BFS through the safer right path.
        U=1 (west wind): would drift the agent westward off the right column
                         into the adjacent lava at x=8. Restrict the right
                         vertical, forcing BFS through the safer left path.

        Restriction applies to oracle planning only; the env's true walkable
        cells include both branches, and the actor at eval time may pick either.
        """
        walkable = ConfoundedForkEnv.walkable_cells()
        if hidden_u == 0:
            # U=0 east wind → left dangerous → force right
            danger = {(1, y) for y in range(1, 8)}
            return walkable - danger
        # U=1 west wind → right dangerous → force left
        danger = {(9, y) for y in range(2, 9)}
        return walkable - danger
