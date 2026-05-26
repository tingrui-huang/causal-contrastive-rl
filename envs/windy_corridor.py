"""
WindyCorridor — 15x15 MiniGrid map for causal contrastive RL.

This module only defines the static grid (walls, lava, goal, start).
Wind is applied by wrapping with `causal_gym.WindyMiniGridSCM`.

Map layout
----------
    x:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
y=0     W  W  W  W  W  W  W  W  W  W  W  W  W  W  W
y=1     W  A  .  .  .  .  .  .  .  .  .  .  .  G  W   NEAR row
y=2     W  .  W  L  L  L  W  W  L  L  L  W  W  .  W   lava strip
y=3     W  .  W  W  W  W  W  W  W  W  W  W  W  .  W
y=4-11  W  .  W  ── interior walls ────────────  W  .  W
y=12    W  .  W  W  W  W  W  W  W  W  W  W  W  .  W
y=13    W  .  .  .  .  .  .  .  .  .  .  .  .  .  W   FAR row
y=14    W  W  W  W  W  W  W  W  W  W  W  W  W  W  W

Start (1,1), goal (13,1). Two routes:
  NEAR: along y=1, ~13 steps, sometimes lethal under south wind.
  FAR:  (1,1) → x=1 down → y=13 east → x=13 up → goal, ~39 steps, always safe.

Lethal NEAR positions under south wind: x ∈ {2,3,4,7,8,9} at y=1.
Safe waiting points: x ∈ {5,6,10,11} at y=1.

Note on (12, 2): kept as Wall (not Lava). Setting it to Lava would create a
death trap on the FAR route at (13, 3) facing north under west wind, where
lateral drift lands the agent on (12, 2).
"""
from __future__ import annotations

from typing import Any

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

SIZE = 15
START_POS: tuple[int, int] = (1, 1)
GOAL_POS: tuple[int, int] = (13, 1)
LETHAL_X: frozenset[int] = frozenset({2, 3, 4, 7, 8, 9})
SAFE_WAIT_X: frozenset[int] = frozenset({5, 6, 10, 11})


class WindyCorridorEnv(MiniGridEnv):
    """Static 15x15 corridor grid; wind is applied externally."""

    def __init__(
        self,
        size: int = SIZE,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        if size != SIZE:
            raise ValueError(f"WindyCorridorEnv requires size={SIZE}")
        if max_steps is None:
            max_steps = 4 * size**2

        mission_space = MissionSpace(mission_func=lambda: "reach the goal")
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _y2_pattern() -> dict[int, str]:
        """Cells of y=2 row. 'W'=wall, 'L'=lava, '.'=open corridor."""
        # Indices: 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14
        pattern = "W   .   W   L   L   L   W   W   L   L   L   W   W   .   W".split()
        return {x: pattern[x] for x in range(SIZE)}

    @staticmethod
    def walkable_cells() -> set[tuple[int, int]]:
        """Cells the agent can legally occupy. Used by BFS / planners."""
        cells: set[tuple[int, int]] = set()
        cells.update((x, 1) for x in range(1, 14))      # NEAR
        cells.update((x, 13) for x in range(1, 14))     # FAR
        cells.update((1, y) for y in range(1, 14))      # left connector
        cells.update((13, y) for y in range(1, 14))     # right connector
        return cells

    @staticmethod
    def start_pos() -> tuple[int, int]:
        return START_POS

    @staticmethod
    def goal_pos() -> tuple[int, int]:
        return GOAL_POS

    @staticmethod
    def lethal_near_x() -> frozenset[int]:
        return LETHAL_X

    @staticmethod
    def safe_wait_near_x() -> frozenset[int]:
        return SAFE_WAIT_X

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        walkable = self.walkable_cells()
        y2 = self._y2_pattern()

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if (x, y) in walkable:
                    continue
                if y == 2:
                    cell = y2[x]
                    if cell == "L":
                        self.grid.set(x, y, Lava())
                    else:
                        self.grid.set(x, y, Wall())
                else:
                    self.grid.set(x, y, Wall())

        gx, gy = self.goal_pos()
        self.put_obj(Goal(), gx, gy)

        self.agent_pos = self.start_pos()
        self.agent_dir = 0  # face east
        self.mission = "reach the goal"
