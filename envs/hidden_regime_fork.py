"""
Hidden regime fork: two map layouts differ only in which branch to the goal is open.

At each reset, sample ``U ∈ {0,1}`` (when ``confound=True``). The observation does not
contain ``U``; it is exposed only in ``info["confounder"]`` for debugging.

Layout (9×9 interior): goal at top center ``(4, 1)``, agent at bottom center ``(4, 7)``
facing up. A vertical trunk leads to a fork at ``(4, 3)``; left vs right detour merges
at ``(4, 2)`` before the goal. Exactly one branch is blocked by a wall depending on ``U``.
"""
from __future__ import annotations

from typing import Any

from gymnasium.core import ObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class HiddenRegimeForkEnv(MiniGridEnv):
    """
    Parameters
    ----------
    size
        Full grid size (default 9). Inner walkable region is ``(1..size-2)``.
    confound
        If True, sample ``U`` each episode. If False, fix ``U=0`` (single clean layout).
    """

    def __init__(
        self,
        size: int = 9,
        confound: bool = True,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        assert size >= 7 and size % 2 == 1, "size must be odd and >= 7"
        self._size = size
        self.confound = confound
        self.hidden_u: int = 0

        mission_space = MissionSpace(mission_func=lambda: "reach the goal at the top")

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        if self.confound:
            self.hidden_u = int(self.np_random.integers(0, 2))
        else:
            self.hidden_u = 0

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        walkable = self._walkable_cells(width, height, self.hidden_u)

        for j in range(1, height - 1):
            for i in range(1, width - 1):
                if (i, j) not in walkable:
                    self.grid.set(i, j, Wall())

        cx = width // 2
        self.put_obj(Goal(), cx, 1)

        self.agent_pos = (cx, height - 2)
        self.agent_dir = 3  # up (negative y)
        self.mission = "reach the goal at the top"

    def _walkable_cells(self, width: int, height: int, u: int) -> set[tuple[int, int]]:
        """Return floor cells for regime ``u`` (0 = left branch open, 1 = right)."""
        cx = width // 2
        bottom = height - 2
        # Vertical trunk from fork row y=3 up to start row, then (cx,2),(cx,1)
        cells = {(cx, y) for y in range(3, bottom + 1)}
        cells.update({(cx, 2), (cx, 1)})
        if u == 0:
            cells.update({(cx - 1, 3), (cx - 1, 2)})
        else:
            cells.update({(cx + 1, 3), (cx + 1, 2)})
        return cells

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["confounder"] = int(self.hidden_u)
        return obs, info
