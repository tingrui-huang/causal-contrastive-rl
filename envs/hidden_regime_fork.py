"""
Hidden regime fork: two map layouts differ in which side detour reconnects to the goal.

At each reset, sample ``U ∈ {0,1}`` (when ``confound=True``). The observation does not
contain ``U``; it is exposed only in ``info["confounder"]`` for debugging.

**Layout (default size 15×15):** goal at top center ``(cx, 1)``, agent at bottom center
``(cx, height-2)`` facing up. A **long** vertical corridor runs up the center column, but
the center cell at ``(cx, fork_row)`` is blocked. The agent must detour around that wall:
``U=0`` opens only the left bypass, while ``U=1`` opens only the right bypass. The fork sits
far enough **above** the starting position that, with MiniGrid's 7×7 egocentric view
(facing up), the blocked fork and side corridor are **outside** the field of view for the
first few ``forward`` steps — so ``U=0`` vs ``U=1`` yield identical ``obs["image"]`` at
reset and after 1–2 forwards (same seed), while futures still diverge once the agent
approaches the fork.

See ``experiments/inspect_hidden_fork.py`` to verify aliasing.
"""
from __future__ import annotations

from typing import Any

from gymnasium.core import ObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


# Row where left/right branches split (must stay "above" the 7×7 view for agent at
# bottom for the first ~2 forward steps; validated with size >= 15).
_DEFAULT_FORK_ROW = 4


class HiddenRegimeForkEnv(MiniGridEnv):
    """
    Parameters
    ----------
    size
        Full grid size (odd, **>= 15** recommended for early-step observation aliasing).
    confound
        If True, sample ``U`` each episode (unless ``fixed_u`` is set). If False, fix ``U=0``.
    fixed_u
        If ``0`` or ``1``, always use this regime (for diagnostics / inspection). Ignored when
        ``None``. When set, overrides random sampling even if ``confound=True``.
    fork_row
        Grid row ``y`` of the fork (smaller ``y`` = higher on screen). Default 4.
    """

    def __init__(
        self,
        size: int = 15,
        confound: bool = True,
        fixed_u: int | None = None,
        fork_row: int | None = None,
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        assert size >= 7 and size % 2 == 1, "size must be odd and >= 7"
        if size < 15:
            raise ValueError(
                "HiddenRegimeForkEnv needs size >= 15 so the fork sits outside the "
                "initial 7×7 view (reset + 1–2 forward). Use a larger grid."
            )
        self._size = size
        self.confound = confound
        if fixed_u is not None and fixed_u not in (0, 1):
            raise ValueError("fixed_u must be None, 0, or 1")
        self.fixed_u: int | None = fixed_u
        self.hidden_u: int = 0
        self._fork_row = int(fork_row) if fork_row is not None else _DEFAULT_FORK_ROW

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
        if self.fixed_u is not None:
            self.hidden_u = int(self.fixed_u)
        elif self.confound:
            self.hidden_u = int(self.np_random.integers(0, 2))
        else:
            self.hidden_u = 0

        bottom = height - 2
        if not (1 < self._fork_row < bottom - 1):
            raise ValueError("fork_row must be strictly inside the inner grid above the agent.")

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        walkable = self._walkable_cells(width, height, self.hidden_u)

        for j in range(1, height - 1):
            for i in range(1, width - 1):
                if (i, j) not in walkable:
                    self.grid.set(i, j, Wall())

        cx = width // 2
        self.put_obj(Goal(), cx, 1)

        self.agent_pos = (cx, bottom)
        self.agent_dir = 3  # up (negative y)
        self.mission = "reach the goal at the top"

    def _walkable_cells(self, width: int, height: int, u: int) -> set[tuple[int, int]]:
        """Return floor cells for regime ``u`` (0 = left branch open, 1 = right)."""
        cx = width // 2
        bottom = height - 2
        f = self._fork_row

        # Center spine from the goal to the agent start, except for the blocked fork cell.
        cells = {(cx, y) for y in range(1, bottom + 1) if y != f}

        # Regime-dependent bypass around the blocked center cell at y=f.
        if u == 0:
            cells.update({(cx - 1, f + 1), (cx - 1, f), (cx - 1, f - 1)})
        else:
            cells.update({(cx + 1, f + 1), (cx + 1, f), (cx + 1, f - 1)})

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
