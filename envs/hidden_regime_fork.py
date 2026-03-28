"""
Hidden regime fork with switchable map variants.

At each reset, sample ``U ∈ {0,1}`` (when ``confound=True``). The observation does not
contain ``U``; it is exposed only in ``info["confounder"]`` for debugging.

Supported map variants:

- ``branch_wall`` (legacy / default): one visible branch is blocked by walls, the other
  branch is open.
- ``hidden_trap``: both branches look open and symmetric, but one side is a latent
  regime-dependent failure corridor that terminates the episode with zero reward.

See ``experiments/inspect_hidden_fork.py`` to verify aliasing.
"""
from __future__ import annotations

from typing import Any

from gymnasium import spaces
from gymnasium.core import ObsType
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


# Row where left/right branches split (must stay "above" the 7×7 view for agent at
# bottom for the first ~2 forward steps; validated with size >= 15).
_DEFAULT_FORK_ROW = 4
_SUPPORTED_MAP_VARIANTS = ("branch_wall", "hidden_trap")


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
        map_variant: str = "branch_wall",
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
        if map_variant not in _SUPPORTED_MAP_VARIANTS:
            raise ValueError(
                f"map_variant must be one of {_SUPPORTED_MAP_VARIANTS}, got {map_variant!r}"
            )
        self.map_variant = map_variant
        if fixed_u is not None and fixed_u not in (0, 1):
            raise ValueError("fixed_u must be None, 0, or 1")
        self.fixed_u: int | None = fixed_u
        self.hidden_u: int = 0
        self._fork_row = int(fork_row) if fork_row is not None else _DEFAULT_FORK_ROW
        self._unsafe_branch_cells: set[tuple[int, int]] = set()

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
        self.observation_space = spaces.Dict(
            {
                **self.observation_space.spaces,
                "agent_pos": spaces.Box(
                    low=0,
                    high=self._size - 1,
                    shape=(2,),
                    dtype=np.int64,
                ),
            }
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
        self._unsafe_branch_cells = self._hidden_trap_cells(width, self.hidden_u)

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
        """Return floor cells for regime ``u`` under the selected map variant."""
        cx = width // 2
        bottom = height - 2
        f = self._fork_row

        # Center spine from the goal to the agent start, except for the blocked fork cell.
        cells = {(cx, y) for y in range(1, bottom + 1) if y != f}

        left_branch = {(cx - 1, f + 1), (cx - 1, f), (cx - 1, f - 1)}
        right_branch = {(cx + 1, f + 1), (cx + 1, f), (cx + 1, f - 1)}
        if self.map_variant == "branch_wall":
            if u == 0:
                cells.update(left_branch)
            else:
                cells.update(right_branch)
            return cells
        if self.map_variant == "hidden_trap":
            cells.update(left_branch)
            cells.update(right_branch)
            return cells
        raise RuntimeError(f"Unsupported map_variant: {self.map_variant!r}")

    def _hidden_trap_cells(self, width: int, u: int) -> set[tuple[int, int]]:
        if self.map_variant != "hidden_trap":
            return set()
        cx = width // 2
        f = self._fork_row
        if u == 0:
            unsafe_x = cx + 1
        else:
            unsafe_x = cx - 1
        return {(unsafe_x, f + 1), (unsafe_x, f), (unsafe_x, f - 1)}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        obs = self._augment_obs(obs)
        info = dict(info)
        info["confounder"] = int(self.hidden_u)
        info["map_variant"] = self.map_variant
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        obs = self._augment_obs(obs)
        info = dict(info)
        if (
            self.map_variant == "hidden_trap"
            and not terminated
            and not truncated
            and tuple(int(v) for v in self.agent_pos) in self._unsafe_branch_cells
        ):
            terminated = True
            reward = 0
            info["hidden_trap_triggered"] = True
        else:
            info["hidden_trap_triggered"] = False
        info["map_variant"] = self.map_variant
        return obs, reward, terminated, truncated, info

    def _augment_obs(self, obs: ObsType) -> ObsType:
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["agent_pos"] = np.array(self.agent_pos, dtype=np.int64)
        return obs
