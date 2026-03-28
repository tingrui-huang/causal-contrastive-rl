"""
Windy corridor environment with multiple route segments.

Phase W2 implementation:
- preferred host topology for wind experiments
- hidden regime ``U`` sampled at reset
- wind direction sampled from ``P(wind | U)``
- wind is observable in ``info`` only, not in ``obs``
- wind affects `forward` only:
  - same direction: move 2
  - opposite direction: move 0
  - otherwise: move 1
"""
from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.core import ObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

_DEFAULT_WIND_DIST: dict[int, tuple[float, ...]] = {
    0: (0.05, 0.05, 0.05, 0.05, 0.80),
    1: (0.25, 0.25, 0.05, 0.05, 0.40),
}


class WindyCorridorEnv(MiniGridEnv):
    """
    Preferred host topology for wind confounding.

    The map contains three horizontal corridor bands plus staggered vertical passages.
    This produces several viable routes from the bottom-left start to the top-right goal.
    During Phase W2, wind alters only the forward action while preserving observation shape.
    """

    def __init__(
        self,
        size: int = 15,
        confound: bool = True,
        fixed_u: int | None = None,
        wind_dist: dict[int, tuple[float, ...]] | None = None,
        wind_strength: float = 1.0,
        wind_per: str = "step",
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        if size != 15:
            raise ValueError("WindyCorridorEnv currently supports only size=15.")
        if fixed_u is not None and fixed_u not in (0, 1):
            raise ValueError("fixed_u must be None, 0, or 1")
        if wind_per not in {"step", "episode"}:
            raise ValueError("wind_per must be 'step' or 'episode'")
        if not 0.0 <= float(wind_strength) <= 1.0:
            raise ValueError("wind_strength must be in [0.0, 1.0]")

        self._size = size
        self.confound = confound
        self.fixed_u = fixed_u
        self.hidden_u: int = 0
        self.wind_strength = float(wind_strength)
        self.wind_per = wind_per
        self.wind_dist = self._validate_wind_dist(wind_dist or _DEFAULT_WIND_DIST)
        self.wind_direction: int = 4
        self._step_count_for_debug = 0

        mission_space = MissionSpace(mission_func=lambda: "reach the goal through the corridor network")
        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _validate_wind_dist(
        wind_dist: dict[int, tuple[float, ...]],
    ) -> dict[int, tuple[float, ...]]:
        validated: dict[int, tuple[float, ...]] = {}
        for u in (0, 1):
            if u not in wind_dist:
                raise ValueError(f"wind_dist missing key {u}")
            probs = tuple(float(v) for v in wind_dist[u])
            if len(probs) != 5:
                raise ValueError("each wind_dist[u] must have exactly 5 probabilities")
            if any(v < 0.0 for v in probs):
                raise ValueError("wind probabilities must be non-negative")
            total = sum(probs)
            if not np.isclose(total, 1.0):
                raise ValueError(f"wind_dist[{u}] must sum to 1.0, got {total}")
            validated[u] = probs
        return validated

    def _effective_wind_probs(self) -> tuple[float, ...]:
        calm = np.array((0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        regime = np.array(self.wind_dist[self.hidden_u], dtype=np.float64)
        mixed = (1.0 - self.wind_strength) * calm + self.wind_strength * regime
        return tuple(float(v) for v in mixed)

    def _sample_hidden_u(self) -> int:
        if self.fixed_u is not None:
            return int(self.fixed_u)
        if self.confound:
            return int(self.np_random.integers(0, 2))
        return 0

    def _sample_wind_direction(self) -> int:
        probs = np.array(self._effective_wind_probs(), dtype=np.float64)
        return int(self.np_random.choice(5, p=probs))

    @staticmethod
    def walkable_cells() -> set[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()

        # Three horizontal corridor bands.
        cells.update((x, 13) for x in range(1, 14))
        cells.update((x, 9) for x in range(3, 12))
        cells.update((x, 5) for x in range(7, 14))

        # Staggered vertical passages.
        cells.update((3, y) for y in range(9, 14))
        cells.update((7, y) for y in range(5, 10))
        cells.update((11, y) for y in range(5, 14))
        return cells

    @staticmethod
    def high_wind_cells() -> set[tuple[int, int]]:
        return {
            *( (x, 13) for x in range(8, 14) ),
            *( (11, y) for y in range(6, 14) ),
            (12, 5),
            (13, 5),
        }

    @staticmethod
    def start_pos() -> tuple[int, int]:
        return (1, 13)

    @staticmethod
    def goal_pos() -> tuple[int, int]:
        return (13, 5)

    def _gen_grid(self, width: int, height: int) -> None:
        self.hidden_u = self._sample_hidden_u()

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        walkable = self.walkable_cells()
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if (x, y) not in walkable:
                    self.grid.set(x, y, Wall())

        gx, gy = self.goal_pos()
        self.put_obj(Goal(), gx, gy)

        self.agent_pos = self.start_pos()
        self.agent_dir = 0  # face right
        self.mission = "reach the goal through the corridor network"

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self._step_count_for_debug = 0
        self.wind_direction = self._sample_wind_direction()

        info = dict(info)
        info["confounder"] = int(self.hidden_u)
        info["wind_direction"] = int(self.wind_direction)
        info["wind_strength"] = float(self.wind_strength)
        info["wind_per"] = self.wind_per
        return obs, info

    def step(self, action: int):
        if self.wind_per == "step":
            self.wind_direction = self._sample_wind_direction()
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == self.actions.forward:
            reward, terminated = self._step_forward_with_wind()
        elif action == self.actions.pickup:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell and fwd_cell.can_pickup() and self.carrying is None:
                self.carrying = fwd_cell
                self.carrying.cur_pos = np.array([-1, -1])
                self.grid.set(fwd_pos[0], fwd_pos[1], None)
        elif action == self.actions.drop:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None
        elif action == self.actions.toggle:
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)
        elif action == self.actions.done:
            pass
        else:
            raise ValueError(f"Unknown action: {action}")

        self._step_count_for_debug += 1

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()
        info: dict[str, Any] = {}
        info["confounder"] = int(self.hidden_u)
        info["wind_direction"] = int(self.wind_direction)
        info["wind_strength"] = float(self.wind_strength)
        info["wind_per"] = self.wind_per
        return obs, reward, terminated, truncated, info

    def _step_forward_with_wind(self) -> tuple[float, bool]:
        moves = self._wind_adjusted_forward_steps()
        reward = 0.0
        terminated = False

        for _ in range(moves):
            step_reward, step_terminated = self._advance_once()
            reward += step_reward
            terminated = terminated or step_terminated
            if terminated:
                break
        return reward, terminated

    def _wind_adjusted_forward_steps(self) -> int:
        if self.wind_strength <= 0.0:
            return 1
        if self.wind_direction == 4:
            return 1
        if self.agent_dir == self.wind_direction:
            return 2
        if (self.agent_dir - 2) % 4 == self.wind_direction:
            return 0
        return 1

    def _advance_once(self) -> tuple[float, bool]:
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        reward = 0.0
        terminated = False

        if fwd_cell is None or fwd_cell.can_overlap():
            self.agent_pos = tuple(fwd_pos)
        if fwd_cell is not None and fwd_cell.type == "goal":
            terminated = True
            reward = float(self._reward())
        if fwd_cell is not None and fwd_cell.type == "lava":
            terminated = True

        return reward, terminated
