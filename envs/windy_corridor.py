"""
Windy corridor environment with multiple route segments.

Current wind semantics follow the Causal-Gymnasium pattern:
- hidden regime ``U`` sampled at reset
- wind direction sampled from ``P(wind | U)``
- wind is observable in ``info`` only, not in ``obs``
- wind rewrites the physical effect of `forward`
  - same direction: move 2
  - opposite direction: move 0
  - lateral direction: move 1 and drift 1 cell sideways when possible
- optional lethal boundaries can turn drift mistakes into immediate failure
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

WindDistLike = dict[int, tuple[float, ...]] | Callable[[int, tuple[int, int]], tuple[float, ...]]

_DEFAULT_SHELTERED_WIND_DIST: dict[int, tuple[float, ...]] = {
    0: (0.00, 0.00, 0.00, 0.00, 1.00),
    1: (0.20, 0.20, 0.05, 0.05, 0.50),
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
        wind_dist: WindDistLike | None = None,
        wind_strength: float = 1.0,
        wind_per: str = "step",
        lethal_boundaries: bool = False,
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
        self.lethal_boundaries = bool(lethal_boundaries)
        self.wind_dist = self._validate_wind_dist(wind_dist or self._default_wind_dist)
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

    @staticmethod
    def _validate_wind_probs(probs_like: tuple[float, ...] | list[float]) -> tuple[float, ...]:
        probs = tuple(float(v) for v in probs_like)
        if len(probs) != 5:
            raise ValueError("wind probabilities must have exactly 5 entries")
        if any(v < 0.0 for v in probs):
            raise ValueError("wind probabilities must be non-negative")
        total = sum(probs)
        if not np.isclose(total, 1.0):
            raise ValueError(f"wind probabilities must sum to 1.0, got {total}")
        return probs

    @classmethod
    def _validate_wind_dist(
        cls,
        wind_dist: WindDistLike,
    ) -> WindDistLike:
        if callable(wind_dist):
            return wind_dist

        validated: dict[int, tuple[float, ...]] = {}
        for u in (0, 1):
            if u not in wind_dist:
                raise ValueError(f"wind_dist missing key {u}")
            validated[u] = cls._validate_wind_probs(wind_dist[u])
        return validated

    def _default_wind_dist(self, u: int, pos: tuple[int, int]) -> tuple[float, ...]:
        x, y = pos
        if x >= 8 and y == 13:
            exposed = {
                0: (0.00, 0.00, 0.00, 0.00, 1.00),
                1: (0.55, 0.10, 0.05, 0.05, 0.25),
            }
            return exposed[u]
        if x == 11 and 6 <= y <= 13:
            exposed = {
                0: (0.00, 0.00, 0.00, 0.00, 1.00),
                1: (0.55, 0.10, 0.05, 0.05, 0.25),
            }
            return exposed[u]
        if (x, y) in {(12, 5), (13, 5)}:
            exposed = {
                0: (0.00, 0.00, 0.00, 0.00, 1.00),
                1: (0.50, 0.05, 0.05, 0.05, 0.35),
            }
            return exposed[u]
        return _DEFAULT_SHELTERED_WIND_DIST[u]

    def _base_wind_probs(self, pos: tuple[int, int] | None = None) -> tuple[float, ...]:
        position = tuple(int(v) for v in (self.agent_pos if pos is None else pos))
        if callable(self.wind_dist):
            return self._validate_wind_probs(self.wind_dist(int(self.hidden_u), position))
        return self._validate_wind_probs(self.wind_dist[self.hidden_u])

    def _effective_wind_probs(self, pos: tuple[int, int] | None = None) -> tuple[float, ...]:
        calm = np.array((0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float64)
        regime = np.array(self._base_wind_probs(pos), dtype=np.float64)
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
    def hazard_cells() -> set[tuple[int, int]]:
        return {
            (10, 10),
            (10, 11),
            (12, 10),
            (12, 11),
            (12, 12),
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
        hazards = self.hazard_cells() if self.lethal_boundaries else set()
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if (x, y) not in walkable:
                    self.grid.set(x, y, Lava() if (x, y) in hazards else Wall())

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
        obs = self._augment_obs(obs)

        info = dict(info)
        info["confounder"] = int(self.hidden_u)
        info["wind_direction"] = int(self.wind_direction)
        info["wind_strength"] = float(self.wind_strength)
        info["wind_per"] = self.wind_per
        info["lethal_boundaries"] = bool(self.lethal_boundaries)
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

        obs = self._augment_obs(self.gen_obs())
        info: dict[str, Any] = {}
        info["confounder"] = int(self.hidden_u)
        info["wind_direction"] = int(self.wind_direction)
        info["wind_strength"] = float(self.wind_strength)
        info["wind_per"] = self.wind_per
        info["lethal_boundaries"] = bool(self.lethal_boundaries)
        return obs, reward, terminated, truncated, info

    def _augment_obs(self, obs: ObsType) -> ObsType:
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["agent_pos"] = np.array(self.agent_pos, dtype=np.int64)
        return obs

    def _step_forward_with_wind(self) -> tuple[float, bool]:
        action_sequence = self._wind_action_sequence()
        reward = 0.0
        terminated = False

        for action in action_sequence:
            step_reward, step_terminated = self._apply_internal_action(action)
            reward += step_reward
            terminated = terminated or step_terminated
            if terminated:
                break
        return reward, terminated

    def _wind_action_sequence(self) -> tuple[int, ...]:
        if self.wind_strength <= 0.0:
            return (self.actions.forward,)
        if self.wind_direction == 4:
            return (self.actions.forward,)
        if self.agent_dir == self.wind_direction:
            return (self.actions.forward, self.actions.forward)
        if (self.agent_dir - 2) % 4 == self.wind_direction:
            return ()

        turn_action = self._turn_action_toward(self.wind_direction)
        turn_back_action = self.actions.left if turn_action == self.actions.right else self.actions.right
        return (
            self.actions.forward,
            turn_action,
            self.actions.forward,
            turn_back_action,
        )

    def _turn_action_toward(self, direction: int) -> int:
        delta = (direction - self.agent_dir) % 4
        if delta == 1:
            return self.actions.right
        if delta == 3:
            return self.actions.left
        raise ValueError(
            f"Wind direction {direction} is not lateral to agent_dir {self.agent_dir}"
        )

    def _apply_internal_action(self, action: int) -> tuple[float, bool]:
        if action == self.actions.left:
            self.agent_dir = (self.agent_dir - 1) % 4
            return 0.0, False
        if action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
            return 0.0, False
        if action == self.actions.forward:
            return self._advance_once()
        if action == self.actions.done:
            return 0.0, False
        raise ValueError(f"Unsupported internal action: {action}")

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
