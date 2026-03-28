"""Wind-enabled legacy HiddenRegimeFork environment for comparison only."""
from __future__ import annotations

from typing import Any

import numpy as np

from envs.hidden_regime_fork import HiddenRegimeForkEnv

_DEFAULT_WIND_DIST: dict[int, tuple[float, ...]] = {
    0: (0.05, 0.05, 0.05, 0.05, 0.80),
    1: (0.25, 0.25, 0.05, 0.05, 0.40),
}


class WindyHiddenRegimeForkEnv(HiddenRegimeForkEnv):
    """Legacy fork host with the same wind mechanics as `WindyCorridorEnv`."""

    def __init__(
        self,
        *,
        wind_dist: dict[int, tuple[float, ...]] | None = None,
        wind_strength: float = 1.0,
        wind_per: str = "step",
        **kwargs: Any,
    ) -> None:
        if wind_per not in {"step", "episode"}:
            raise ValueError("wind_per must be 'step' or 'episode'")
        if not 0.0 <= float(wind_strength) <= 1.0:
            raise ValueError("wind_strength must be in [0.0, 1.0]")

        self.wind_strength = float(wind_strength)
        self.wind_per = wind_per
        self.wind_dist = self._validate_wind_dist(wind_dist or _DEFAULT_WIND_DIST)
        self.wind_direction: int = 4
        super().__init__(**kwargs)

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

    def _sample_wind_direction(self) -> int:
        probs = np.array(self._effective_wind_probs(), dtype=np.float64)
        return int(self.np_random.choice(5, p=probs))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        obs = self._augment_obs(obs)
        self.wind_direction = self._sample_wind_direction()
        info = dict(info)
        info["wind_direction"] = int(self.wind_direction)
        info["wind_strength"] = float(self.wind_strength)
        info["wind_per"] = self.wind_per
        return obs, info

    def step(self, action: int):
        if self.wind_per == "step":
            self.wind_direction = self._sample_wind_direction()

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

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

        if (
            self.map_variant == "hidden_trap"
            and not terminated
            and not truncated
            and tuple(int(v) for v in self.agent_pos) in self._unsafe_branch_cells
        ):
            terminated = True
            reward = 0.0
            info["hidden_trap_triggered"] = True
        else:
            info["hidden_trap_triggered"] = False

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self._augment_obs(self.gen_obs())
        info["confounder"] = int(self.hidden_u)
        info["map_variant"] = self.map_variant
        info["wind_direction"] = int(self.wind_direction)
        info["wind_strength"] = float(self.wind_strength)
        info["wind_per"] = self.wind_per
        return obs, reward, terminated, truncated, info

    def _augment_obs(self, obs):
        if isinstance(obs, dict):
            obs = dict(obs)
            obs["agent_pos"] = np.array(self.agent_pos, dtype=np.int64)
        return obs

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
