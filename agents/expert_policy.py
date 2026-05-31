"""
Expert policy for the WindyCorridor environment.

Behavior:
- At reset, pick NEAR with prob ``p_near`` (default 0.8), else FAR.
- NEAR: walk east along y=1; if on a lethal x (south wind would drift into lava),
        emit ``done`` to wait one step.
- FAR:  state machine x=1 south → y=13 east → x=13 north; always forward when
        facing the right way, otherwise turn. FAR is guaranteed lava-free under
        the wind distribution defined in ``envs/wind_dist.py``.

The expert is *deterministic given (route, position, wind direction)*. Stochastic
propensity ``P_obs(a | s)`` is induced solely by the wind, which is the latent
confounder ``U`` (excluded from the agent's observation).

Wind drift never changes net agent direction (causal_gym's wind action sequence
ends with ``second_turn`` that cancels ``first_turn``), so tracking direction
from the expert's own chosen actions is exact.
"""
from __future__ import annotations

import numpy as np
from minigrid.core.actions import Actions

from configs.corridor_defaults import P_NEAR
from envs.windy_corridor import LETHAL_X, START_POS

WIND_SOUTH = 1

DIR_EAST = 0
DIR_SOUTH = 1
DIR_NORTH = 3


class CorridorExpertPolicy:
    """Stateful expert. Call :meth:`reset` once per episode before stepping."""

    NEAR = "near"
    FAR = "far"

    def __init__(
        self,
        p_near: float = P_NEAR,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._p_near = float(p_near)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._route: str | None = None
        self._agent_dir: int = DIR_EAST

    def reset(
        self,
        *,
        seed: int | None = None,
        agent_dir: int = DIR_EAST,
    ) -> str:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._route = self.NEAR if self._rng.random() < self._p_near else self.FAR
        self._agent_dir = int(agent_dir)
        return self._route

    @property
    def route(self) -> str | None:
        return self._route

    @property
    def agent_dir(self) -> int:
        return self._agent_dir

    def __call__(self, agent_pos, wind_dir) -> int:
        if self._route is None:
            raise RuntimeError("CorridorExpertPolicy.reset() must be called first")

        pos = (int(agent_pos[0]), int(agent_pos[1]))
        wind = int(wind_dir)

        if self._route == self.NEAR:
            action = self._near_action(pos, wind)
        else:
            action = self._far_action(pos)

        if action == int(Actions.left):
            self._agent_dir = (self._agent_dir - 1) % 4
        elif action == int(Actions.right):
            self._agent_dir = (self._agent_dir + 1) % 4
        return action

    def _near_action(self, pos: tuple[int, int], wind: int) -> int:
        x, y = pos
        if self._agent_dir != DIR_EAST:
            return self._turn_toward(DIR_EAST)
        if y == 1 and x in LETHAL_X and wind == WIND_SOUTH:
            return int(Actions.done)
        return int(Actions.forward)

    def _far_action(self, pos: tuple[int, int]) -> int:
        target = self._far_target_dir(pos)
        if self._agent_dir != target:
            return self._turn_toward(target)
        return int(Actions.forward)

    @staticmethod
    def _far_target_dir(pos: tuple[int, int]) -> int:
        x, y = pos
        if x == 1 and y < 13:
            return DIR_SOUTH
        if y == 13 and x < 13:
            return DIR_EAST
        return DIR_NORTH  # x=13 ascending, or off-route fallback

    def _turn_toward(self, target_dir: int) -> int:
        delta = (target_dir - self._agent_dir) % 4
        if delta == 3:
            return int(Actions.left)
        return int(Actions.right)  # delta in {1, 2} → right (delta=2 takes 2 turns)
