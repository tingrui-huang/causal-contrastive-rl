"""WindyCorridor env exposure.

Two entry points:

* ``make_windy_corridor_scm(...)``: factory returning a ``WindyMiniGridSCM`` —
  use this everywhere except gym-introspection code paths.
* gym registration ``CausalContrastive-WindyCorridor-15x15-v0``: registers the
  bare ``MiniGridEnv`` (no wind). Useful for ``gym.make`` introspection and as
  a building block for the factory above.
"""

from __future__ import annotations

from typing import Any, Callable

from gymnasium.envs.registration import register

from causal_gym.envs import WindyMiniGridSCM, WindyMiniGridPCH
from .windy_corridor import WindyCorridorEnv


WINDY_CORRIDOR_GYM_ID = "CausalContrastive-WindyCorridor-15x15-v0"


def register_causal_envs() -> None:
    register(
        id=WINDY_CORRIDOR_GYM_ID,
        entry_point="envs.windy_corridor:WindyCorridorEnv",
        kwargs={"size": 15},
    )


register_causal_envs()


def make_windy_corridor_scm(
    *,
    policy: Callable | None = None,
    show_wind: bool = False,
    wind_dist: Callable | tuple | None = None,
    render_mode: str | None = "rgb_array",
    **env_kwargs: Any,
) -> WindyMiniGridSCM:
    """Build a WindyCorridorEnv and wrap with causal_gym's WindyMiniGridSCM.

    ``wind_dist=None`` (default) → upstream's ``WIND_DIST = (.1,.1,.1,.1,.6)``,
    aligned with ``test_windyminigrid.ipynb``. Pass a tuple/callable only to
    deliberately override for sweeps.
    """
    base = WindyCorridorEnv(render_mode=render_mode, **env_kwargs)
    kwargs: dict[str, Any] = {"policy": policy, "show_wind": show_wind}
    if wind_dist is not None:
        kwargs["wind_dist"] = wind_dist
    return WindyMiniGridSCM(env=base, **kwargs)


def make_windy_corridor_pch(
    *,
    policy: Callable | None = None,
    show_wind: bool = False,
    wind_dist: Callable | tuple | None = None,
    render_mode: str | None = "rgb_array",
    **env_kwargs: Any,
) -> WindyMiniGridPCH:
    """Build a WindyCorridorEnv and wrap with causal_gym's WindyMiniGridPCH."""
    base = WindyCorridorEnv(render_mode=render_mode, **env_kwargs)
    kwargs: dict[str, Any] = {"policy": policy, "show_wind": show_wind}
    if wind_dist is not None:
        kwargs["wind_dist"] = wind_dist
    return WindyMiniGridPCH(env=base, **kwargs)
