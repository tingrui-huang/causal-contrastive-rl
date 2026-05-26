"""
Wind distribution for WindyCorridor.

We use causal_gym's module-level default ``WIND_DIST = (.1, .1, .1, .1, .6)``
verbatim — the same distribution used throughout ``test_windyminigrid.ipynb``.
By re-exporting (not copying), the value cannot drift from upstream.
"""
from causal_gym.envs.windy_minigrid import WIND_DIST

CORRIDOR_WIND_DIST = WIND_DIST
