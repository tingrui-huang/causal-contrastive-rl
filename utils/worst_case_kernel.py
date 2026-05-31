r"""
Worst-case transition kernel $\underline{T}_\pi$ from Theorem 2 (revised).

For each step given (S_t, X_t):
    (i)  w.p. P_b(X_t | S_t):     S_{t+1} ~ P_obs(· | S_t, X_t)
    (ii) w.p. 1 - P_b(X_t | S_t): S_{t+1} = argmin_{s'' in N(S_t, X_t)} V_lower(s'')

Discount γ is NOT in this kernel — it lives in the outer Geom(1-γ) stopping
time on S_T (see Theorem 2 statement). This module describes the pure
transition process only.

Implementation choices:
* (i) the observational branch. The theory uses the *conditional* P_obs(·|s,x)
  — the transition actually observed in the data after action x. Pass
  ``obs_transition_fn`` (see utils/obs_transition.py) to replay it from data.
  Under confounding this differs from the marginal wind roll: e.g. the expert
  only goes `forward` on a lethal cell when wind != south, so the data-conditional
  P_obs(lava | lethal, forward) = 0, whereas a fresh WIND_DIST roll injects ~10%
  lava. Without obs_transition_fn we fall back to the (legacy) marginal roll:
  sample wind ~ WIND_DIST and apply the analytical step.
* (ii) enumerates all 5 wind outcomes via enumerate_neighbors and picks the
  argmin by V_lower. Ties are broken by first-encountered.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from envs.wind_dist import CORRIDOR_WIND_DIST
from utils.wind_dynamics import analytical_step, enumerate_neighbors


class WorstCaseKernel:
    """Stateless callable implementing $\\underline{T}_\\pi$ at a single step."""

    def __init__(
        self,
        propensity_fn: Callable[[tuple[int, int, int], int], float],
        v_lower_fn: Callable[[tuple[int, int, int]], float],
        wind_dist: tuple[float, ...] = CORRIDOR_WIND_DIST,
        obs_transition_fn: Callable[
            [tuple[int, int, int], int, np.random.Generator],
            tuple[tuple[int, int, int], bool] | None,
        ]
        | None = None,
    ) -> None:
        self._propensity_fn = propensity_fn
        self._v_lower_fn = v_lower_fn
        self._wind_dist = np.asarray(wind_dist, dtype=np.float64)
        self._wind_dist /= self._wind_dist.sum()
        # When provided, branch (i) replays the empirical P_obs(s'|s,x) from data
        # instead of re-rolling a fresh wind. This is the theoretically correct
        # observational branch (see utils/obs_transition.py). Returns None for
        # (s,x) pairs never seen in the data -> we fall back to analytical Te.
        self._obs_transition_fn = obs_transition_fn

    def step(
        self,
        state: tuple[int, int, int],
        action: int,
        rng: np.random.Generator,
    ) -> tuple[tuple[int, int, int], bool, int, str]:
        """Return (next_state, terminated, sampled_wind, branch_label).

        ``branch_label`` is "obs" for (ii) and "adv" for (iii). ``sampled_wind``
        is the wind actually used (for (ii) the WIND_DIST sample; for (iii) we
        return ``-1`` since no real wind was drawn).
        """
        pos = (int(state[0]), int(state[1]))
        dir_ = int(state[2])

        p_b = self._propensity_fn(state, action)
        if rng.random() < p_b:
            # (i) observational branch: replay P_obs(s'|s,x) from data when available.
            if self._obs_transition_fn is not None:
                sampled = self._obs_transition_fn(state, action, rng)
                if sampled is not None:
                    (sp, term) = sampled
                    return (int(sp[0]), int(sp[1]), int(sp[2])), term, -1, "obs"
                # (s,x) unseen in data -> fall back to analytical marginal Te.
                wind = int(rng.choice(len(self._wind_dist), p=self._wind_dist))
                new_pos, new_dir, term = analytical_step(pos, dir_, action, wind)
                return (new_pos[0], new_pos[1], new_dir), term, wind, "obs_fallback"
            # No data sampler: analytical marginal Te (legacy behavior).
            wind = int(rng.choice(len(self._wind_dist), p=self._wind_dist))
            new_pos, new_dir, term = analytical_step(pos, dir_, action, wind)
            return (new_pos[0], new_pos[1], new_dir), term, wind, "obs"

        # (iii) adversarial branch — argmin V_lower over N(s, x)
        neighbors = enumerate_neighbors(pos, dir_, action)
        worst_idx = 0
        worst_val = self._v_lower_fn(
            (neighbors[0][0][0], neighbors[0][0][1], neighbors[0][1])
        )
        for i in range(1, len(neighbors)):
            (np_, nd_, _term) = neighbors[i]
            val = self._v_lower_fn((np_[0], np_[1], nd_))
            if val < worst_val:
                worst_val = val
                worst_idx = i
        new_pos, new_dir, term = neighbors[worst_idx]
        return (new_pos[0], new_pos[1], new_dir), term, -1, "adv"
