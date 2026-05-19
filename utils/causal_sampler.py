"""Theorem 2 recursive sampler from the causal-contrastive paper.

Generates samples Sf ~ d̲^π(· | s, x), the pessimistic reachability distribution
induced by the Manski-bound Bellman operator T̲. Used as positives for the
contrastive critic, which then converges to log[d̲^π / p] (Theorem 3) and bakes
pessimism into the critic itself — no actor-side modification needed.

Procedure (slide 5 of the PDF):
    (i)   w.p. (1 - γ): output Sf = s, terminate
    (ii)  w.p. γ · P̂(x | s): observational transition
              s' ~ P_obs(· | s, x), x' ~ behavior policy
    (iii) w.p. γ · (1 - P̂(x | s)): pessimistic transition
              s' = argmin over neighbors of V_f, x' ~ behavior policy

The new action x' is sampled from the empirical behavior policy (propensity table),
not the learned actor. This avoids the chicken-and-egg of needing an eval goal
to query a goal-conditioned actor mid-recursion, and is consistent with the
critic learning d̲^{π_behavior}.
"""
from __future__ import annotations

import numpy as np

from utils.domain_knowledge import POS_X_IDX, POS_Y_IDX, reachable_positions
from utils.obs_transition import sample_obs_transition
from utils.propensity import propensity, sample_action_from_propensity
from utils.value_function import lookup_v_f


def _cell_of(s: np.ndarray) -> tuple[int, int]:
    return int(round(s[POS_X_IDX])), int(round(s[POS_Y_IDX]))


def sample_Sf_thm2(
    s_anchor: np.ndarray,
    a_anchor: int,
    *,
    propensity_table,
    obs_transition_table,
    v_f_table,
    pos_index,
    walkable_cells: set[tuple[int, int]],
    n_actions: int,
    rng: np.random.Generator,
    gamma: float = 0.95,
    max_manhattan: int = 2,
    max_depth: int = 100,
    v_f_fallback: float = 0.0,
    diag: dict | None = None,
) -> np.ndarray:
    """Run the Thm 2 recursive procedure from (s_anchor, a_anchor), return Sf.

    If diag is provided, accumulate per-call counts:
        diag["calls"]              total invocations
        diag["terminations"]       count of (i) firings
        diag["obs_branches"]       count of (ii) firings
        diag["pess_branches"]      count of (iii) firings
        diag["max_depth_hits"]     count of max_depth cutoffs
        diag["depth_sum"]          sum of recursion depths
    """
    if diag is not None:
        diag["calls"] = diag.get("calls", 0) + 1

    s = np.asarray(s_anchor, dtype=np.float64)
    x = int(a_anchor)

    for depth in range(max_depth):
        # (i) terminate
        if rng.random() < (1.0 - gamma):
            if diag is not None:
                diag["terminations"] = diag.get("terminations", 0) + 1
                diag["depth_sum"] = diag.get("depth_sum", 0) + depth
            return s

        p_hat = propensity(s, x, propensity_table, n_actions, fallback_uniform=True)

        if rng.random() < p_hat:
            # (ii) observational
            s_next = sample_obs_transition(s, x, obs_transition_table, rng, fallback_state=s)
            if diag is not None:
                diag["obs_branches"] = diag.get("obs_branches", 0) + 1
        else:
            # (iii) pessimistic: argmin V_f over walkable physical neighbors
            cur_cell = _cell_of(s)
            neighbor_cells = reachable_positions(
                cur_cell[0], cur_cell[1], max_manhattan, walkable=walkable_cells
            )
            walkable_neighbors = [c for c in neighbor_cells if c in walkable_cells]

            if not walkable_neighbors:
                # degenerate: stay put
                s_next = s.copy()
            else:
                v_values = [lookup_v_f(c, v_f_table, fallback=v_f_fallback)
                            for c in walkable_neighbors]
                worst_idx = int(np.argmin(v_values))
                worst_cell = walkable_neighbors[worst_idx]
                if worst_cell in pos_index and len(pos_index[worst_cell]) > 0:
                    pool = pos_index[worst_cell]
                    s_next = pool[int(rng.integers(0, len(pool)))]
                else:
                    s_next = s.copy()
            if diag is not None:
                diag["pess_branches"] = diag.get("pess_branches", 0) + 1

        # New action from behavior policy (empirical propensity)
        x_next = sample_action_from_propensity(s_next, propensity_table, n_actions, rng)

        s, x = np.asarray(s_next, dtype=np.float64), int(x_next)

    # max_depth hit — return current s
    if diag is not None:
        diag["max_depth_hits"] = diag.get("max_depth_hits", 0) + 1
        diag["depth_sum"] = diag.get("depth_sum", 0) + max_depth
    return s


def fresh_diag() -> dict:
    return {
        "calls": 0,
        "terminations": 0,
        "obs_branches": 0,
        "pess_branches": 0,
        "max_depth_hits": 0,
        "depth_sum": 0,
    }


def diag_summary(diag: dict) -> dict:
    calls = max(diag.get("calls", 0), 1)
    obs = diag.get("obs_branches", 0)
    pess = diag.get("pess_branches", 0)
    total_branches = max(obs + pess, 1)
    return {
        "calls": diag.get("calls", 0),
        "mean_depth": diag.get("depth_sum", 0) / calls,
        "max_depth_hit_rate": diag.get("max_depth_hits", 0) / calls,
        "obs_branch_rate": obs / total_branches,
        "pess_branch_rate": pess / total_branches,
    }
