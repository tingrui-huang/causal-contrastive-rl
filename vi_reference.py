"""
Formalized pessimistic VI for WindyCorridor.

Lifts the inline VI from tmp_repro_correct.py (lines 144-176) into a real module.
Intended target path in the repo: utils/vi_reference.py

It exposes TWO objects, because there are genuinely two different V's (the same
distinction you flagged earlier):

  solve_pessimistic_vi(...)      GREEDY optimal pessimistic planner (control).
                                 adv branch = true min over the iterate V.
                                 This is your tmp oracle/sanity-check V.
                                 -> reference for Gate 3c (true-min operator).

  pessimistic_occupancy(...)     POLICY-EVALUATION lower-bound occupancy d_low,
                                 under a FIXED policy. With adv_mode="vlower_argmin"
                                 its adversarial branch jumps to argmin_N v_lower,
                                 EXACTLY mirroring WorstCaseKernel.
                                 -> reference for Gate 3a (sampler self-consistency).

Acceptance check for the formalization itself: solve_pessimistic_vi returns
start_forward / start_right. Compare them to the values tmp_repro_correct.py
prints for qval((1,1,0),2) and qval((1,1,0),1). If they match, the lift is faithful.

CRITICAL signature (do not guess): enumerate_neighbors(pos, dir_, action),
                                   v_lower((x, y, dir)).
"""
from __future__ import annotations
import numpy as np

from utils.wind_dynamics import (
    analytical_step, enumerate_neighbors, is_goal, is_lava, WALKABLE_CELLS,
)
from utils.v_lower_oracle import v_lower
from utils.propensity import estimate_propensity, lookup
from utils.obs_transition import estimate_obs_transition

GOAL_POS = (13, 1)


def _all_states():
    return [(int(x), int(y), d) for (x, y) in WALKABLE_CELLS for d in range(4)]


def _make_qval(V, prop_tbl, obs_tbl, *, gamma, wind_dist, prop_default, adv_mode):
    dist = np.asarray(wind_dist, dtype=float)
    dist = dist / dist.sum()

    def Vget(state):
        p = (int(state[0]), int(state[1]))
        if is_lava(p):
            return 0.0
        if is_goal(p):
            return 1.0
        return V.get((int(state[0]), int(state[1]), int(state[2])), 0.0)

    def qval(s, a):
        s = (int(s[0]), int(s[1]), int(s[2]))
        a = int(a)
        pb = lookup(prop_tbl, s, a, default_p=prop_default)

        # observational branch (consistency): empirical P_obs, else analytical marginal
        key = (s, a)
        if key in obs_tbl:
            next_states, probs = obs_tbl[key]
            ov = float(sum(pr * Vget(ns) for ns, pr in zip(next_states, probs)))
        else:  # obs_fallback: marginal wind (mirrors the kernel)
            ov = 0.0
            for w, pw in enumerate(dist):
                np_, nd_, _t = analytical_step((s[0], s[1]), s[2], a, w)
                ov += pw * Vget((np_[0], np_[1], nd_))

        # adversarial branch: N(s,x) = enumerate_neighbors(pos, dir_, action)
        nbrs = enumerate_neighbors((s[0], s[1]), s[2], a)
        nb_states = [(np_[0], np_[1], nd_) for (np_, nd_, _t) in nbrs]
        if adv_mode == "true_min":
            worst = min(Vget(ns) for ns in nb_states)          # operator's own min
        elif adv_mode == "vlower_argmin":
            j = int(np.argmin([v_lower(ns) for ns in nb_states]))
            worst = Vget(nb_states[j])                          # mirrors the sampler
        else:
            raise ValueError(adv_mode)

        return gamma * (pb * ov + (1.0 - pb) * worst)

    return qval, Vget


def _solve(episode_states, episode_actions, *, n_actions, gamma, valid_actions,
           wind_dist, max_iters, tol, prop_default, adv_mode, policy):
    prop_tbl = estimate_propensity(episode_states, episode_actions, n_actions=n_actions)
    obs_tbl = estimate_obs_transition(episode_states, episode_actions)
    states = _all_states()
    V = {s: 0.0 for s in states}

    it_final = 0
    for it in range(max_iters):
        qval, _ = _make_qval(V, prop_tbl, obs_tbl, gamma=gamma, wind_dist=wind_dist,
                             prop_default=prop_default, adv_mode=adv_mode)
        delta = 0.0
        newV = {}
        for s in states:
            if is_goal((s[0], s[1])) or is_lava((s[0], s[1])):
                newV[s] = V[s]
                continue
            qs = [qval(s, a) for a in valid_actions]
            if policy is None:               # greedy (control)
                v_new = max(qs)
            else:                            # policy evaluation (occupancy)
                probs = policy(s)
                v_new = float(sum(probs[a] * qval(s, a) for a in valid_actions))
            newV[s] = v_new
            delta = max(delta, abs(v_new - V[s]))
        V = newV
        it_final = it + 1
        if delta < tol:
            break

    qval, _ = _make_qval(V, prop_tbl, obs_tbl, gamma=gamma, wind_dist=wind_dist,
                         prop_default=prop_default, adv_mode=adv_mode)
    return V, qval, valid_actions, it_final


def solve_pessimistic_vi(
    episode_states, episode_actions, *, n_actions,
    gamma=0.99, valid_actions=(0, 1, 2, 6),
    wind_dist=(0.1, 0.1, 0.1, 0.1, 0.6),
    max_iters=4000, tol=1e-9, prop_default=0.0,
):
    """Greedy optimal pessimistic planner (your tmp VI), true-min adv branch."""
    V, qval, va, it = _solve(
        episode_states, episode_actions, n_actions=n_actions, gamma=gamma,
        valid_actions=list(valid_actions), wind_dist=wind_dist, max_iters=max_iters,
        tol=tol, prop_default=prop_default, adv_mode="true_min", policy=None,
    )
    greedy = {s: int(max(va, key=lambda a: qval(s, a))) for s in V}
    return {
        "V": V, "qval": qval, "policy": greedy, "iterations": it,
        "start_forward": qval((1, 1, 0), 2),
        "start_right": qval((1, 1, 0), 1),
    }


def pessimistic_occupancy(
    episode_states, episode_actions, *, n_actions,
    policy=None, adv_mode="vlower_argmin",
    gamma=0.99, valid_actions=(0, 1, 2, 6),
    wind_dist=(0.1, 0.1, 0.1, 0.1, 0.6),
    max_iters=4000, tol=1e-9, prop_default=0.0,
):
    """Policy-evaluation d_low. Default policy = uniform over valid_actions.
    adv_mode='vlower_argmin' mirrors WorstCaseKernel (use for Gate 3a)."""
    va = list(valid_actions)
    if policy is None:
        p = np.zeros(n_actions)
        for a in va:
            p[a] = 1.0 / len(va)
        policy = lambda s: p
    V, qval, _, it = _solve(
        episode_states, episode_actions, n_actions=n_actions, gamma=gamma,
        valid_actions=va, wind_dist=wind_dist, max_iters=max_iters, tol=tol,
        prop_default=prop_default, adv_mode=adv_mode, policy=policy,
    )
    occ = {((s[0], s[1], s[2]), a): qval(s, a) for s in V for a in va}
    return {"V": V, "qval": qval, "occ": occ, "iterations": it}
