"""
Toy experiment: verify a causal lower bound on the discounted goal-occupancy d^pi.

Setup
-----
- 3 states (0, 1, 2), 2 actions (0, 1), goal s_g = 2, gamma = 0.9.
- Binary confounder U with P(U|s) = uniform (0.5 / 0.5).
- Behavior policy mu(a=0|s, U=0) = 0.8, mu(a=0|s, U=1) = 0.2.
- Confounded transition p(s'|s, a, U=u): two regimes, action effect is small
  inside each regime but the regimes themselves differ a lot. This is the
  textbook confounding pattern: in observational data action looks important
  because it is correlated with U; under do(a) it is not.
- True transition p_true(s'|s, a) = E_U[p(s'|s, a, U)]   (P(U|s) = 0.5).
- Observational transition p_e(s'|s, a) = E_{U|s,a}[p(s'|s, a, U)]
  with P(U|s, a) computed via Bayes from mu and P(U|s) = 0.5.
- Propensity score rho(a|s) = E_U[mu(a|s, U)] = 0.5 mu(a|s,U=0) + 0.5 mu(a|s,U=1).

Quantities
----------
1. d_true:    standard VI on the goal indicator under p_true.
2. d_e:       same VI but using observational p_e -- biased by confounding.
3. d_lower:   VI on the causal Bellman equation
                d(s,a) = (1-gamma) 1[s=g]
                       + gamma * rho(a|s)   * E_{p_e}[V(s')]
                       + gamma * (1-rho(.)) * min_{s': p_e(s'|s,a)>0} V(s')
              where V(s') = sum_{a'} pi(a'|s') d(s',a').
              (min restricted to support of p_e, assuming confounder
               does not change transition support.)
4. baseline:  rho(a|s) * d_e(s,a) -- the trivial propensity-weighted value.

Target policy pi is uniform (0.5, 0.5).

Claims to verify numerically
----------------------------
(C1)  d_lower(s,a) <= d_true(s,a)              -- valid lower bound on the truth
(C2)  d_lower(s,a) >= rho(a|s) * d_e(s,a)      -- tighter than the trivial baseline
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- problem data

N_STATES = 3
N_ACTIONS = 2
GOAL = 2
GAMMA = 0.9
TOL = 1e-12

# Behavior policy mu(a=0 | s, U=u). mu(a=1 | s, u) = 1 - mu(a=0 | s, u).
MU_A0_U0 = 0.8
MU_A0_U1 = 0.2


def behavior_mu(a: int, u: int) -> float:
    p_a0 = MU_A0_U0 if u == 0 else MU_A0_U1
    return p_a0 if a == 0 else 1.0 - p_a0


# Confounded transition p[u, s, a, s'].
# Regime U=0: mostly stays put. Regime U=1: advances toward the goal.
# Inside each regime, the two actions are nearly identical -- so under do(a)
# the action has little causal effect, but observationally a=1 looks great
# (because a=1 is strongly correlated with U=1 via the behavior policy).
p_su = np.zeros((2, N_STATES, N_ACTIONS, N_STATES))

# U = 0 (stuck regime)
p_su[0, 0, 0] = [0.90, 0.10, 0.00]
p_su[0, 0, 1] = [0.85, 0.15, 0.00]
p_su[0, 1, 0] = [0.10, 0.90, 0.00]
p_su[0, 1, 1] = [0.10, 0.85, 0.05]
p_su[0, 2, 0] = [0.00, 0.00, 1.00]
p_su[0, 2, 1] = [0.00, 0.00, 1.00]

# U = 1 (advancing regime)
p_su[1, 0, 0] = [0.20, 0.70, 0.10]
p_su[1, 0, 1] = [0.15, 0.70, 0.15]
p_su[1, 1, 0] = [0.00, 0.30, 0.70]
p_su[1, 1, 1] = [0.00, 0.25, 0.75]
p_su[1, 2, 0] = [0.00, 0.00, 1.00]
p_su[1, 2, 1] = [0.00, 0.00, 1.00]

assert np.allclose(p_su.sum(axis=-1), 1.0), "every transition row must sum to 1"


# ---------------------------------------------------------------- distributions

def compute_p_true() -> np.ndarray:
    """p_true(s' | s, a) = E_U[p(s' | s, a, U)] under P(U|s) = 0.5."""
    return 0.5 * p_su[0] + 0.5 * p_su[1]  # shape (S, A, S')


def compute_p_obs() -> np.ndarray:
    """p_e(s' | s, a) = sum_u P(U=u|s,a) p(s'|s,a,U=u).

    Bayes:
        P(U=u|s,a) = mu(a|s,U=u) P(U=u|s)
                     ----------------------------------
                     sum_{u'} mu(a|s,U=u') P(U=u'|s)
        with P(U|s) = 0.5, the 0.5 cancels.
    """
    p_e = np.zeros((N_STATES, N_ACTIONS, N_STATES))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            mu0 = behavior_mu(a, 0)
            mu1 = behavior_mu(a, 1)
            denom = mu0 + mu1
            w0, w1 = mu0 / denom, mu1 / denom
            p_e[s, a] = w0 * p_su[0, s, a] + w1 * p_su[1, s, a]
    return p_e


def compute_propensity() -> np.ndarray:
    """rho(a|s) = E_U[mu(a|s, U)]."""
    rho = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            rho[s, a] = 0.5 * behavior_mu(a, 0) + 0.5 * behavior_mu(a, 1)
    return rho


# ---------------------------------------------------------------- value iteration

def value_iteration(p: np.ndarray, pi: np.ndarray, max_iter: int = 10_000) -> np.ndarray:
    """Solve d(s,a) = (1-gamma) 1[s=g] + gamma sum_{s'} p(s'|s,a) sum_{a'} pi(a'|s') d(s',a')."""
    d = np.zeros((N_STATES, N_ACTIONS))
    reward = np.zeros((N_STATES, N_ACTIONS))
    reward[GOAL, :] = 1.0 - GAMMA
    for _ in range(max_iter):
        v = (pi * d).sum(axis=1)              # V(s') = sum_a' pi d
        nxt = reward + GAMMA * np.einsum("sax,x->sa", p, v)
        if np.max(np.abs(nxt - d)) < TOL:
            return nxt
        d = nxt
    return d


def causal_lower_bound_vi(
    p_e: np.ndarray, rho: np.ndarray, pi: np.ndarray, max_iter: int = 10_000
) -> np.ndarray:
    """Solve the causal pessimistic Bellman equation."""
    d = np.zeros((N_STATES, N_ACTIONS))
    reward = np.zeros((N_STATES, N_ACTIONS))
    reward[GOAL, :] = 1.0 - GAMMA
    reachable = p_e > 0  # (S, A, S') — support of observational transition
    for _ in range(max_iter):
        v = (pi * d).sum(axis=1)              # (S,)
        v_broadcast = np.where(reachable, v[np.newaxis, np.newaxis, :], np.inf)
        v_min = v_broadcast.min(axis=2)       # min over reachable s' per (s,a)
        e_v = np.einsum("sax,x->sa", p_e, v)  # E_{p_e}[V(s')]
        nxt = reward + GAMMA * (rho * e_v + (1.0 - rho) * v_min)
        if np.max(np.abs(nxt - d)) < TOL:
            return nxt
        d = nxt
    return d


# ---------------------------------------------------------------- pretty print

def fmt_d(name: str, d: np.ndarray) -> str:
    out = [f"  {name}"]
    out.append(f"    {'state':>5} {'a=0':>10} {'a=1':>10}")
    for s in range(N_STATES):
        out.append(f"    {s:>5} {d[s, 0]:>10.6f} {d[s, 1]:>10.6f}")
    return "\n".join(out)


def fmt_p(name: str, p: np.ndarray) -> str:
    out = [f"  {name}  (rows: s' | columns: a, blocks: s)"]
    out.append(f"    s | a |  s'=0     s'=1     s'=2")
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            row = "  ".join(f"{p[s, a, sp]:.4f}" for sp in range(N_STATES))
            out.append(f"    {s} | {a} |  {row}")
    return "\n".join(out)


# ---------------------------------------------------------------- driver

def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    p_true = compute_p_true()
    p_e = compute_p_obs()
    rho = compute_propensity()
    pi = np.full((N_STATES, N_ACTIONS), 1.0 / N_ACTIONS)

    d_true = value_iteration(p_true, pi)
    d_e = value_iteration(p_e, pi)
    d_lower = causal_lower_bound_vi(p_e, rho, pi)
    baseline = rho * d_e

    print(f"[ToyExp] gamma={GAMMA}  goal={GOAL}  pi=uniform")
    print(f"[ToyExp] mu(a=0|s,U=0)={MU_A0_U0}  mu(a=0|s,U=1)={MU_A0_U1}")
    print()
    print(fmt_p("p_true(s'|s,a)", p_true))
    print()
    print(fmt_p("p_e(s'|s,a)  -- observational, confounded", p_e))
    print()
    print("  rho(a|s):")
    print(rho)
    print()
    print(fmt_d("d_true   (ground truth, under p_true)", d_true))
    print()
    print(fmt_d("d_e      (observational, under p_e -- biased)", d_e))
    print()
    print(fmt_d("d_lower  (our causal lower bound)", d_lower))
    print()
    print(fmt_d("baseline (rho * d_e)", baseline))
    print()

    # --- bound checks -------------------------------------------------
    eps = 1e-9
    valid = bool(np.all(d_lower <= d_true + eps))
    tighter = bool(np.all(d_lower >= baseline - eps))
    valid_gap = d_true - d_lower
    bound_gap = d_lower - baseline

    print("[ToyExp] (C1) d_lower <= d_true                : ", valid)
    print(f"           gap d_true - d_lower    min={valid_gap.min(): .6f}  max={valid_gap.max(): .6f}  mean={valid_gap.mean(): .6f}")
    print("[ToyExp] (C2) d_lower >= rho * d_e (baseline)  : ", tighter)
    print(f"           gap d_lower - baseline  min={bound_gap.min(): .6f}  max={bound_gap.max(): .6f}  mean={bound_gap.mean(): .6f}")
    print()

    if valid and tighter:
        print("[ToyExp] PASS -- both bounds verified on this instance.")
    else:
        print("[ToyExp] BOUND VIOLATED on at least one (s, a) -- inspect the tables above.")


if __name__ == "__main__":
    main()
