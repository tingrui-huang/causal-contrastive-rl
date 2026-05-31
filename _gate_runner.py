"""Throwaway gate harness for ccrl-windy-diagnose (Gates 1-3 + VI acceptance).

Read-only w.r.t. the repo's source. Stubs the missing `causal_gym` dependency
so the real formal modules (envs/utils) can be imported and asserted against.
Writes a machine-readable verdict to _gate_results.txt.
"""
from __future__ import annotations
import sys, types

# --- stub the missing causal_gym dep (gates never use the SCM wrapper) ---
def _stub(name):
    m = types.ModuleType(name); m.__path__ = []; sys.modules[name] = m; return m
cg = _stub("causal_gym")
cg_envs = _stub("causal_gym.envs")
cg_wm = _stub("causal_gym.envs.windy_minigrid")
cg_wm.WIND_DIST = (0.1, 0.1, 0.1, 0.1, 0.6)
cg_envs.WindyMiniGridSCM = object
cg_envs.WindyMiniGridPCH = object
cg_envs.windy_minigrid = cg_wm
cg.envs = cg_envs

import numpy as np
from envs import windy_corridor as wc
from utils import wind_dynamics as wd
from utils import v_lower_oracle as vo
from utils.worst_case_kernel import WorstCaseKernel
from utils.propensity import estimate_propensity, lookup
from utils.obs_transition import estimate_obs_transition, sample_obs_transition
import vi_reference as vir

GAMMA = 0.99
VALID = [0, 1, 2, 6]
START = (1, 1, 0)
GOAL = (13, 1)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)

# ============================================================ GATE 1
log("===== GATE 1  env / dynamics =====")
g1 = True
def c1(name, cond):
    global g1; g1 = g1 and bool(cond); log(("  PASS " if cond else "  FAIL "), name)
c1("SIZE==15", wc.SIZE == 15)
c1("START==(1,1)", wc.START_POS == (1, 1))
c1("GOAL==(13,1)", wc.GOAL_POS == (13, 1))
c1("LETHAL_X=={2,3,4,7,8,9}", set(wc.LETHAL_X) == {2, 3, 4, 7, 8, 9})
from envs.wind_dist import CORRIDOR_WIND_DIST
d = list(CORRIDOR_WIND_DIST)
c1("wind_dist sums to 1", abs(sum(d) - 1) < 1e-9)
c1("wind_dist[4]==0.6", abs(d[4] - 0.6) < 1e-9)
c1("each cardinal==0.1", all(abs(d[i] - 0.1) < 1e-9 for i in range(4)))
lethal_die = all((lambda r: r[2] and wd.is_lava(r[0]))(wd.analytical_step((x, 1), 0, 2, 1)) for x in wc.LETHAL_X)
still_safe = all(not wd.is_lava(wd.analytical_step((x, 1), 0, 2, 4)[0]) for x in wc.LETHAL_X)
c1("forward+south at lethal -> lava&term", lethal_die)
c1("forward+still at lethal -> safe", still_safe)
c1("enumerate_neighbors returns 5", len(wd.enumerate_neighbors((1, 1), 0, 2)) == 5)
c1("non-forward collapses to 1", len(set(wd.enumerate_neighbors((1, 1), 0, 1))) == 1)
c1("is_goal((13,1))", wd.is_goal((13, 1)))
c1("is_lava on LAVA member", wd.is_lava(next(iter(wd.LAVA_CELLS))))
p_death = 1 - 0.9 ** 6
c1("p_death_naive ~ 0.4686", abs(p_death - 0.4686) < 1e-3)
log("  LAVA_CELLS =", sorted(wd.LAVA_CELLS))
log("  p_death_naive =", round(p_death, 4))
log("GATE1:", "PASS" if g1 else "FAIL")

# ============================================================ GATE 2
log("\n===== GATE 2  oracle V_lower =====")
g2 = True
def c2(name, cond):
    global g2; g2 = g2 and bool(cond); log(("  PASS " if cond else "  FAIL "), name)
lava = next(iter(wd.LAVA_CELLS))
c2("v_lower(lava)==-1e9", vo.v_lower((lava[0], lava[1], 0)) == -1e9)
c2("bfs_distance(goal)==0", vo.bfs_distance(GOAL) == 0)
c2("v_lower(goal)==0", vo.v_lower((13, 1, 0)) == 0.0)
ds = [vo.bfs_distance((x, 1)) for x in range(13, 0, -1)]
c2("bfs increases away from goal", all(ds[i + 1] >= ds[i] for i in range(len(ds) - 1)))
c2("v_lower dir-invariant", vo.v_lower((5, 1, 0)) == vo.v_lower((5, 1, 3)) == vo.v_lower((5, 1, 1)))
c2("unreachable==-1e6", vo.v_lower((0, 0, 0)) == -1e6)
log("GATE2:", "PASS" if g2 else "FAIL")

# ============================================================ load obs data
log("\n----- loading observational data -----")
obs = np.load("data/corridor_expert_n1000.npz", allow_pickle=True)
S = [np.asarray(s) for s in obs["episodes_states"]]
A = [np.asarray(a) for a in obs["episodes_actions"]]
NA = int(obs["n_actions"])
log("  episodes =", len(S), " n_actions =", NA)

prop_tbl = estimate_propensity(S, A, n_actions=NA)
obs_tbl = estimate_obs_transition(S, A)
def propensity_fn(state, action):
    return lookup(prop_tbl, state, action, default_p=0.0)
def obs_transition_fn(state, action, rng):
    return sample_obs_transition(obs_tbl, state, action, rng)

def behavior_policy(s):
    p = prop_tbl.get((int(s[0]), int(s[1]), int(s[2])))
    if p is None:
        out = np.zeros(NA)
        for a in VALID: out[a] = 1.0 / len(VALID)
        return out
    return p
uni = np.zeros(NA)
for a in VALID: uni[a] = 1.0 / len(VALID)
def uniform_policy(s): return uni

# ============================================================ VI ACCEPTANCE
log("\n===== VI ACCEPTANCE (formalized vi_reference vs tmp/SUMMARY) =====")
res_greedy = vir.solve_pessimistic_vi(S, A, n_actions=NA, gamma=GAMMA, valid_actions=tuple(VALID))
log("  solve_pessimistic_vi (greedy, true_min): start_forward=%.4f start_right=%.4f (iters=%d)"
    % (res_greedy["start_forward"], res_greedy["start_right"], res_greedy["iterations"]))
occ_beh = vir.pessimistic_occupancy(S, A, n_actions=NA, policy=behavior_policy,
                                    adv_mode="true_min", gamma=GAMMA, valid_actions=tuple(VALID))
qb = occ_beh["qval"]
fwd_beh, rgt_beh = qb(START, 2), qb(START, 1)
log("  pessimistic_occupancy (behavior-policy, true_min): forward=%.4f right=%.4f" % (fwd_beh, rgt_beh))
log("  SUMMARY/tmp reference:                              forward=0.4550 right=0.6370")
accept = abs(fwd_beh - 0.455) < 0.02 and abs(rgt_beh - 0.637) < 0.02
log("  ACCEPTANCE (behavior-eval matches tmp):", "MATCH" if accept else "MISMATCH")

# ============================================================ GATE 3
log("\n===== GATE 3  worst-case sampler identity =====")
# invariant 2: kernel built with the ORACLE v_lower, never a critic/VI value
kernel = WorstCaseKernel(propensity_fn=propensity_fn, v_lower_fn=vo.v_lower,
                         obs_transition_fn=obs_transition_fn)
inv2 = kernel._v_lower_fn is vo.v_lower
log("  invariant 2 (kernel uses oracle v_lower):", "OK" if inv2 else "VIOLATED")

# VI_sampler: operator whose adv branch = argmin_N v_lower, fixed uniform policy
occ_samp = vir.pessimistic_occupancy(S, A, n_actions=NA, policy=uniform_policy,
                                     adv_mode="vlower_argmin", gamma=GAMMA, valid_actions=tuple(VALID))
qs = occ_samp["qval"]

# 3a: Monte-Carlo the kernel under the SAME uniform policy; d_hat = E[gamma^t_goal]
def mc_dhat(s0, a0, n=8000, maxs=300, seed=0):
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(n):
        state = tuple(int(v) for v in s0); a = a0; disc = 1.0; hit = 0.0
        for t in range(maxs):
            ns, term, _w, _b = kernel.step(state, a, rng)
            disc *= GAMMA
            if term:
                if (ns[0], ns[1]) == GOAL: hit = disc
                break
            state = ns
            a = int(rng.choice(NA, p=uniform_policy(state)))
        acc += hit
    return acc / n

log("  --- Gate 3a: sampler MC  vs  VI_sampler (vlower_argmin) ---")
test_states = [((1, 1, 0), 2, "start-forward/NEAR"),
               ((1, 1, 0), 1, "start-right/FAR"),
               ((1, 1, 0), 0, "start-left"),
               ((5, 1, 0), 2, "wait-cell-forward")]
g3a = True; max_err = 0.0
for st, a, lab in test_states:
    dh = mc_dhat(st, a)
    vv = qs(st, a)
    err = abs(dh - vv); max_err = max(max_err, err)
    ok = err < 0.02
    g3a = g3a and ok
    log("    %-22s a=%d  d_hat=%.4f  VI=%.4f  |err|=%.4f  %s"
        % (lab, a, dh, vv, err, "ok" if ok else "MISMATCH"))
log("  Gate3a max|d_hat-VI| = %.4f -> %s" % (max_err, "PASS" if g3a else "FAIL"))

# 3b: validity  VI_sampler <= d_true (true wind-marginalized occupancy)
log("  --- Gate 3b: validity  VI_sampler <= d_true ---")
dist = np.asarray(d, float); dist = dist / dist.sum()
states_all = [(int(x), int(y), dr) for (x, y) in wd.WALKABLE_CELLS for dr in range(4)]
Vtrue = {s: 0.0 for s in states_all}
def Vt(state):
    p = (int(state[0]), int(state[1]))
    if wd.is_lava(p): return 0.0
    if wd.is_goal(p): return 1.0
    return Vtrue.get((int(state[0]), int(state[1]), int(state[2])), 0.0)
def qtrue(s, a):
    ov = 0.0
    for w, pw in enumerate(dist):
        np_, nd_, _t = wd.analytical_step((s[0], s[1]), s[2], a, w)
        ov += pw * Vt((np_[0], np_[1], nd_))
    return GAMMA * ov
for _ in range(4000):
    newV = {}; delta = 0.0
    for s in states_all:
        if wd.is_goal((s[0], s[1])) or wd.is_lava((s[0], s[1])):
            newV[s] = Vtrue[s]; continue
        pr = uniform_policy(s)
        v = sum(pr[a] * qtrue(s, a) for a in VALID)
        newV[s] = v; delta = max(delta, abs(v - Vtrue[s]))
    Vtrue = newV
    if delta < 1e-9: break
viol = 0; worst_gap = 0.0; checked = 0
for s in states_all:
    if wd.is_goal((s[0], s[1])) or wd.is_lava((s[0], s[1])): continue
    for a in VALID:
        checked += 1
        gap = qs(s, a) - qtrue(s, a)        # want <= 0
        if gap > 1e-6: viol += 1
        worst_gap = max(worst_gap, gap)
g3b = viol == 0
log("    checked=%d  violations(VI_sampler>d_true)=%d  worst_gap=%.4f -> %s"
    % (checked, viol, worst_gap, "PASS" if g3b else "FAIL"))

# 3c: order-consistency argmin_N v_lower == argmin_N V_VI (greedy fixed point)
log("  --- Gate 3c: order-consistency (warning-only) ---")
Vvi = res_greedy["V"]
def Vvi_get(state):
    p = (int(state[0]), int(state[1]))
    if wd.is_lava(p): return 0.0
    if wd.is_goal(p): return 1.0
    return Vvi.get((int(state[0]), int(state[1]), int(state[2])), 0.0)
agree = 0; total = 0
for s in states_all:
    if wd.is_goal((s[0], s[1])) or wd.is_lava((s[0], s[1])): continue
    for a in VALID:
        nb = wd.enumerate_neighbors((s[0], s[1]), s[2], a)
        nb_states = [(p[0], p[1], dd) for (p, dd, _t) in nb]
        i_vl = int(np.argmin([vo.v_lower(ns) for ns in nb_states]))
        i_vi = int(np.argmin([Vvi_get(ns) for ns in nb_states]))
        # compare by resulting value (ties ok)
        total += 1
        if abs(vo.v_lower(nb_states[i_vl]) - vo.v_lower(nb_states[i_vi])) < 1e-9 or i_vl == i_vi:
            agree += 1
log("    argmin agreement = %d/%d (%.1f%%)" % (agree, total, 100.0 * agree / total))

log("\n===== VERDICT SUMMARY =====")
log("GATE1 env       :", "PASS" if g1 else "FAIL")
log("GATE2 v_lower   :", "PASS" if g2 else "FAIL")
log("VI acceptance   :", "MATCH" if accept else "MISMATCH")
log("GATE3a sampler  :", "PASS" if g3a else "FAIL", "(max|err|=%.4f)" % max_err)
log("GATE3b validity :", "PASS" if g3b else "FAIL", "(viol=%d)" % viol)

with open("_gate_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
print("\nWROTE _gate_results.txt")
