"""Diagnose why AWR actor goes near despite critic preferring far at start.
Tests: (1) actor's action probs at start, (2) effect of sharper beta, (3) effect of one-hot actor.
"""
import numpy as np, sys
sys.path.insert(0, ".")
from collections import defaultdict, Counter, deque
from agents.contrastive_critic_numpy import ContrastiveCriticNumpy, _embed_backward, _embed_forward
from agents.goal_conditioned_actor_numpy import GoalConditionedActorNumpy

LAVA = {(3,2),(4,2),(5,2),(8,2),(9,2),(10,2)}
GOAL = (13,1); START = (1,1)
WALK = set()
WALK |= {(x,1) for x in range(1,14)}; WALK |= {(x,13) for x in range(1,14)}
WALK |= {(1,y) for y in range(1,14)}; WALK |= {(13,y) for y in range(1,14)}
WIND = (.1,.1,.1,.1,.6); GAMMA = 0.99
VALID = [0,1,2,6]
DIRV = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1), 4:(0,0)}

def prim(p,d,a):
    if a==0: return p,(d-1)%4,False
    if a==1: return p,(d+1)%4,False
    if a==2:
        dx,dy=DIRV[d]; t=(p[0]+dx,p[1]+dy)
        if t in LAVA: return t,d,True
        if t==GOAL: return t,d,True
        if t in WALK: return t,d,False
        return p,d,False
    return p,d,False
def wseq(d,w):
    if w==4: return [2]
    if d==w: return [2,2]
    if (d-2)%4==w: return [6]
    return [2,0,2,1] if (d-w) in (1,-3) else [2,1,2,0]
def astep(p,d,a,w):
    if a!=2: return prim(p,d,a)
    cp,cd=p,d
    for ac in wseq(d,w):
        cp,cd,t=prim(cp,cd,ac)
        if t: return cp,cd,True
    return cp,cd,False

# Load corrected worst-case data
d = np.load("data/corridor_worst_case_n1000_fixed.npz", allow_pickle=True)
S_wc = [np.asarray(s, dtype=np.float64) for s in d["episodes_states"]]
A_wc = [np.asarray(a, dtype=np.int64) for a in d["episodes_actions"]]
N_ACTIONS = int(d["n_actions"])

# Index counts at start
start_actions = Counter()
for s, a in zip(S_wc, A_wc):
    if len(a) > 0:
        a0 = int(a[0])
        start_actions[a0] += 1
nm = {0:"left", 1:"right(far)", 2:"forward(near)", 6:"done"}
print("data action counts at start (1,1,east):")
for k,v in sorted(start_actions.items()): print(f"  {nm.get(k,k)}: {v}")
total_starts = sum(start_actions.values())
print(f"  total: {total_starts}")

# Build indices + train strong critic (one-hot, uniform neg, balanced, tau=0.1)
def kk3(s): return (int(round(s[0])), int(round(s[1])), int(round(s[2])))
def kk2(s): return (int(round(s[0])), int(round(s[1])))
sa_idx = {}; g_idx = {}
for tr in S_wc:
    for s in tr:
        sa_idx.setdefault(kk3(s), len(sa_idx))
        g_idx.setdefault(kk2(s), len(g_idx))
Nsa, Ng = len(sa_idx), len(g_idx)
gcells_list = list(g_idx.keys())
def sa1(s):
    v = np.zeros(Nsa); k = kk3(s)
    if k in sa_idx: v[sa_idx[k]] = 1.0
    return v
def g1cell(cell):
    v = np.zeros(Ng)
    if cell in g_idx: v[g_idx[cell]] = 1.0
    return v
def g1(s): return g1cell(kk2(s))
anchors = [(ei, t) for ei, tr in enumerate(S_wc) for t in range(len(tr) - 1)]
bykey = defaultdict(list)
for ei, t in anchors:
    bykey[(kk3(S_wc[ei][t]), int(A_wc[ei][t]))].append((ei, t))
keys = list(bykey.keys())
TAU = 0.1
rng = np.random.default_rng(0)
def goff(m):
    o = int(rng.geometric(1 - GAMMA)); return min(max(o, 1), m)

print("\nTraining strong critic (30k steps)...")
critic = ContrastiveCriticNumpy(state_dim=Nsa, n_actions=N_ACTIONS, hidden=128, emb_dim=64,
                                tau=TAU, seed=0, goal_feat_dim=Ng)
K = 48; LR = 1e-3
for step in range(30000):
    sb=[];ab=[];sfb=[]
    for _ in range(64):
        kk = keys[rng.integers(len(keys))]
        lst = bykey[kk]
        ei, t = lst[rng.integers(len(lst))]
        tr = S_wc[ei]
        sb.append(sa1(tr[t])); ab.append(int(A_wc[ei][t]))
        off = goff(len(tr) - t - 1); sfb.append(g1(tr[t + off]))
    sb = np.array(sb); ab = np.array(ab); sfb = np.array(sfb)
    negc = [gcells_list[i] for i in rng.integers(0, len(gcells_list), K)]
    neg = np.stack([g1cell(c) for c in negc])
    h_sa, c_sa = critic._embed_sa(sb, ab)
    h_pos, c_pos = _embed_forward(sfb, critic.g_W1, critic.g_b1, critic.g_W2, critic.g_b2)
    h_neg, c_neg = _embed_forward(neg, critic.g_W1, critic.g_b1, critic.g_W2, critic.g_b2)
    B = 64
    lp = (h_sa * h_pos).sum(1) / TAU; ln = (h_sa @ h_neg.T) / TAU
    sp = 1 / (1 + np.exp(-np.clip(lp, -50, 50)))
    sn = 1 / (1 + np.exp(-np.clip(ln, -50, 50)))
    dlp = (sp - 1) / B; dln = sn / (B * K)
    d_h_sa = (dlp[:, None] * h_pos) / TAU + (dln @ h_neg) / TAU
    d_h_pos = (dlp[:, None] * h_sa) / TAU
    d_h_neg = (dln.T @ h_sa) / TAU
    g_sa = _embed_backward(d_h_sa, c_sa, critic.sa_W1, critic.sa_W2)
    g_p = _embed_backward(d_h_pos, c_pos, critic.g_W1, critic.g_W2)
    g_n = _embed_backward(d_h_neg, c_neg, critic.g_W1, critic.g_W2)
    critic.sa_W1 -= LR * g_sa[0]; critic.sa_b1 -= LR * g_sa[1]
    critic.sa_W2 -= LR * g_sa[2]; critic.sa_b2 -= LR * g_sa[3]
    for g in (g_p, g_n):
        critic.g_W1 -= LR * g[0]; critic.g_b1 -= LR * g[1]
        critic.g_W2 -= LR * g[2]; critic.g_b2 -= LR * g[3]

# Critic at start: all 7 actions
def scores_at_full(x, y, dd, goal_cell):
    hg = _embed_forward(g1cell(goal_cell)[None], critic.g_W1, critic.g_b1, critic.g_W2, critic.g_b2)[0]
    out = {}
    for a in range(N_ACTIONS):
        hsa, _ = critic._embed_sa(sa1(np.array([x,y,dd]))[None], np.array([a]))
        out[a] = float((hsa * hg).sum() / TAU)
    return out
print("\nCRITIC scores at start (toward goal (13,1)):")
sc_full = scores_at_full(1, 1, 0, (13,1))
for a in range(N_ACTIONS):
    tag = nm.get(a, f"a{a}") + (" ★greedy" if a == max(range(N_ACTIONS), key=lambda x: sc_full[x]) else "")
    print(f"  a={a} ({tag}): {sc_full[a]:+.3f}")

# Train AWR actor with different betas and run rollouts
def diragn(arr): a = arr.copy(); a[:, 2] = 0.0; return a
def scores_batch(s_raw, goal_raw):
    out = np.zeros((len(s_raw), N_ACTIONS))
    hg_all = _embed_forward(np.array([g1(g) for g in goal_raw]),
                            critic.g_W1, critic.g_b1, critic.g_W2, critic.g_b2)[0]
    senc = np.array([sa1(s) for s in s_raw])
    for a in range(N_ACTIONS):
        hsa, _ = critic._embed_sa(senc, np.full(len(s_raw), a))
        out[:, a] = (hsa * hg_all).sum(1) / TAU
    return out

def train_awr(beta, steps=15000, lr=2e-3, seed=1):
    act = GoalConditionedActorNumpy(state_dim=3, n_actions=N_ACTIONS, hidden=128,
                                    seed=seed, valid_actions=VALID)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        ids = rng.integers(0, len(anchors), 64); pr = [anchors[i] for i in ids]
        s = np.stack([S_wc[e][t] for e, t in pr]).astype(np.float64)
        a = np.array([int(A_wc[e][t]) for e, t in pr])
        sf = np.stack([S_wc[e][min(t + goff(len(S_wc[e]) - t - 1), len(S_wc[e]) - 1)]
                      for e, t in pr]).astype(np.float64)
        sc = scores_batch(s, sf)
        # IMPORTANT: baseline only over VALID actions to avoid OOD action contamination
        sc_valid = sc[:, VALID]
        Q = sc[np.arange(64), a]
        Vb = sc_valid.mean(1)
        adv = Q - Vb
        w = np.exp(np.clip(adv / beta, -10, 10)); w = np.clip(w, 0, 20)
        z2, (x, z1, h1) = act._forward(s, diragn(sf))
        z2s = z2 - z2.max(1, keepdims=True); ez = np.exp(z2s)
        p = ez / ez.sum(1, keepdims=True)
        oh = np.zeros_like(p); oh[np.arange(64), a] = 1.0
        dz2 = (w[:, None] * (p - oh)) / 64
        dW2 = h1.T @ dz2; db2 = dz2.sum(0)
        dh1 = dz2 @ act.W2.T; dz1 = dh1 * (z1 > 0)
        dW1 = x.T @ dz1; db1 = dz1.sum(0)
        act.W1 -= lr * dW1; act.b1 -= lr * db1
        act.W2 -= lr * dW2; act.b2 -= lr * db2
    return act

def actor_probs_at_start(act):
    p = act.action_probs(np.array([[1, 1, 0]], dtype=np.float64),
                         np.array([[13, 1, 0]], dtype=np.float64))[0]
    return p

def actor_rollout(act, force=None, seed=0, n=1500):
    rng = np.random.default_rng(seed); out = Counter(); rt = Counter()
    goal = np.array([13, 1, 0.0])
    for _ in range(n):
        pos, dd = START, 0; far = False; o = "timeout"
        for _ in range(200):
            st = np.array([pos[0], pos[1], dd], dtype=np.float64)
            a = act.greedy_action(st, goal, valid_actions=VALID)
            w = force if force is not None else int(rng.choice(5, p=WIND))
            pos, dd, term = astep(pos, dd, a, w)
            if pos[1] >= 12: far = True
            if term:
                o = "goal" if pos == GOAL else ("lava" if pos in LAVA else "x")
                break
        out[o] += 1; rt["far" if far else "near"] += 1
    return out, rt

print("\nSWEEP beta:")
for beta in [0.5, 0.2, 0.1, 0.05]:
    act = train_awr(beta)
    p = actor_probs_at_start(act)
    out, rt = actor_rollout(act)
    out_w, rt_w = actor_rollout(act, force=1)
    print(f"  beta={beta}: actor@start probs " +
          " ".join(f"{nm.get(a,'a'+str(a))}={p[a]:.2f}" for a in VALID))
    print(f"         natural: " + " ".join(f"{k}:{int(round(v/sum(out.values())*100))}%" for k,v in sorted(out.items())) +
          " | " + " ".join(f"{k}:{int(round(v/sum(rt.values())*100))}%" for k,v in sorted(rt.items())))
    print(f"         worst-S: " + " ".join(f"{k}:{int(round(v/sum(out_w.values())*100))}%" for k,v in sorted(out_w.items())) +
          " | " + " ".join(f"{k}:{int(round(v/sum(rt_w.values())*100))}%" for k,v in sorted(rt_w.items())))
print("DONE")
