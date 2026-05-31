"""One-hot AWR actor on the strong critic. Plus deep-far diagnosis.

Compares:
  - β=0.1 RAW actor (flips first step but timeouts)
  - β=0.5 ONE-HOT actor (predicted: should flip cleanly without sharp β)

Diagnoses: at deep far states, what does each actor do?
"""
import numpy as np, sys
sys.path.insert(0, ".")
from collections import defaultdict, Counter
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

d = np.load("data/corridor_worst_case_n1000_fixed.npz", allow_pickle=True)
S_wc = [np.asarray(s, dtype=np.float64) for s in d["episodes_states"]]
A_wc = [np.asarray(a, dtype=np.int64) for a in d["episodes_actions"]]
N_ACTIONS = int(d["n_actions"])

def kk3(s): return (int(round(s[0])), int(round(s[1])), int(round(s[2])))
def kk2(s): return (int(round(s[0])), int(round(s[1])))
sa_idx = {}; g_idx = {}
for tr in S_wc:
    for s in tr:
        sa_idx.setdefault(kk3(s), len(sa_idx))
        g_idx.setdefault(kk2(s), len(g_idx))
# Ensure goal cell is in g_idx
g_idx.setdefault(GOAL, len(g_idx))
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

print("Training strong critic (one-hot + uniform-neg + balanced, 30k steps)...")
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
print("  critic done")

def scores_batch_enc(s_enc, sf_enc):
    """Compute f(s,a,g) for all 7 actions given pre-encoded inputs."""
    out = np.zeros((len(s_enc), N_ACTIONS))
    hg_all = _embed_forward(sf_enc, critic.g_W1, critic.g_b1, critic.g_W2, critic.g_b2)[0]
    for a in range(N_ACTIONS):
        hsa, _ = critic._embed_sa(s_enc, np.full(len(s_enc), a))
        out[:, a] = (hsa * hg_all).sum(1) / TAU
    return out

def train_awr(beta, encoding, steps=15000, lr=2e-3, seed=1):
    """encoding: 'raw' or 'onehot'."""
    if encoding == "raw":
        act = GoalConditionedActorNumpy(state_dim=3, goal_dim=3, n_actions=N_ACTIONS, hidden=128,
                                        seed=seed, valid_actions=VALID)
    else:
        act = GoalConditionedActorNumpy(state_dim=Nsa, goal_dim=Ng, n_actions=N_ACTIONS, hidden=128,
                                        seed=seed, valid_actions=VALID)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        ids = rng.integers(0, len(anchors), 64); pr = [anchors[i] for i in ids]
        s_raw = np.stack([S_wc[e][t] for e, t in pr]).astype(np.float64)
        a = np.array([int(A_wc[e][t]) for e, t in pr])
        sf_raw = np.stack([S_wc[e][min(t + goff(len(S_wc[e]) - t - 1), len(S_wc[e]) - 1)]
                           for e, t in pr]).astype(np.float64)
        # Encode for critic (always one-hot, since critic is one-hot)
        s_enc_c = np.array([sa1(x) for x in s_raw])
        sf_enc_c = np.array([g1(x) for x in sf_raw])
        sc = scores_batch_enc(s_enc_c, sf_enc_c)
        sc_v = sc[:, VALID]
        Q = sc[np.arange(64), a]
        Vb = sc_v.mean(1)
        adv = Q - Vb
        w = np.exp(np.clip(adv / beta, -10, 10)); w = np.clip(w, 0, 20)
        # Actor input: depends on encoding
        if encoding == "raw":
            s_in = s_raw.copy()
            sf_in = sf_raw.copy(); sf_in[:, 2] = 0.0  # dir-agnostic
        else:
            s_in = s_enc_c
            sf_in = sf_enc_c
        z2, (xc, z1, h1) = act._forward(s_in, sf_in)
        z2s = z2 - z2.max(1, keepdims=True); ez = np.exp(z2s)
        p = ez / ez.sum(1, keepdims=True)
        oh = np.zeros_like(p); oh[np.arange(64), a] = 1.0
        dz2 = (w[:, None] * (p - oh)) / 64
        dW2 = h1.T @ dz2; db2 = dz2.sum(0)
        dh1 = dz2 @ act.W2.T; dz1 = dh1 * (z1 > 0)
        dW1 = xc.T @ dz1; db1 = dz1.sum(0)
        act.W1 -= lr * dW1; act.b1 -= lr * db1
        act.W2 -= lr * dW2; act.b2 -= lr * db2
    return act

def actor_probs(act, state_xyd, encoding):
    if encoding == "raw":
        s_in = np.array([list(state_xyd)], dtype=np.float64)
        sf_in = np.array([[GOAL[0], GOAL[1], 0]], dtype=np.float64)
    else:
        s_in = sa1(np.array(state_xyd))[None]
        sf_in = g1cell(GOAL)[None]
    return act.action_probs(s_in, sf_in)[0]

def rollout(act, encoding, force=None, seed=0, n=1500):
    rng = np.random.default_rng(seed); out = Counter(); rt = Counter()
    for _ in range(n):
        pos, dd = START, 0; far = False; o = "timeout"
        for _ in range(200):
            state_xyd = (pos[0], pos[1], dd)
            if encoding == "raw":
                s_in = np.array([list(state_xyd)], dtype=np.float64)
                sf_in = np.array([[GOAL[0], GOAL[1], 0]], dtype=np.float64)
            else:
                s_in = sa1(np.array(state_xyd))[None]
                sf_in = g1cell(GOAL)[None]
            z2, _ = act._forward(s_in, sf_in)
            logits = z2[0]
            mask = np.full(N_ACTIONS, -np.inf); mask[VALID] = logits[VALID]
            a = int(np.argmax(mask))
            w = force if force is not None else int(rng.choice(5, p=WIND))
            pos, dd, term = astep(pos, dd, a, w)
            if pos[1] >= 12: far = True
            if term:
                o = "goal" if pos == GOAL else ("lava" if pos in LAVA else "x")
                break
        out[o] += 1; rt["far" if far else "near"] += 1
    return out, rt

nm = {0:"left", 1:"right(far)", 2:"forward(near)", 6:"done"}

print("\n=== Training and evaluating actors ===")
configs = [
    ("raw", 0.5),
    ("raw", 0.1),
    ("onehot", 0.5),
    ("onehot", 0.2),
]
for enc, beta in configs:
    act = train_awr(beta, enc)
    p0 = actor_probs(act, (1,1,0), enc)
    print(f"\n[{enc} actor, beta={beta}]")
    print(f"  probs@start (1,1,东): " + " ".join(f"{nm.get(a,'a'+str(a))}={p0[a]:.2f}" for a in VALID))
    # Diagnose deep far states
    for ds in [(1,5,1), (1,10,1), (1,13,1), (13,8,3)]:
        p = actor_probs(act, ds, enc)
        print(f"  probs@{ds}: " + " ".join(f"{nm.get(a,'a'+str(a))}={p[a]:.2f}" for a in VALID))
    out, rt = rollout(act, enc); n = sum(out.values())
    print(f"  natural: " + " ".join(f"{k}:{int(round(v/n*100))}%" for k,v in sorted(out.items())) +
          " | " + " ".join(f"{k}:{int(round(v/sum(rt.values())*100))}%" for k,v in sorted(rt.items())))
    out_w, rt_w = rollout(act, enc, force=1); n_w = sum(out_w.values())
    print(f"  worst-S: " + " ".join(f"{k}:{int(round(v/n_w*100))}%" for k,v in sorted(out_w.items())) +
          " | " + " ".join(f"{k}:{int(round(v/sum(rt_w.values())*100))}%" for k,v in sorted(rt_w.items())))
print("DONE")
