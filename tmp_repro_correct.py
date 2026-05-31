"""End-to-end re-derivation with corrected lava set.

Inline dynamics (avoids causal_gym dep), correct lava {(3,2),(4,2),(5,2),(8,2),(9,2),(10,2)}.
"""
import numpy as np, json, sys
sys.path.insert(0, ".")
from collections import defaultdict, Counter, deque
from agents.contrastive_critic_numpy import ContrastiveCriticNumpy, _embed_backward, _embed_forward
from agents.goal_conditioned_actor_numpy import GoalConditionedActorNumpy

# ============ CORRECTED dynamics ============
LAVA = {(3,2),(4,2),(5,2),(8,2),(9,2),(10,2)}        # FIXED from env _y2_pattern
GOAL = (13,1); START = (1,1)
WALK = set()
WALK |= {(x,1) for x in range(1,14)}
WALK |= {(x,13) for x in range(1,14)}
WALK |= {(1,y) for y in range(1,14)}
WALK |= {(13,y) for y in range(1,14)}
WIND = (.1,.1,.1,.1,.6); GAMMA = 0.99
VALID = [0,1,2,6]
DIRV = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1), 4:(0,0)}

def prim(p, d, a):
    if a == 0: return p, (d-1) % 4, False
    if a == 1: return p, (d+1) % 4, False
    if a == 2:
        dx, dy = DIRV[d]; t = (p[0]+dx, p[1]+dy)
        if t in LAVA: return t, d, True
        if t == GOAL: return t, d, True
        if t in WALK: return t, d, False
        return p, d, False
    return p, d, False

def wseq(d, w):
    if w == 4: return [2]
    if d == w: return [2,2]
    if (d-2) % 4 == w: return [6]
    return [2,0,2,1] if (d-w) in (1,-3) else [2,1,2,0]

def astep(p, d, a, w):
    if a != 2: return prim(p, d, a)
    cp, cd = p, d
    for ac in wseq(d, w):
        cp, cd, t = prim(cp, cd, ac)
        if t: return cp, cd, True
    return cp, cd, False

def neighbors(p, d, a): return [astep(p, d, a, w) for w in range(5)]

# ============ V_lower (BFS) ============
def bfs():
    dist = {GOAL: 0}; q = deque([GOAL])
    while q:
        x, y = q.popleft(); dd = dist[(x,y)]
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            n = (x+dx, y+dy)
            if n in WALK and n not in dist: dist[n] = dd+1; q.append(n)
    return dist
BFS = bfs()
def V_lower(cell):
    if cell in LAVA: return -1e9
    d = BFS.get(cell)
    return -1e6 if d is None else float(-d)

# ============ Pb and Pobs from observational data ============
print("[1/6] estimating Pb and Pobs ...")
obs = np.load("data/corridor_expert_n1000.npz", allow_pickle=True)
S_obs = [np.asarray(s) for s in obs["episodes_states"]]
A_obs = [np.asarray(a) for a in obs["episodes_actions"]]
N_ACTIONS = int(obs["n_actions"])
def k3(s): return (int(round(s[0])), int(round(s[1])), int(round(s[2])))
sa_cnt = defaultdict(Counter); pobs_cnt = defaultdict(Counter)
for ss, aa in zip(S_obs, A_obs):
    for t in range(len(aa)):
        s = k3(ss[t]); a = int(aa[t]); sp = k3(ss[t+1])
        sa_cnt[s][a] += 1; pobs_cnt[(s,a)][sp] += 1
def Pb(s, a, lap=1.0):
    c = sa_cnt.get(s)
    if c is None: return 1.0 / N_ACTIONS
    return (c.get(a, 0) + lap) / (sum(c.values()) + N_ACTIONS * lap)
def Pb_dist(s):
    c = sa_cnt.get(s)
    if c is None: return np.ones(N_ACTIONS) / N_ACTIONS
    p = np.array([c.get(a, 0) + 1.0 for a in range(N_ACTIONS)])
    return p / p.sum()
def sample_Pobs(s, a, rng):
    tbl = pobs_cnt.get((s, a))
    if not tbl: return None
    sps = list(tbl.keys()); ps = np.array([tbl[sp] for sp in sps], dtype=float); ps /= ps.sum()
    return sps[int(rng.choice(len(sps), p=ps))]

# ============ Generate worst-case data with corrected lava ============
print("[2/6] generating worst-case data (1000 episodes)...")
def worst_case_step(state, a, rng):
    pos = (state[0], state[1]); d = state[2]
    pb = Pb(state, a)
    if rng.random() < pb:
        sp = sample_Pobs(state, a, rng)
        if sp is not None:
            term = (sp[0], sp[1]) in LAVA or (sp[0], sp[1]) == GOAL
            return sp, term, "obs"
        w = int(rng.choice(5, p=WIND))
        np_, nd_, t = astep(pos, d, a, w); return (np_[0], np_[1], nd_), t, "obs_fb"
    nb = neighbors(pos, d, a)
    vals = [V_lower((p[0], p[1])) for p, _, _ in nb]
    idx = int(np.argmin(vals)); p, nd, t = nb[idx]
    return (p[0], p[1], nd), t, "adv"

def rollout_data(seed, max_steps=400):
    rng = np.random.default_rng(seed)
    state = (START[0], START[1], 0); traj_s = [list(state)]; traj_a = []
    for _ in range(max_steps):
        dist = Pb_dist(state); a = int(rng.choice(N_ACTIONS, p=dist))
        new_state, term, _ = worst_case_step(state, a, rng)
        traj_a.append(a); traj_s.append(list(new_state)); state = new_state
        if term: break
    last = state
    outcome = "goal" if (last[0], last[1]) == GOAL else ("lava" if (last[0], last[1]) in LAVA else "timeout")
    route = "far" if any(s[1] >= 12 for s in traj_s) else "near"
    return np.array(traj_s, dtype=np.float32), np.array(traj_a, dtype=np.int64), outcome, route

S_wc = []; A_wc = []; outs = []; routes = []
for ep in range(1000):
    s, a, o, r = rollout_data(seed=ep)
    S_wc.append(s); A_wc.append(a); outs.append(o); routes.append(r)
oc = Counter(outs); rc = Counter(routes)
ro = {"near": Counter(), "far": Counter()}
for o, r in zip(outs, routes):
    if r in ro: ro[r][o] += 1
print("  worst-case data: outcomes=" + str(dict(oc)) + "  routes=" + str(dict(rc)))
print("  near outcomes: " + str(dict(ro["near"])) + "   far outcomes: " + str(dict(ro["far"])))

out_path = "data/corridor_worst_case_n1000_fixed.npz"
np.savez(out_path,
    episodes_states=np.array(S_wc, dtype=object),
    episodes_actions=np.array(A_wc, dtype=object),
    episodes_outcomes=np.array(outs, dtype=object),
    episodes_routes=np.array(routes, dtype=object),
    goal_state=np.array([13,1,0], dtype=np.float32),
    state_dim=np.array(3), n_actions=np.array(N_ACTIONS),
    n_episodes=np.array(1000), p_near=np.array(0.8, dtype=np.float32))
print("  saved -> " + out_path)

# ============ Tabular VI ============
print("[3/6] tabular pessimistic VI ...")
states = [(x, y, dr) for (x, y) in WALK for dr in range(4)]
V = {s: 0.0 for s in states}
def Vget(cell, dr):
    if cell == GOAL: return 1.0
    if cell in LAVA: return 0.0
    return V.get((cell[0], cell[1], dr), 0.0)
def qval(s, a):
    (x, y, dr) = s; pos = (x, y); pb = Pb(s, a)
    tbl = pobs_cnt.get((s, a))
    if tbl:
        tot = sum(tbl.values())
        ov = sum(c / tot * Vget((sp[0], sp[1]), sp[2]) for sp, c in tbl.items())
    else:
        ov = sum(WIND[w] * (lambda r: Vget((r[0][0], r[0][1]), r[1]))(astep(pos, dr, a, w)) for w in range(5))
    nb = neighbors(pos, dr, a)
    worst = min(Vget((p[0], p[1]), dd) for (p, dd, _t) in nb)
    return GAMMA * (pb * ov + (1 - pb) * worst)

it_final = 0
for it in range(4000):
    Vn = {}
    for s in states:
        (x, y, dr) = s
        if (x, y) == GOAL: Vn[s] = 1.0; continue
        if (x, y) in LAVA: Vn[s] = 0.0; continue
        Vn[s] = sum(Pb(s, a) * qval(s, a) for a in VALID)
    diff = max(abs(Vn[s] - V[s]) for s in states); V = Vn; it_final = it
    if diff < 1e-9: break
print("  VI converged in " + str(it_final) + " iters")
print("  d_lower(goal|start):  forward=" + str(round(qval((1,1,0),2),4)) + "   right=" + str(round(qval((1,1,0),1),4)))
def tab_greedy(s): return max(VALID, key=lambda a: qval(s, a))

# ============ Rollouts ============
def true_rollout(policy_fn, force=None, seed=0, n=2000):
    rng = np.random.default_rng(seed); out = Counter(); rt = Counter()
    for _ in range(n):
        pos, d = START, 0; far = False; o = "timeout"
        for _ in range(200):
            a = policy_fn((pos[0], pos[1], d))
            w = force if force is not None else int(rng.choice(5, p=WIND))
            pos, d, term = astep(pos, d, a, w)
            if pos[1] >= 12: far = True
            if term:
                o = "goal" if pos == GOAL else ("lava" if pos in LAVA else "x")
                break
        out[o] += 1; rt["far" if far else "near"] += 1
    return out, rt

print("[4/6] rollouts: TABULAR oracle (corrected)")
for tag, fw in [("natural", None), ("worst-south", 1)]:
    o, r = true_rollout(tab_greedy, force=fw); n = sum(o.values())
    line = "  tabular [" + tag + "]: "
    for k, v in sorted(o.items()): line += k + ":" + str(int(round(v / n * 100))) + "%  "
    line += "| routes "
    for k, v in sorted(r.items()): line += k + ":" + str(int(round(v / n * 100))) + "%  "
    print(line)

# ============ Strong critic ============
print("[5/6] training strong critic (one-hot + uniform-neg + balanced + tau=0.1, 30k steps)...")
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

critic = ContrastiveCriticNumpy(state_dim=Nsa, n_actions=N_ACTIONS, hidden=128, emb_dim=64,
                                tau=TAU, seed=0, goal_feat_dim=Ng)
K = 48; LR = 1e-3
for step in range(30000):
    sb = []; ab = []; sfb = []
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

def scores_at(x, y, dd):
    hg = _embed_forward(g1cell(GOAL)[None], critic.g_W1, critic.g_b1, critic.g_W2, critic.g_b2)[0]
    out = {}
    for a in VALID:
        hsa, _ = critic._embed_sa(sa1(np.array([x, y, dd]))[None], np.array([a]))
        out[a] = float((hsa * hg).sum() / TAU)
    return out

for st in [(1,1,0), (1,1,1), (1,2,1)]:
    s = scores_at(*st); best = max(VALID, key=lambda a: s[a])
    print("  f" + str(st) + ": " + "  ".join("a" + str(a) + "=" + str(round(s[a], 2)) for a in VALID) + "  greedy=" + str(best))

# ============ AWR actor ============
print("[6/6] training AWR actor (beta=0.5) + rollouts ...")
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

def train_awr(beta=0.5, steps=15000, lr=2e-3, seed=1):
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
        Q = sc[np.arange(64), a]; Vb = sc.mean(1); adv = Q - Vb
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

def actor_rollout(act, force=None, seed=0, n=2000):
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

act = train_awr()
for tag, fw in [("natural", None), ("worst-south", 1)]:
    o, r = actor_rollout(act, force=fw); n = sum(o.values())
    line = "  AWR-actor [" + tag + "]: "
    for k, v in sorted(o.items()): line += k + ":" + str(int(round(v / n * 100))) + "%  "
    line += "| routes "
    for k, v in sorted(r.items()): line += k + ":" + str(int(round(v / n * 100))) + "%  "
    print(line)

print("DONE")
