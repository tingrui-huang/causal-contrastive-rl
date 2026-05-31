r"""
Train a one-hot (tabular-feature) contrastive critic on WindyCorridor worst-case
data — the configuration that actually makes goal-reachability learnable here.

Why this script exists
----------------------
The vanilla critic (raw [x, y, dir] inputs, in-batch negatives) collapses on this
testbed: margin ~0.16, and it ranks the lethal NEAR route ABOVE the safe FAR route
because it learns coordinate proximity, not reachability. Three things fix it:

  1. one-hot state features      — kill the raw-(x,y) "close to goal = high" bias
  2. direction-agnostic goal     — goal is the CELL (x,y), not cell+facing, so
                                   "reach (13,1) facing east" (NEAR's arrival) is
                                   not a different goal from "facing north" (FAR)
  3. false-negative masking      — built into ContrastiveCriticNumpy.loss_and_grads;
                                   the goal cell is ~70% of all futures, so a batch
                                   of in-batch negatives is mostly the goal itself,
                                   contradicting the positives unless masked.

Encoding split: the (s,a) encoder sees a one-hot over (x, y, dir) (facing matters
for what an action does); the goal encoder sees a one-hot over (x, y) only.

Run::
    python experiments/train_corridor_onehot.py \
        --data data/corridor_worst_case_n1000.npz \
        --num-steps 30000 \
        --output checkpoints/corridor_onehot_seed0.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from configs import corridor_defaults as C

GAMMA = C.GAMMA
GOAL_CELL = C.GOAL_CELL
VALID_ACTIONS = C.VALID_ACTIONS  # left, right, forward, done


def _k3(s) -> tuple[int, int, int]:
    return (int(round(s[0])), int(round(s[1])), int(round(s[2])))


def _k2(s) -> tuple[int, int]:
    return (int(round(s[0])), int(round(s[1])))


def build_indices(episode_states):
    """Map (x,y,dir) -> sa index and (x,y) -> goal index, from observed states."""
    sa_index: dict[tuple[int, int, int], int] = {}
    goal_index: dict[tuple[int, int], int] = {}
    for tr in episode_states:
        for s in tr:
            sa_index.setdefault(_k3(s), len(sa_index))
            goal_index.setdefault(_k2(s), len(goal_index))
    return sa_index, goal_index


def encode_sa(states_raw: np.ndarray, sa_index: dict) -> np.ndarray:
    out = np.zeros((len(states_raw), len(sa_index)), dtype=np.float64)
    for i, s in enumerate(states_raw):
        k = _k3(s)
        if k in sa_index:
            out[i, sa_index[k]] = 1.0
    return out


def encode_goal(states_raw: np.ndarray, goal_index: dict) -> np.ndarray:
    out = np.zeros((len(states_raw), len(goal_index)), dtype=np.float64)
    for i, s in enumerate(states_raw):
        k = _k2(s)
        if k in goal_index:
            out[i, goal_index[k]] = 1.0
    return out


def _geom_offset(rng: np.random.Generator, max_val: int) -> int:
    off = int(rng.geometric(1.0 - GAMMA))
    return min(max(off, 1), max_val)


def empirical_d_lower(episode_states) -> dict[tuple[int, int], float]:
    """Monte-Carlo discounted goal occupancy per cell from the (worst-case) data.

    This is a direct estimate of d_lower(goal | cell) — the target the critic
    should track. Used only for the verification correlation.
    """
    occ: dict[tuple[int, int], list[float]] = {}
    for tr in episode_states:
        cells = [_k2(s) for s in tr]
        T = len(cells)
        for t in range(T):
            val = 0.0
            for k in range(t, T):
                if cells[k] == GOAL_CELL:
                    val += GAMMA ** (k - t)
            occ.setdefault(cells[t], []).append(val)
    return {c: float(np.mean(v)) for c, v in occ.items()}


def train(
    *,
    data_path: Path,
    output_path: Path,
    seed: int = 0,
    num_steps: int = 30000,
    batch_size: int = 64,
    critic_lr: float = 2e-3,
    hidden: int = 128,
    emb_dim: int = 64,
    tau: float = 0.07,
    log_interval: int = 2000,
    verbose: bool = True,
) -> dict:
    data = np.load(data_path, allow_pickle=True)
    episode_states = [np.asarray(s, dtype=np.float64) for s in data["episodes_states"]]
    episode_actions = [np.asarray(a, dtype=np.int64) for a in data["episodes_actions"]]
    n_actions = int(data["n_actions"])

    sa_index, goal_index = build_indices(episode_states)
    n_sa, n_goal = len(sa_index), len(goal_index)
    anchors = [(ei, t) for ei, tr in enumerate(episode_states) for t in range(len(tr) - 1)]
    if verbose:
        print(f"[onehot] sa_features={n_sa}  goal_features={n_goal}  anchors={len(anchors)}")
        print(f"[onehot] steps={num_steps}  tau={tau}  gamma={GAMMA}")

    rng = np.random.default_rng(seed)
    critic = ContrastiveCriticNumpy(
        state_dim=n_sa,
        n_actions=n_actions,
        hidden=hidden,
        emb_dim=emb_dim,
        tau=tau,
        seed=seed,
        goal_feat_dim=n_goal,
    )

    margins: list[float] = []
    c_losses: list[float] = []
    t0 = time.time()
    for step in range(1, num_steps + 1):
        ids = rng.integers(0, len(anchors), size=batch_size)
        pairs = [anchors[int(i)] for i in ids]
        s_raw = np.stack([episode_states[ep][t] for ep, t in pairs])
        a = np.array([episode_actions[ep][t] for ep, t in pairs], dtype=np.int64)
        sf_raw = []
        for ep, t in pairs:
            off = _geom_offset(rng, len(episode_states[ep]) - t - 1)
            sf_raw.append(episode_states[ep][t + off])
        sf_raw = np.stack(sf_raw)

        s_enc = encode_sa(s_raw, sa_index)
        sf_enc = encode_goal(sf_raw, goal_index)

        loss, grads = critic.loss_and_grads(s_enc, a, sf_enc)
        critic.apply_sgd(grads, lr=critic_lr)
        c_losses.append(float(loss))
        margins.append(critic.margin(s_enc, a, sf_enc))

        if verbose and (step % log_interval == 0 or step == 1):
            print(f"  step {step:6d}  c_loss={np.mean(c_losses[-log_interval:]):.4f}  "
                  f"margin={np.mean(margins[-log_interval:]):+.3f}")

    if verbose:
        print(f"[onehot] done in {time.time() - t0:.1f}s  final margin={np.mean(margins[-500:]):+.3f}")

    # ---- save ----
    sd = critic.state_dict()
    config = {
        "seed": seed, "data_path": str(data_path), "num_steps": num_steps,
        "batch_size": batch_size, "critic_lr": critic_lr, "gamma": GAMMA,
        "hidden": hidden, "emb_dim": emb_dim, "tau": tau,
        "state_dim": n_sa, "n_actions": n_actions, "goal_feat_dim": n_goal,
        "encoding": "onehot",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        sa_W1=sd["sa_W1"], sa_b1=sd["sa_b1"], sa_W2=sd["sa_W2"], sa_b2=sd["sa_b2"],
        g_W1=sd["g_W1"], g_b1=sd["g_b1"], g_W2=sd["g_W2"], g_b2=sd["g_b2"],
        c_losses=np.array(c_losses, dtype=np.float32),
        margins=np.array(margins, dtype=np.float32),
        sa_index_json=np.array(json.dumps({f"{k[0]},{k[1]},{k[2]}": v for k, v in sa_index.items()})),
        goal_index_json=np.array(json.dumps({f"{k[0]},{k[1]}": v for k, v in goal_index.items()})),
        config_json=np.array(json.dumps(config)),
    )
    if verbose:
        print(f"[onehot] checkpoint -> {output_path}")

    return {
        "critic": critic, "sa_index": sa_index, "goal_index": goal_index,
        "episode_states": episode_states, "final_margin": float(np.mean(margins[-500:])),
    }


def verify(result: dict) -> None:
    """Check: fork ranking, route means, and correlation with empirical d_lower."""
    critic = result["critic"]
    sa_index, goal_index = result["sa_index"], result["goal_index"]
    goal_enc = encode_goal(np.array([[GOAL_CELL[0], GOAL_CELL[1], 0]], dtype=np.float64), goal_index)

    def f_val(x, y, d):
        s_enc = encode_sa(np.array([[x, y, d]], dtype=np.float64), sa_index)
        scores = critic.score_actions(s_enc, goal_enc)[0]
        return max(scores[a] for a in VALID_ACTIONS)

    print("\n" + "=" * 64)
    print("VERIFY — does the learned critic match the tabular pessimistic ranking?")
    print("=" * 64)
    near = f_val(2, 1, 0)
    far = f_val(1, 2, 1)
    print(f"FORK:  f(near-next 2,1)={near:+.3f}   f(far-next 1,2)={far:+.3f}   "
          f"-> {'FAR (safe) ***' if far > near else 'NEAR (dies)'}")

    near_mean = np.mean([f_val(x, 1, 0) for x in range(1, 14)])
    far_cells = [f_val(1, y, 1) for y in range(2, 13)] + [f_val(x, 13, 0) for x in range(1, 14)] + \
                [f_val(13, y, 3) for y in range(12, 0, -1)]
    print(f"route means:  NEAR={near_mean:+.3f}   FAR={np.mean(far_cells):+.3f}  "
          "(note: late-NEAR cells are genuinely safe+close, so the *fork* is the decision)")

    # correlation between learned f(cell) and empirical d_lower(goal|cell)
    dlo = empirical_d_lower(result["episode_states"])
    cells, fs, ds = [], [], []
    for (x, y), dval in dlo.items():
        if (x, y) == GOAL_CELL:
            continue
        # best facing per cell
        best = max(f_val(x, y, d) for d in range(4))
        cells.append((x, y)); fs.append(best); ds.append(dval)
    fs, ds = np.array(fs), np.array(ds)
    if np.std(fs) > 1e-9 and np.std(ds) > 1e-9:
        corr = float(np.corrcoef(fs, ds)[0, 1])
        print(f"corr( learned f , empirical d_lower ) over {len(fs)} cells = {corr:+.3f}  "
              f"(was NEGATIVE for the broken critic; want strongly positive)")


def main() -> None:
    p = argparse.ArgumentParser(description="Train one-hot contrastive critic on corridor worst-case data.")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "corridor_worst_case_n1000.npz")
    p.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "corridor_onehot_seed0.npz")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--num-steps", type=int, default=C.CRITIC_STEPS)
    p.add_argument("--tau", type=float, default=C.TAU)
    p.add_argument("--critic-lr", type=float, default=C.CRITIC_LR)
    args = p.parse_args()

    result = train(
        data_path=args.data, output_path=args.output, seed=args.seed,
        num_steps=args.num_steps, tau=args.tau, critic_lr=args.critic_lr,
    )
    verify(result)


if __name__ == "__main__":
    main()
