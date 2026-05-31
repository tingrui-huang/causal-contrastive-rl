r"""
Diagnose WHERE pessimism failed to enter the contrastive critic.

Question being answered
-----------------------
We trained the contrastive critic on *worst-case-kernel* data (NEAR route dies
74%, FAR route 100% safe), hoping the critic would learn "NEAR is dangerous ->
low value, FAR is safe -> high value". The trained policy instead always walks
NEAR and dies. This script localizes the failure across four checks:

  [1] Training health        -- did the contrastive critic even fit?
                               (final margin, critic loss, BC accuracy)
  [2] Critic value profile   -- critic's reachability score along the canonical
                               NEAR vs FAR routes. Flat / NEAR-not-penalized
                               means pessimism is absent from the critic.
  [3] Data ground truth      -- empirical P(reach goal | cell visited) in the
                               worst-case dataset. This is the signal a perfect
                               goal-reachability critic SHOULD encode.
  [4] Critic vs truth        -- does the critic's value track the data truth, and
                               does critic-greedy actually pick FAR?

Run::

    python experiments/diagnose_corridor_critic.py \
        --checkpoint checkpoints/corridor_causal_seed0.npz \
        --data data/corridor_worst_case_n1000.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows consoles default to GBK; force UTF-8 so non-ASCII output never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from minigrid.core.actions import Actions

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from envs.windy_corridor import GOAL_POS, LETHAL_X
from utils.v_lower_oracle import bfs_distance

DIR_EAST, DIR_SOUTH, DIR_WEST, DIR_NORTH = 0, 1, 2, 3
VALID_ACTIONS = [int(Actions.left), int(Actions.right), int(Actions.forward), int(Actions.done)]
ACTION_NAMES = {0: "left", 1: "right", 2: "fwd", 6: "done"}


# --------------------------------------------------------------------------- #
# Canonical route definitions (cell, advancing direction at that cell)        #
# --------------------------------------------------------------------------- #
def near_route() -> list[tuple[int, int, int]]:
    """(x, 1, EAST) for x in 1..13 -- the short, sometimes-lethal route."""
    return [(x, 1, DIR_EAST) for x in range(1, 14)]


def far_route() -> list[tuple[int, int, int]]:
    """Down x=1, east along y=13, up x=13 -- the long, always-safe route."""
    cells = [(1, y, DIR_SOUTH) for y in range(1, 13)]
    cells += [(x, 13, DIR_EAST) for x in range(1, 13)]
    cells += [(13, y, DIR_NORTH) for y in range(13, 0, -1)]
    return cells


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def load_critic(path: Path) -> tuple[ContrastiveCriticNumpy, dict, np.ndarray]:
    ckpt = np.load(path, allow_pickle=True)
    cfg = json.loads(str(ckpt["config_json"]))
    gd = cfg.get("goal_dims")
    critic = ContrastiveCriticNumpy(
        state_dim=int(cfg["state_dim"]), n_actions=int(cfg["n_actions"]),
        hidden=int(cfg["hidden"]), emb_dim=int(cfg["emb_dim"]),
        tau=float(cfg["tau"]), seed=int(cfg["seed"]),
        goal_dims=tuple(gd) if gd is not None else None,
    )
    for k in ("sa_W1", "sa_b1", "sa_W2", "sa_b2", "g_W1", "g_b1", "g_W2", "g_b2"):
        getattr(critic, k)[...] = ckpt[k]
    curves = {k: np.asarray(ckpt[k]) for k in ("c_losses", "margins", "a_losses", "bc_accs") if k in ckpt}
    goal_state = np.array([GOAL_POS[0], GOAL_POS[1], DIR_EAST], dtype=np.float64)
    return critic, curves, goal_state


def critic_value(critic: ContrastiveCriticNumpy, state, goal_state: np.ndarray) -> tuple[float, int]:
    """max_a over valid actions of f(s, a, goal). Returns (value, best_action)."""
    s = np.asarray(state, dtype=np.float64)[None]
    scores = critic.score_actions(s, goal_state[None])[0]
    best_a = VALID_ACTIONS[int(np.argmax([scores[a] for a in VALID_ACTIONS]))]
    return float(scores[best_a]), int(best_a)


# --------------------------------------------------------------------------- #
# Data ground truth: P(reach goal | cell visited)                             #
# --------------------------------------------------------------------------- #
def empirical_goal_prob(data_path: Path) -> tuple[dict[tuple[int, int], float], dict]:
    data = np.load(data_path, allow_pickle=True)
    states = list(data["episodes_states"])
    outcomes = list(data["episodes_outcomes"])
    routes = list(data["episodes_routes"])

    visits: dict[tuple[int, int], int] = {}
    goals: dict[tuple[int, int], int] = {}
    for traj, outcome in zip(states, outcomes):
        reached = (outcome == "goal")
        seen = {(int(s[0]), int(s[1])) for s in traj}
        for cell in seen:
            visits[cell] = visits.get(cell, 0) + 1
            if reached:
                goals[cell] = goals.get(cell, 0) + 1
    p_goal = {c: goals.get(c, 0) / visits[c] for c in visits}

    near_eps = [o for r, o in zip(routes, outcomes) if r == "near"]
    far_eps = [o for r, o in zip(routes, outcomes) if r == "far"]
    summary = {
        "near_n": len(near_eps),
        "near_goal_rate": (sum(o == "goal" for o in near_eps) / len(near_eps)) if near_eps else float("nan"),
        "far_n": len(far_eps),
        "far_goal_rate": (sum(o == "goal" for o in far_eps) / len(far_eps)) if far_eps else float("nan"),
    }
    return p_goal, summary


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def report_training_health(curves: dict) -> dict:
    section("[1] TRAINING HEALTH -- did the contrastive critic fit at all?")
    out = {}
    if "margins" in curves and curves["margins"].size:
        m = curves["margins"]
        out["margin_final"] = float(np.mean(m[-500:]))
        print(f"  critic margin (pos_logit - neg_logit):  final={out['margin_final']:+.3f}  "
              f"max={float(m.max()):+.3f}")
        print("    -> a healthy contrastive critic has margin >> 1 (positives clearly")
        print("      above negatives). Near 0 means it cannot tell a real future from")
        print("      a random other-trajectory state -> reachability scores collapse flat.")
    if "c_losses" in curves and curves["c_losses"].size:
        c = curves["c_losses"]
        print(f"  critic BCE loss:                         final={float(np.mean(c[-500:])):.4f}  "
              f"(start={float(c[0]):.4f})")
    if "bc_accs" in curves and curves["bc_accs"].size:
        b = curves["bc_accs"]
        out["bc_acc_final"] = float(np.mean(b[-500:]))
        print(f"  actor BC accuracy (clones data action):  final={out['bc_acc_final']:.3f}")
        print("    -> high BC acc means the actor reproduces the dataset's actions,")
        print("      which are NEAR-dominated regardless of the critic.")
    return out


def report_route_profiles(critic, goal_state, p_goal) -> dict:
    section("[2]+[3] CRITIC VALUE vs DATA TRUTH along NEAR and FAR routes")
    rows = {"near": [], "far": []}
    for name, route in (("near", near_route()), ("far", far_route())):
        print(f"\n  --- {name.upper()} route (advancing toward goal) ---")
        print(f"  {'cell':>9} {'dist':>4} {'critic_V':>9} {'best_a':>6} {'data_P(goal)':>13} {'lethal':>6}")
        for (x, y, d) in route:
            v, ba = critic_value(critic, (x, y, d), goal_state)
            dist = bfs_distance((x, y))
            pg = p_goal.get((x, y), float("nan"))
            lethal = "YES" if (y == 1 and x in LETHAL_X) else ""
            rows[name].append({"cell": (x, y), "V": v, "p_goal": pg, "dist": dist})
            print(f"  {str((x, y)):>9} {('' if dist is None else dist):>4} {v:>9.3f} "
                  f"{ACTION_NAMES.get(ba, ba):>6} {pg:>13.2f} {lethal:>6}")
    return rows


def report_verdict(curves_out, rows, data_summary, critic, goal_state) -> None:
    section("[4] VERDICT -- where did pessimism fail to enter the critic?")

    near_V = np.array([r["V"] for r in rows["near"]])
    far_V = np.array([r["V"] for r in rows["far"]])

    # Pool cells where we have both critic value and data truth, measure tracking.
    allr = rows["near"] + rows["far"]
    V = np.array([r["V"] for r in allr])
    P = np.array([r["p_goal"] for r in allr])
    mask = ~np.isnan(P)
    if mask.sum() > 2 and np.std(V[mask]) > 1e-9 and np.std(P[mask]) > 1e-9:
        corr = float(np.corrcoef(V[mask], P[mask])[0, 1])
    else:
        corr = float("nan")

    # Lookahead at the fork: NEAR-next = (2,1), FAR-next = (1,2).
    v_near_next, _ = critic_value(critic, (2, 1, DIR_EAST), goal_state)
    v_far_next, _ = critic_value(critic, (1, 2, DIR_SOUTH), goal_state)

    print(f"  data truth:    NEAR reaches goal {data_summary['near_goal_rate']:.0%} "
          f"(n={data_summary['near_n']})   FAR reaches goal "
          f"{data_summary['far_goal_rate']:.0%} (n={data_summary['far_n']})")
    print(f"                 -> a pessimistic critic SHOULD rank FAR cells above NEAR.")
    print(f"  critic value:  mean(NEAR)={near_V.mean():+.3f}   mean(FAR)={far_V.mean():+.3f}   "
          f"spread(all)={V.max() - V.min():.3f}")
    print(f"  fork lookahead: V(2,1 NEAR-next)={v_near_next:+.3f}   "
          f"V(1,2 FAR-next)={v_far_next:+.3f}   "
          f"{'(NEAR higher -> walks into danger)' if v_near_next >= v_far_next else '(FAR higher -> would avoid)'}")
    print(f"  corr(critic_V, data_P_goal) over cells: {corr:+.3f}   "
          f"(~=0 means the critic ignored the danger signal that IS in the data)")

    print("\n  CONCLUSION:")
    margin = curves_out.get("margin_final", float("nan"))
    if margin == margin and margin < 1.0:
        print(f"   * Critic barely fit (margin {margin:+.2f} ~= 0): contrastive embeddings")
        print("     did not separate 'leads to goal' from 'leads anywhere else', so every")
        print("     (s,a) scores ~the same toward the goal -> no pessimism CAN be expressed.")
    if corr == corr and corr < -0.1:
        print(f"   * WORSE than flat -- ANTI-pessimism: critic value is NEGATIVELY")
        print(f"     correlated with the data's goal-reach probability (corr={corr:+.2f}).")
        print("     What it actually learned is proximity (BFS distance): the SHORT route")
        print("     scores highest, and the short route is exactly the deadly NEAR one.")
        print("     Reachability=='few steps away' rewards the dangerous shortcut.")
    elif not (corr == corr) or abs(corr) < 0.3:
        print("   * Critic value does NOT track the data's goal-reach probability: the")
        print("     'NEAR dies / FAR survives' signal present in the dataset never made")
        print("     it into the learned reachability scores.")
    print("   * Structural reason: contrastive reachability has NO negative example for")
    print("     death. A lava death just truncates the trajectory; the pre-death cells")
    print("     are still emitted as positive (s, s_future) pairs. Nothing tells the")
    print("     critic 'this action killed you' -- only rewards/values can.")


def maybe_plot(rows, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"\n[plot] skipped (matplotlib unavailable: {e})")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, color in (("near", "tab:red"), ("far", "tab:blue")):
        n = len(rows[name])
        frac = np.linspace(0, 1, n)
        V = [r["V"] for r in rows[name]]
        P = [r["p_goal"] for r in rows[name]]
        ax1.plot(frac, V, "-o", color=color, label=name.upper(), markersize=3)
        ax2.plot(frac, P, "-o", color=color, label=name.upper(), markersize=3)
    ax1.set_ylabel("critic value (max-action score -> goal)")
    ax1.set_title("Critic learned value along each route (flat / NEAR-not-penalized = no pessimism)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_ylabel("data P(reach goal | cell)")
    ax2.set_xlabel("fraction of route from start (0) to goal (1)")
    ax2.set_title("Ground truth in the worst-case data (what the critic SHOULD have learned)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\n[plot] saved -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "corridor_causal_seed0.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "corridor_worst_case_n1000.npz")
    parser.add_argument("--plot", type=Path, default=ROOT / "figures" / "corridor_critic_diagnosis.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"checkpoint: {args.checkpoint.name}")
    print(f"data:       {args.data.name}")

    critic, curves, goal_state = load_critic(args.checkpoint)
    p_goal, data_summary = empirical_goal_prob(args.data)

    curves_out = report_training_health(curves)
    rows = report_route_profiles(critic, goal_state, p_goal)
    report_verdict(curves_out, rows, data_summary, critic, goal_state)
    if not args.no_plot:
        maybe_plot(rows, args.plot)


if __name__ == "__main__":
    main()
