"""
Action-score heatmap diagnostics for HiddenFork robust-v1 models.

For a paired fixed_u=0 / fixed_u=1 rollout with the same seed, pick the last
anchor state whose observation is still identical across U. Then compute,
for every action, the robust critic score toward:
  - the U=0 future at t+k
  - the U=1 future at t+k

This helps visualize whether different weights make the model more or less
confident about branch-specific futures from the same local observation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs  # noqa: F401

from configs.training_defaults import (
    ROBUST_V1_COLLECTOR_MODE,
    ROBUST_V1_LOGIT_SCALE,
    ROBUST_V1_M,
    ROBUST_V1_NUM_EPISODES,
    ROBUST_V1_ORACLE_EPSILON,
    ROBUST_V1_P,
    ROBUST_V1_SWEEP_WEIGHTS,
    TRAIN_NUMPY_LR,
    TRAIN_NUM_STEPS,
)
from experiments.train_contrastive_robust_v1 import train_numpy_robust_v1
from utils.offline_data import build_policy
from utils.collector import rollout_episode
from utils.contrastive_sampling import DEFAULT_K
from utils.preprocess import extract_state

ENV_CONF = "CausalContrastive-HiddenFork-15x15-v0"
ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

CSV_FIELDS = [
    "weight",
    "seed",
    "anchor_step",
    "contrastive_k",
    "action_id",
    "action_name",
    "score_to_u0_future",
    "score_to_u1_future",
    "score_margin_u0_minus_u1",
]


def _paired_rollouts(
    *,
    seed: int,
    env_id: str,
    collector_mode: str,
    oracle_epsilon: float,
) -> tuple[list[dict], list[dict]]:
    pair = []
    for fixed_u in (0, 1):
        env = gym.make(env_id, render_mode="rgb_array", fixed_u=fixed_u)
        env.action_space.seed(seed)
        policy = build_policy(
            env,
            collector_mode=collector_mode,
            oracle_epsilon=oracle_epsilon,
            seed=seed,
        )
        traj = rollout_episode(env, policy, max_steps=500, seed=seed)
        pair.append(traj)
    return pair[0], pair[1]


def _select_anchor_and_futures(
    traj_u0: list[dict],
    traj_u1: list[dict],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    T = min(len(traj_u0), len(traj_u1))
    last_equal_t: int | None = None
    for t in range(T - k):
        o0 = traj_u0[t]["obs"]
        o1 = traj_u1[t]["obs"]
        if np.array_equal(o0["image"], o1["image"]):
            last_equal_t = t
        else:
            break

    if last_equal_t is None or last_equal_t + k >= T:
        raise RuntimeError("Could not find a valid equal-observation anchor with t+k in range.")

    s_anchor = extract_state(traj_u0[last_equal_t]["obs"]).astype(np.float64)
    s_u0 = extract_state(traj_u0[last_equal_t + k]["obs"]).astype(np.float64)
    s_u1 = extract_state(traj_u1[last_equal_t + k]["obs"]).astype(np.float64)
    return s_anchor, s_u0, s_u1, last_equal_t


def _score_triplet(model, s: np.ndarray, a: int, s_future: np.ndarray, logit_scale: float) -> float:
    s_batch = s[None, :]
    a_batch = np.array([a], dtype=np.int64)
    s_future_batch = s_future[None, :]
    h, _ = model._embed(s_batch, a_batch)
    hf, _ = model._embed(s_future_batch, a_batch)
    return float(np.sum(h * hf, axis=1)[0] * logit_scale)


def run_heatmap(
    *,
    env_id: str,
    weights: list[float],
    collector_mode: str,
    oracle_epsilon: float,
    num_train_episodes: int,
    num_train_steps: int,
    lr: float,
    P: int,
    M: int,
    logit_scale: float,
    seed: int,
) -> list[dict[str, object]]:
    traj_u0, traj_u1 = _paired_rollouts(
        seed=seed,
        env_id=env_id,
        collector_mode=collector_mode,
        oracle_epsilon=oracle_epsilon,
    )
    s_anchor, s_u0, s_u1, anchor_t = _select_anchor_and_futures(traj_u0, traj_u1, DEFAULT_K)

    rows: list[dict[str, object]] = []
    for w in weights:
        result = train_numpy_robust_v1(
            seed=seed,
            env_id=env_id,
            num_episodes=num_train_episodes,
            num_steps=num_train_steps,
            lr=lr,
            P=P,
            M=M,
            w=w,
            collector_mode=collector_mode,
            oracle_epsilon=oracle_epsilon,
            logit_scale=logit_scale,
            return_model=True,
            verbose=False,
        )
        model = result["model"]
        for action_id in range(model.n_actions):
            s0 = _score_triplet(model, s_anchor, action_id, s_u0, logit_scale)
            s1 = _score_triplet(model, s_anchor, action_id, s_u1, logit_scale)
            rows.append(
                {
                    "weight": w,
                    "seed": seed,
                    "anchor_step": anchor_t,
                    "contrastive_k": DEFAULT_K,
                    "action_id": action_id,
                    "action_name": ACTION_NAMES.get(action_id, f"action_{action_id}"),
                    "score_to_u0_future": s0,
                    "score_to_u1_future": s1,
                    "score_margin_u0_minus_u1": s0 - s1,
                }
            )
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _maybe_write_png(rows: list[dict[str, object]], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    weights = sorted({float(r["weight"]) for r in rows})
    action_names = [ACTION_NAMES.get(i, f"action_{i}") for i in sorted({int(r["action_id"]) for r in rows})]
    fig, axes = plt.subplots(len(weights), 2, figsize=(10, 3 * len(weights)), squeeze=False)
    for i, w in enumerate(weights):
        sub = [r for r in rows if float(r["weight"]) == w]
        sub = sorted(sub, key=lambda r: int(r["action_id"]))
        mat = np.array(
            [[float(r["score_to_u0_future"]) for r in sub],
             [float(r["score_to_u1_future"]) for r in sub]],
            dtype=np.float64,
        )
        im = axes[i, 0].imshow(mat, aspect="auto", cmap="viridis")
        axes[i, 0].set_title(f"w={w} scores")
        axes[i, 0].set_yticks([0, 1], labels=["future U0", "future U1"])
        axes[i, 0].set_xticks(range(len(action_names)), labels=action_names, rotation=45, ha="right")
        fig.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)

        mat2 = np.array([[float(r["score_margin_u0_minus_u1"]) for r in sub]], dtype=np.float64)
        im2 = axes[i, 1].imshow(mat2, aspect="auto", cmap="coolwarm")
        axes[i, 1].set_title(f"w={w} margin (U0-U1)")
        axes[i, 1].set_yticks([0], labels=["margin"])
        axes[i, 1].set_xticks(range(len(action_names)), labels=action_names, rotation=45, ha="right")
        fig.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Action heatmap diagnostics for robust-v1 HiddenFork models")
    p.add_argument("-o", "--output", type=Path, default=ROOT / "results" / "hidden_fork_action_heatmap_robust_v1.csv")
    p.add_argument("--png-output", type=Path, default=ROOT / "results" / "hidden_fork_action_heatmap_robust_v1.png")
    p.add_argument("--env-id", type=str, default=ENV_CONF)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-episodes", type=int, default=ROBUST_V1_NUM_EPISODES)
    p.add_argument("--num-steps", type=int, default=TRAIN_NUM_STEPS)
    p.add_argument("--lr", type=float, default=TRAIN_NUMPY_LR)
    p.add_argument("--P", type=int, default=ROBUST_V1_P)
    p.add_argument("--M", type=int, default=ROBUST_V1_M)
    p.add_argument("--logit-scale", type=float, default=ROBUST_V1_LOGIT_SCALE)
    p.add_argument(
        "--collector-mode",
        type=str,
        default=ROBUST_V1_COLLECTOR_MODE,
        choices=["random", "oracle_eps"],
    )
    p.add_argument("--oracle-epsilon", type=float, default=ROBUST_V1_ORACLE_EPSILON)
    p.add_argument("--w", type=float, default=None)
    p.add_argument("--weights", type=float, nargs="+")
    p.add_argument("--stdout-only", action="store_true")
    args = p.parse_args()

    if args.weights is not None:
        weights = list(args.weights)
    elif args.w is not None:
        weights = [args.w]
    else:
        weights = list(ROBUST_V1_SWEEP_WEIGHTS)

    rows = run_heatmap(
        env_id=args.env_id,
        weights=weights,
        collector_mode=args.collector_mode,
        oracle_epsilon=args.oracle_epsilon,
        num_train_episodes=args.num_episodes,
        num_train_steps=args.num_steps,
        lr=args.lr,
        P=args.P,
        M=args.M,
        logit_scale=args.logit_scale,
        seed=args.seed,
    )

    if not args.stdout_only:
        _write_csv(rows, args.output)
        _maybe_write_png(rows, args.png_output)

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()

