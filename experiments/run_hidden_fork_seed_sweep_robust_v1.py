"""
Run robust-v1 experiments over seeds / regimes / weights.

Bare defaults are intended to match the current fair-comparison setting and
save checkpoints by default.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.training_defaults import (
    ROBUST_V1_COLLECTOR_MODE,
    ROBUST_V1_LOGIT_SCALE,
    ROBUST_V1_ORACLE_EPSILON,
    ROBUST_V1_M,
    ROBUST_V1_NUM_EPISODES,
    ROBUST_V1_P,
    ROBUST_V1_SWEEP_WEIGHTS,
    ROBUST_V1_W,
    TRAIN_NUMPY_LR,
    TRAIN_NUM_STEPS,
)
from experiments.train_contrastive_robust_v1 import train_numpy_robust_v1

ENV_CLEAN = "CausalContrastive-HiddenFork-15x15-Clean-v0"
ENV_CONF = "CausalContrastive-HiddenFork-15x15-v0"
SEEDS = (0, 1, 2, 3, 4)

CSV_FIELDS = [
    "seed",
    "env_id",
    "regime",
    "trajectory_length",
    "num_episodes",
    "valid_anchor_count",
    "contrastive_k",
    "P",
    "M",
    "w",
    "logit_scale",
    "collector_mode",
    "oracle_epsilon",
    "num_train_steps",
    "backend",
    "last_loss",
    "last_mean_pos_logit",
    "last_mean_neg_logit",
    "last_mean_pos_minus_neg_logit",
    "last_frac_pos_logit_gt_neg_logit",
    "last_mean_pos_obs_minus_surr_logit",
    "last_mean_neg_surr_minus_obs_logit",
    "mean_loss",
    "mean_mean_pos_logit",
    "mean_mean_neg_logit",
    "mean_mean_pos_minus_neg_logit",
    "mean_frac_pos_logit_gt_neg_logit",
    "mean_mean_pos_obs_minus_surr_logit",
    "mean_mean_neg_surr_minus_obs_logit",
]


def _regime_label(env_id: str) -> str:
    if "Clean" in env_id:
        return "clean"
    return "confounded"


def _row_from_result(r: dict) -> dict[str, object]:
    return {
        "seed": r["seed"],
        "env_id": r["env_id"],
        "regime": _regime_label(r["env_id"]),
        "trajectory_length": r["trajectory_length"],
        "num_episodes": r["num_episodes"],
        "valid_anchor_count": r["valid_anchor_count"],
        "contrastive_k": r["contrastive_k"],
        "P": r["P"],
        "M": r["M"],
        "w": r["w"],
        "logit_scale": r["logit_scale"],
        "collector_mode": r["collector_mode"],
        "oracle_epsilon": r["oracle_epsilon"],
        "num_train_steps": r["num_train_steps"],
        "backend": r["backend"],
        "last_loss": r["last_loss"],
        "last_mean_pos_logit": r["last_mean_pos_logit"],
        "last_mean_neg_logit": r["last_mean_neg_logit"],
        "last_mean_pos_minus_neg_logit": r["last_mean_pos_minus_neg_logit"],
        "last_frac_pos_logit_gt_neg_logit": r["last_frac_pos_logit_gt_neg_logit"],
        "last_mean_pos_obs_minus_surr_logit": r["last_mean_pos_obs_minus_surr_logit"],
        "last_mean_neg_surr_minus_obs_logit": r["last_mean_neg_surr_minus_obs_logit"],
        "mean_loss": r["mean_loss"],
        "mean_mean_pos_logit": r["mean_mean_pos_logit"],
        "mean_mean_neg_logit": r["mean_mean_neg_logit"],
        "mean_mean_pos_minus_neg_logit": r["mean_mean_pos_minus_neg_logit"],
        "mean_frac_pos_logit_gt_neg_logit": r["mean_frac_pos_logit_gt_neg_logit"],
        "mean_mean_pos_obs_minus_surr_logit": r["mean_mean_pos_obs_minus_surr_logit"],
        "mean_mean_neg_surr_minus_obs_logit": r["mean_mean_neg_surr_minus_obs_logit"],
    }


def run_sweep(
    *,
    num_episodes: int,
    num_steps: int,
    lr: float,
    P: int,
    M: int,
    weights: list[float],
    logit_scale: float,
    collector_mode: str,
    oracle_epsilon: float,
    save_checkpoint: bool,
) -> list[dict[str, object]]:
    print("[Sweep] backend=numpy robust_v1", file=sys.stderr)
    rows: list[dict[str, object]] = []
    for w in weights:
        for seed in SEEDS:
            for env_id in (ENV_CLEAN, ENV_CONF):
                print(
                    f"[Sweep] w={w} seed={seed} env={env_id!r} ...",
                    file=sys.stderr,
                    flush=True,
                )
                result = train_numpy_robust_v1(
                    seed=seed,
                    env_id=env_id,
                    num_episodes=num_episodes,
                    num_steps=num_steps,
                    lr=lr,
                    P=P,
                    M=M,
                    w=w,
                    logit_scale=logit_scale,
                    collector_mode=collector_mode,
                    oracle_epsilon=oracle_epsilon,
                    save_checkpoint=save_checkpoint,
                    verbose=False,
                )
                rows.append(_row_from_result(result))
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(
        description="HiddenFork robust-v1 seed × regime × weight sweep → CSV"
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_seed_sweep_robust_v1.csv",
        help="CSV path (default: results/hidden_fork_seed_sweep_robust_v1.csv)",
    )
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
    p.add_argument(
        "--oracle-epsilon",
        type=float,
        default=ROBUST_V1_ORACLE_EPSILON,
    )
    p.add_argument("--w", type=float, default=None)
    p.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Optional sweep over multiple shared weights, e.g. --weights 0.5 0.7 0.9",
    )
    p.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print CSV to stdout only; do not write a file",
    )
    p.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Skip writing per-run checkpoints",
    )
    args = p.parse_args()

    if args.weights is not None:
        weights = list(args.weights)
    elif args.w is not None:
        weights = [args.w]
    else:
        weights = list(ROBUST_V1_SWEEP_WEIGHTS)
    rows = run_sweep(
        num_steps=args.num_steps,
        num_episodes=args.num_episodes,
        lr=args.lr,
        P=args.P,
        M=args.M,
        weights=weights,
        logit_scale=args.logit_scale,
        collector_mode=args.collector_mode,
        oracle_epsilon=args.oracle_epsilon,
        save_checkpoint=not args.no_save_checkpoint,
    )

    if not args.stdout_only:
        _write_csv(rows, args.output)
        print(f"[Sweep] wrote {args.output}", file=sys.stderr)

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()

