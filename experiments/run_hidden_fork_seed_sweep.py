"""
Run softmax-baseline experiments over seeds × regimes and write CSV.

This sweep is intended to be runnable with bare defaults for the current
fair-comparison setting:
- NumPy backend
- multi-episode dataset
- current collector defaults from ``configs/training_defaults.py``
- checkpoints saved by default
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
    TRAIN_COLLECTOR_MODE,
    TRAIN_NUM_EPISODES,
    TRAIN_NUM_STEPS,
    TRAIN_ORACLE_EPSILON,
)
from experiments.train_contrastive_baseline import train_numpy

ENV_CLEAN = "CausalContrastive-HiddenFork-15x15-Clean-v0"
ENV_CONF = "CausalContrastive-HiddenFork-15x15-v0"
SEEDS = (0, 1, 2, 3, 4)

# Flat CSV columns (no nested train_config)
CSV_FIELDS = [
    "seed",
    "env_id",
    "regime",
    "has_hidden_confounder",
    "trajectory_length",
    "num_episodes",
    "contrastive_k",
    "batch_size",
    "collector_mode",
    "oracle_epsilon",
    "num_train_steps",
    "backend",
    "last_loss",
    "last_mean_pos_logit",
    "last_mean_neg_logit",
    "last_mean_pos_minus_neg_logit",
    "mean_loss",
    "mean_mean_pos_logit",
    "mean_mean_neg_logit",
    "mean_mean_pos_minus_neg_logit",
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
        "has_hidden_confounder": r["has_hidden_confounder"],
        "trajectory_length": r["trajectory_length"],
        "num_episodes": r["train_config"].get("num_episodes"),
        "contrastive_k": r["contrastive_k"],
        "batch_size": r["batch_size"],
        "collector_mode": r["train_config"].get("collector_mode"),
        "oracle_epsilon": r["train_config"].get("oracle_epsilon"),
        "num_train_steps": r["num_train_steps"],
        "backend": r["backend"],
        "last_loss": r["last_loss"],
        "last_mean_pos_logit": r["last_mean_pos_logit"],
        "last_mean_neg_logit": r["last_mean_neg_logit"],
        "last_mean_pos_minus_neg_logit": r["last_mean_pos_minus_neg_logit"],
        "mean_loss": r["mean_loss"],
        "mean_mean_pos_logit": r["mean_mean_pos_logit"],
        "mean_mean_neg_logit": r["mean_mean_neg_logit"],
        "mean_mean_pos_minus_neg_logit": r["mean_mean_pos_minus_neg_logit"],
    }


def run_sweep() -> list[dict[str, object]]:
    return run_sweep_config(
        num_episodes=TRAIN_NUM_EPISODES,
        num_steps=TRAIN_NUM_STEPS,
        collector_mode=TRAIN_COLLECTOR_MODE,
        oracle_epsilon=TRAIN_ORACLE_EPSILON,
        save_checkpoint=True,
    )


def run_sweep_config(
    *,
    num_episodes: int,
    num_steps: int,
    collector_mode: str,
    oracle_epsilon: float,
    save_checkpoint: bool,
) -> list[dict[str, object]]:
    print("[Sweep] backend=numpy", file=sys.stderr)

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for env_id in (ENV_CLEAN, ENV_CONF):
            print(f"[Sweep] seed={seed} env={env_id!r} ...", file=sys.stderr, flush=True)
            result = train_numpy(
                seed=seed,
                env_id=env_id,
                num_episodes=num_episodes,
                num_steps=num_steps,
                collector_mode=collector_mode,
                oracle_epsilon=oracle_epsilon,
                verbose=False,
                save_checkpoint=save_checkpoint,
            )
            rows.append(_row_from_result(result))
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="HiddenFork baseline seed × regime sweep → CSV")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_seed_sweep.csv",
        help="CSV path (default: results/hidden_fork_seed_sweep.csv)",
    )
    p.add_argument("--num-episodes", type=int, default=TRAIN_NUM_EPISODES)
    p.add_argument("--num-steps", type=int, default=TRAIN_NUM_STEPS)
    p.add_argument(
        "--collector-mode",
        type=str,
        default=TRAIN_COLLECTOR_MODE,
        choices=["random", "oracle_eps"],
    )
    p.add_argument("--oracle-epsilon", type=float, default=TRAIN_ORACLE_EPSILON)
    p.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Skip writing per-run checkpoints",
    )
    p.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print CSV to stdout only; do not write a file",
    )
    args = p.parse_args()

    rows = run_sweep_config(
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        collector_mode=args.collector_mode,
        oracle_epsilon=args.oracle_epsilon,
        save_checkpoint=not args.no_save_checkpoint,
    )

    if not args.stdout_only:
        _write_csv(rows, args.output)
        print(f"[Sweep] wrote {args.output}", file=sys.stderr)

    w = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(row)


if __name__ == "__main__":
    main()
