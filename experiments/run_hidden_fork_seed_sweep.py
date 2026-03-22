"""
Run 10 contrastive-baseline experiments: seeds ``0..4`` ×
``HiddenFork-15x15-Clean`` vs ``HiddenFork-15x15`` (confounded).

Uses the same training loop as ``train_contrastive_baseline.py`` but with
``verbose=False`` and ``save_checkpoint=False`` so checkpoints are not overwritten.

Output: **CSV** (header + one row per run) — printed to stdout and saved under
``results/`` by default.

```bash
python experiments/run_hidden_fork_seed_sweep.py
python experiments/run_hidden_fork_seed_sweep.py -o my_results.csv
```
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.train_contrastive_baseline import train_numpy, train_torch

_TORCH = False
try:
    import torch  # noqa: F401

    _TORCH = True
except OSError:
    pass

ENV_CLEAN = "CausalContrastive-HiddenFork-15x15-Clean-v0"
ENV_CONF = "CausalContrastive-HiddenFork-15x15-v0"
SEEDS = (0, 1, 2,3,4,5)

# Flat CSV columns (no nested train_config)
CSV_FIELDS = [
    "seed",
    "env_id",
    "regime",
    "has_hidden_confounder",
    "trajectory_length",
    "contrastive_k",
    "batch_size",
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
        "contrastive_k": r["contrastive_k"],
        "batch_size": r["batch_size"],
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
    train = train_torch if _TORCH else train_numpy
    backend = "torch" if _TORCH else "numpy"
    print(f"[Sweep] backend={backend}", file=sys.stderr)

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for env_id in (ENV_CLEAN, ENV_CONF):
            print(f"[Sweep] seed={seed} env={env_id!r} ...", file=sys.stderr, flush=True)
            result = train(
                seed=seed,
                env_id=env_id,
                verbose=False,
                save_checkpoint=False,
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
    p = argparse.ArgumentParser(description="10-run HiddenFork seed × regime sweep → CSV")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_seed_sweep.csv",
        help="CSV path (default: results/hidden_fork_seed_sweep.csv)",
    )
    p.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print CSV to stdout only; do not write a file",
    )
    args = p.parse_args()

    rows = run_sweep()

    if not args.stdout_only:
        _write_csv(rows, args.output)
        print(f"[Sweep] wrote {args.output}", file=sys.stderr)

    w = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow(row)


if __name__ == "__main__":
    main()
