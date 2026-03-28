"""
Run the full confounded HiddenTrap HiddenFork batch unattended:

1. Train baseline for 5 seeds.
2. Train robust_v1 for 5 seeds x requested weights.
3. Evaluate all produced checkpoints with forced-U.
4. Write a compact main table averaged across seeds.

This script is meant to be launched once and left running overnight.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.training_defaults import (  # noqa: E402
    ROBUST_V1_COLLECTOR_MODE,
    ROBUST_V1_LOGIT_SCALE,
    ROBUST_V1_M,
    ROBUST_V1_NUM_EPISODES,
    ROBUST_V1_ORACLE_EPSILON,
    ROBUST_V1_P,
    TRAIN_NUMPY_LR,
    TRAIN_NUM_STEPS,
)
from experiments.evaluate_hidden_fork_forced_u import (  # noqa: E402
    ACTION_NAME_TO_ID,
    CSV_FIELDS as EVAL_CSV_FIELDS,
    ENV_CONF,
    evaluate_checkpoint,
)
from experiments.train_contrastive_baseline import train_numpy  # noqa: E402
from experiments.train_contrastive_robust_v1 import train_numpy_robust_v1  # noqa: E402

DEFAULT_BATCH_SEEDS = (0, 1)
DEFAULT_BATCH_WEIGHTS = (0.5, 0.8, 1.0)

BASELINE_TRAIN_FIELDS = [
    "method",
    "seed",
    "env_id",
    "num_episodes",
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
    "checkpoint_path",
]

ROBUST_TRAIN_FIELDS = [
    "method",
    "seed",
    "env_id",
    "num_episodes",
    "collector_mode",
    "oracle_epsilon",
    "num_train_steps",
    "backend",
    "P",
    "M",
    "w",
    "logit_scale",
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
    "checkpoint_path",
]

MAIN_TABLE_FIELDS = [
    "method",
    "w",
    "n_seeds",
    "success_u0",
    "success_u1",
    "mean_success",
    "worst_case_success",
    "regime_gap",
    "fork_visit_rate_u0",
    "fork_action_prob_left_u0",
    "fork_action_prob_right_u0",
    "fork_action_prob_forward_u0",
    "fork_visit_rate_u1",
    "fork_action_prob_left_u1",
    "fork_action_prob_right_u1",
    "fork_action_prob_forward_u1",
]


def _write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _baseline_row(result: dict[str, object]) -> dict[str, object]:
    cfg = result["train_config"]
    return {
        "method": "softmax_baseline",
        "seed": result["seed"],
        "env_id": result["env_id"],
        "num_episodes": cfg.get("num_episodes"),
        "collector_mode": cfg.get("collector_mode"),
        "oracle_epsilon": cfg.get("oracle_epsilon"),
        "num_train_steps": result["num_train_steps"],
        "backend": result["backend"],
        "last_loss": result["last_loss"],
        "last_mean_pos_logit": result["last_mean_pos_logit"],
        "last_mean_neg_logit": result["last_mean_neg_logit"],
        "last_mean_pos_minus_neg_logit": result["last_mean_pos_minus_neg_logit"],
        "mean_loss": result["mean_loss"],
        "mean_mean_pos_logit": result["mean_mean_pos_logit"],
        "mean_mean_neg_logit": result["mean_mean_neg_logit"],
        "mean_mean_pos_minus_neg_logit": result["mean_mean_pos_minus_neg_logit"],
        "checkpoint_path": result.get("checkpoint_path"),
    }


def _robust_row(result: dict[str, object]) -> dict[str, object]:
    return {
        "method": "robust_v1",
        "seed": result["seed"],
        "env_id": result["env_id"],
        "num_episodes": result["num_episodes"],
        "collector_mode": result["collector_mode"],
        "oracle_epsilon": result["oracle_epsilon"],
        "num_train_steps": result["num_train_steps"],
        "backend": result["backend"],
        "P": result["P"],
        "M": result["M"],
        "w": result["w"],
        "logit_scale": result["logit_scale"],
        "last_loss": result["last_loss"],
        "last_mean_pos_logit": result["last_mean_pos_logit"],
        "last_mean_neg_logit": result["last_mean_neg_logit"],
        "last_mean_pos_minus_neg_logit": result["last_mean_pos_minus_neg_logit"],
        "last_frac_pos_logit_gt_neg_logit": result["last_frac_pos_logit_gt_neg_logit"],
        "last_mean_pos_obs_minus_surr_logit": result["last_mean_pos_obs_minus_surr_logit"],
        "last_mean_neg_surr_minus_obs_logit": result["last_mean_neg_surr_minus_obs_logit"],
        "mean_loss": result["mean_loss"],
        "mean_mean_pos_logit": result["mean_mean_pos_logit"],
        "mean_mean_neg_logit": result["mean_mean_neg_logit"],
        "mean_mean_pos_minus_neg_logit": result["mean_mean_pos_minus_neg_logit"],
        "mean_frac_pos_logit_gt_neg_logit": result["mean_frac_pos_logit_gt_neg_logit"],
        "mean_mean_pos_obs_minus_surr_logit": result["mean_mean_pos_obs_minus_surr_logit"],
        "mean_mean_neg_surr_minus_obs_logit": result["mean_mean_neg_surr_minus_obs_logit"],
        "checkpoint_path": result.get("checkpoint_path"),
    }


def _summarize_main_table(eval_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in eval_rows:
        method = str(row["method"])
        w = "" if row.get("w") in (None, "") else f"{float(row['w']):.2f}"
        groups[(method, w)].append(row)

    out: list[dict[str, object]] = []
    metric_fields = [field for field in MAIN_TABLE_FIELDS if field not in {"method", "w", "n_seeds"}]
    for (method, w), rows in sorted(groups.items()):
        summary: dict[str, object] = {
            "method": method,
            "w": w if w else None,
            "n_seeds": len(rows),
        }
        for field in metric_fields:
            summary[field] = mean(float(r[field]) for r in rows)
        out.append(summary)
    return out


def run_pipeline(
    *,
    seeds: tuple[int, ...],
    weights: tuple[float, ...],
    baseline_output: Path,
    robust_output: Path,
    eval_output: Path,
    main_table_output: Path,
    episodes_per_regime: int,
    action_subset: str,
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_mode: str,
    plan_depth: int,
    max_eval_steps: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
) -> None:
    baseline_rows: list[dict[str, object]] = []
    robust_rows: list[dict[str, object]] = []
    eval_rows: list[dict[str, object]] = []

    checkpoint_paths: list[Path] = []

    print(f"[Batch] training baseline ({len(seeds)} seeds, confounded only)...", flush=True)
    for seed in seeds:
        print(f"[Batch] baseline seed={seed}", flush=True)
        result = train_numpy(
            seed=seed,
            env_id=ENV_CONF,
            num_episodes=ROBUST_V1_NUM_EPISODES,
            num_steps=TRAIN_NUM_STEPS,
            collector_mode=ROBUST_V1_COLLECTOR_MODE,
            oracle_epsilon=ROBUST_V1_ORACLE_EPSILON,
            verbose=False,
            save_checkpoint=True,
        )
        baseline_rows.append(_baseline_row(result))
        checkpoint_paths.append(Path(str(result["checkpoint_path"])))
        _write_csv(baseline_rows, baseline_output, BASELINE_TRAIN_FIELDS)

    print(
        f"[Batch] training robust_v1 ({len(seeds)} seeds x {len(weights)} weights, confounded only)...",
        flush=True,
    )
    for w in weights:
        for seed in seeds:
            print(f"[Batch] robust_v1 w={w:.2f} seed={seed}", flush=True)
            result = train_numpy_robust_v1(
                seed=seed,
                env_id=ENV_CONF,
                num_episodes=ROBUST_V1_NUM_EPISODES,
                num_steps=TRAIN_NUM_STEPS,
                lr=TRAIN_NUMPY_LR,
                P=ROBUST_V1_P,
                M=ROBUST_V1_M,
                w=float(w),
                collector_mode=ROBUST_V1_COLLECTOR_MODE,
                oracle_epsilon=ROBUST_V1_ORACLE_EPSILON,
                logit_scale=ROBUST_V1_LOGIT_SCALE,
                save_checkpoint=True,
                verbose=False,
            )
            robust_rows.append(_robust_row(result))
            checkpoint_paths.append(Path(str(result["checkpoint_path"])))
            _write_csv(robust_rows, robust_output, ROBUST_TRAIN_FIELDS)

    action_names = [part.strip() for part in action_subset.split(",") if part.strip()]
    action_ids = [ACTION_NAME_TO_ID[name] for name in action_names]
    action_subset_label = ",".join(action_names)

    print("[Batch] forced-U evaluation over all produced checkpoints...", flush=True)
    for ckpt in checkpoint_paths:
        print(f"[Batch] eval checkpoint={ckpt.name}", flush=True)
        row = evaluate_checkpoint(
            checkpoint_path=ckpt,
            eval_env_id=ENV_CONF,
            episodes_per_regime=episodes_per_regime,
            action_ids=action_ids,
            action_subset_label=action_subset_label,
            lookahead_k=lookahead_k,
            future_window=future_window,
            alignment_mode=alignment_mode,
            goal_mode=goal_mode,
            plan_depth=plan_depth,
            max_eval_steps=max_eval_steps,
            collision_penalty=collision_penalty,
            turn_penalty=turn_penalty,
            progress_bonus=progress_bonus,
            success_bonus=success_bonus,
            debug=False,
            debug_max_steps=0,
        )
        eval_rows.append(row)
        _write_csv(eval_rows, eval_output, EVAL_CSV_FIELDS)
        main_rows = _summarize_main_table(eval_rows)
        _write_csv(main_rows, main_table_output, MAIN_TABLE_FIELDS)

    print(f"[Batch] wrote baseline training CSV: {baseline_output}", flush=True)
    print(f"[Batch] wrote robust training CSV:   {robust_output}", flush=True)
    print(f"[Batch] wrote forced-U raw CSV:      {eval_output}", flush=True)
    print(f"[Batch] wrote forced-U main table:   {main_table_output}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Overnight confounded HiddenTrap batch runner (default: merge-shared closed-book 2-seed ablation)"
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SEEDS),
        help="Seed list for both baseline and robust runs (default: 0 1)",
    )
    p.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_BATCH_WEIGHTS),
        help="Robust_v1 weight list (default: 0.5 0.8 1.0)",
    )
    p.add_argument(
        "--baseline-output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_hidden_trap_confounded_baseline_2seed.csv",
    )
    p.add_argument(
        "--robust-output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_hidden_trap_confounded_robust_v1_merge_shared_2seed.csv",
    )
    p.add_argument(
        "--eval-output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_hidden_trap_forced_u_eval_merge_shared_2seed.csv",
    )
    p.add_argument(
        "--main-table-output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_hidden_trap_forced_u_main_table_merge_shared_2seed.csv",
    )
    p.add_argument("--episodes-per-regime", type=int, default=100)
    p.add_argument("--action-subset", type=str, default="left,right,forward")
    p.add_argument("--lookahead-k", type=int, default=4)
    p.add_argument("--future-window", type=int, default=2)
    p.add_argument(
        "--alignment-mode",
        type=str,
        default="time",
        choices=["time", "nearest"],
    )
    p.add_argument(
        "--goal-mode",
        type=str,
        default="merge_shared",
        choices=["open_book", "merge_shared"],
    )
    p.add_argument("--plan-depth", type=int, default=2)
    p.add_argument("--max-eval-steps", type=int, default=60)
    p.add_argument("--collision-penalty", type=float, default=3.0)
    p.add_argument("--turn-penalty", type=float, default=0.05)
    p.add_argument("--progress-bonus", type=float, default=0.75)
    p.add_argument("--success-bonus", type=float, default=5.0)
    args = p.parse_args()

    run_pipeline(
        seeds=tuple(args.seeds),
        weights=tuple(args.weights),
        baseline_output=args.baseline_output,
        robust_output=args.robust_output,
        eval_output=args.eval_output,
        main_table_output=args.main_table_output,
        episodes_per_regime=args.episodes_per_regime,
        action_subset=args.action_subset,
        lookahead_k=args.lookahead_k,
        future_window=args.future_window,
        alignment_mode=args.alignment_mode,
        goal_mode=args.goal_mode,
        plan_depth=args.plan_depth,
        max_eval_steps=args.max_eval_steps,
        collision_penalty=args.collision_penalty,
        turn_penalty=args.turn_penalty,
        progress_bonus=args.progress_bonus,
        success_bonus=args.success_bonus,
    )


if __name__ == "__main__":
    main()
