"""
一键运行入口。在 JetBrains 里直接点右上角三角运行。

使用方法：
  1. 改下面的 TASK 变量，选你想跑的任务
  2. 点右上角三角运行

可选任务（复制其中一个填到 TASK）：
  "eval_oracle"       ── 评估 Oracle 模型，结果写入 CSV
  "eval_confounded"   ── 评估 Confounded 模型，结果写入 CSV
  "eval_causal_pessimistic" ── 评估 Causal Pessimistic 模型，结果写入 CSV
  "train_oracle"      ── 训练 Oracle 模型（有 U 信息）
  "train_confounded"  ── 训练 Confounded 模型（无 U 信息）
  "train_causal_pessimistic" ── 训练 Causal Pessimistic 模型
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ────────────────────────────────────────────
#  ★ 改这里选任务 ★
# ────────────────────────────────────────────
TASK = "eval_oracle"
# ────────────────────────────────────────────

import envs  # noqa: F401 — 注册环境

from configs.run_config import (
    EVAL_CAUSAL_PESSIMISTIC_CHECKPOINT,
    EVAL_CAUSAL_PESSIMISTIC_OUTPUT,
    EVAL_CONFOUNDED_CHECKPOINT,
    EVAL_CONFOUNDED_OUTPUT,
    EVAL_ENV_ID,
    EVAL_EPISODES,
    EVAL_ORACLE_CHECKPOINT,
    EVAL_ORACLE_OUTPUT,
    EVAL_TEMPERATURE,
    TRAIN_ACTOR_LR,
    TRAIN_CRITIC_LR,
    TRAIN_CRITIC_WARMUP,
    TRAIN_ENV_ID,
    TRAIN_CONSTANT_M,
    TRAIN_LAM,
    TRAIN_NEG_GOALS_N,
    TRAIN_NEIGHBOR_MAX_MANHATTAN,
    TRAIN_NEIGHBOR_MAX_PER_SAMPLE,
    TRAIN_NUM_EPISODES,
    TRAIN_NUM_STEPS,
    TRAIN_ORACLE_EPSILON,
    TRAIN_PESSIMISM_MODE,
    TRAIN_SEED,
)
from experiments.evaluate_actor_forced_u import CSV_FIELDS, evaluate_checkpoint
from experiments.train_actor_critic import train as _train
from experiments.train_causal_pessimistic import train as _train_causal_pessimistic
import csv


def _save_csv(path: str, row: dict) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(f"[run] 结果已写入: {p}")


# ────────────────────────────────────────────
# 任务分派
# ────────────────────────────────────────────

if TASK == "eval_oracle":
    row = evaluate_checkpoint(
        checkpoint=ROOT / EVAL_ORACLE_CHECKPOINT,
        eval_env_id=EVAL_ENV_ID,
        episodes_per_regime=EVAL_EPISODES,
        temperature=EVAL_TEMPERATURE,
    )
    _save_csv(EVAL_ORACLE_OUTPUT, row)

elif TASK == "eval_confounded":
    row = evaluate_checkpoint(
        checkpoint=ROOT / EVAL_CONFOUNDED_CHECKPOINT,
        eval_env_id=EVAL_ENV_ID,
        episodes_per_regime=EVAL_EPISODES,
        temperature=EVAL_TEMPERATURE,
    )
    _save_csv(EVAL_CONFOUNDED_OUTPUT, row)

elif TASK == "eval_causal_pessimistic":
    row = evaluate_checkpoint(
        checkpoint=ROOT / EVAL_CAUSAL_PESSIMISTIC_CHECKPOINT,
        eval_env_id=EVAL_ENV_ID,
        episodes_per_regime=EVAL_EPISODES,
        temperature=EVAL_TEMPERATURE,
    )
    _save_csv(EVAL_CAUSAL_PESSIMISTIC_OUTPUT, row)

elif TASK == "train_oracle":
    _train(
        seed=TRAIN_SEED,
        env_id=TRAIN_ENV_ID,
        num_episodes=TRAIN_NUM_EPISODES,
        num_steps=TRAIN_NUM_STEPS,
        oracle_state=True,
        lam=TRAIN_LAM,
        critic_lr=TRAIN_CRITIC_LR,
        actor_lr=TRAIN_ACTOR_LR,
        critic_warmup=TRAIN_CRITIC_WARMUP,
        oracle_epsilon=TRAIN_ORACLE_EPSILON,
    )

elif TASK == "train_confounded":
    _train(
        seed=TRAIN_SEED,
        env_id=TRAIN_ENV_ID,
        num_episodes=TRAIN_NUM_EPISODES,
        num_steps=TRAIN_NUM_STEPS,
        oracle_state=False,
        lam=TRAIN_LAM,
        critic_lr=TRAIN_CRITIC_LR,
        actor_lr=TRAIN_ACTOR_LR,
        critic_warmup=TRAIN_CRITIC_WARMUP,
        oracle_epsilon=TRAIN_ORACLE_EPSILON,
    )

elif TASK == "train_causal_pessimistic":
    _train_causal_pessimistic(
        seed=TRAIN_SEED,
        env_id=TRAIN_ENV_ID,
        num_episodes=TRAIN_NUM_EPISODES,
        num_steps=TRAIN_NUM_STEPS,
        oracle_state=False,
        lam=TRAIN_LAM,
        critic_lr=TRAIN_CRITIC_LR,
        actor_lr=TRAIN_ACTOR_LR,
        critic_warmup=TRAIN_CRITIC_WARMUP,
        oracle_epsilon=TRAIN_ORACLE_EPSILON,
        pessimism_mode=TRAIN_PESSIMISM_MODE,
        constant_M=TRAIN_CONSTANT_M,
        neg_goals_n=TRAIN_NEG_GOALS_N,
        neighbor_max_manhattan=TRAIN_NEIGHBOR_MAX_MANHATTAN,
        neighbor_max_per_sample=TRAIN_NEIGHBOR_MAX_PER_SAMPLE,
    )

else:
    print(f"[run] 未知任务: {TASK!r}")
    print(
        "可选任务: eval_oracle / eval_confounded / eval_causal_pessimistic / "
        "train_oracle / train_confounded / train_causal_pessimistic"
    )
