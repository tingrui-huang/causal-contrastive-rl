"""
实验参数配置。

使用方法：
  1. 在这里改参数
  2. 去 experiments/run.py 选任务名，点运行
"""

# ─────────────────────────────────────────────
# 评估参数
# ─────────────────────────────────────────────

EVAL_ENV_ID = "CausalContrastive-WindyCorridor-15x15-Lethal-v0"
EVAL_EPISODES = 50          # 每个 regime 跑多少 episode（越多越准，越慢）
EVAL_TEMPERATURE = 1.0      # 动作采样温度（1.0=正常, 越低越贪心）

EVAL_CONFOUNDED_CHECKPOINT = "checkpoints/actor_critic_confounded_seed0_CausalContrastive_WindyCorridor_15x15_Lethal_v0_oracle_eps.npz"
EVAL_ORACLE_CHECKPOINT     = "checkpoints/actor_critic_oracle_seed0_CausalContrastive_WindyCorridor_15x15_Lethal_v0_oracle_eps.npz"

EVAL_CONFOUNDED_OUTPUT = "results/actor_confounded_forced_u.csv"
EVAL_ORACLE_OUTPUT     = "results/actor_oracle_forced_u.csv"

# ─────────────────────────────────────────────
# 训练参数
# ─────────────────────────────────────────────

TRAIN_ENV_ID       = "CausalContrastive-WindyCorridor-15x15-Lethal-v0"
TRAIN_SEED         = 0
TRAIN_NUM_EPISODES = 50
TRAIN_NUM_STEPS    = 20000
TRAIN_LAM          = 0.5     # BC 系数（0=纯 advantage，1=纯 behavior cloning）
TRAIN_CRITIC_LR    = 0.002
TRAIN_ACTOR_LR     = 0.002
TRAIN_CRITIC_WARMUP = 2000   # 先只训 critic 多少步，再开始训 actor
TRAIN_ORACLE_EPSILON = 0.2   # 数据收集时随机探索概率
