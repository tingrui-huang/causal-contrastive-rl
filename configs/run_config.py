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
EVAL_CAUSAL_PESSIMISTIC_CHECKPOINT = "checkpoints/causal_pessimistic_neighbor_seed0_CausalContrastive_WindyCorridor_15x15_Lethal_v0_oracle_eps.npz"

EVAL_CONFOUNDED_OUTPUT = "results/actor_confounded_forced_u.csv"
EVAL_ORACLE_OUTPUT     = "results/actor_oracle_forced_u.csv"
EVAL_CAUSAL_PESSIMISTIC_OUTPUT = "results/actor_causal_pessimistic_forced_u.csv"

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
TRAIN_PESSIMISM_MODE = "neighbor"  # {"neighbor", "min_neg", "constant"}
TRAIN_CONSTANT_M = 2.0             # 相对当前 batch 最小 critic score 再往下压多少
TRAIN_NEG_GOALS_N = 16             # 悲观 min-neg 模式下采样多少个负 goal
TRAIN_NEIGHBOR_MAX_MANHATTAN = 2   # neighbor 模式：曼哈顿距离上限（覆盖风漂移）
TRAIN_NEIGHBOR_MAX_PER_SAMPLE = 32 # neighbor 模式：每个样本最多用多少个邻居状态
