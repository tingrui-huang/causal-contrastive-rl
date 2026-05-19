# 实验改动记录

按时间顺序记录每一步改了什么、对应的实验数据。  
**重点：改了什么 + 跑出来什么数**，不做深度分析。

所有训练：50 episodes、20000 steps、lam=0.5、critic_warmup=2000、seed=0。  
所有评估：50 episodes/regime、temperature=1.0。

---

## 起点（未做任何改动前）

- 环境：`CausalContrastive-WindyCorridor-15x15-Lethal-v0`
- 数据：`oracle_eps`, ε=0.2

| Method | u0 | u1 | mean | worst | gap |
|---|---|---|---|---|---|
| Oracle 上界 | 1.00 | 0.92 | 0.96 | 0.92 | 0.08 |
| Baseline confounded | 1.00 | 0.44 | 0.72 | **0.44** | 0.56 |
| Option 1 min_neg | 1.00 | 0.50 | 0.75 | **0.50** | 0.50 |

---

## 改动 1：新增 `multigoal_oracle` 数据收集策略

**动机**：诊断发现 oracle_eps 数据下 critic 训练的 hindsight goal 分布 49.6% 集中在单一 cell，critic 退化成单目标二分类。

**新增文件**：
- `utils/multigoal_oracle.py` — multi-goal oracle 策略（D4RL play 对齐）
- `experiments/diag_collector_coverage.py` — 覆盖度诊断脚本

**修改文件**：
- `utils/offline_data.py::build_policy` 加分派 `collector_mode="multigoal_oracle"`

**诊断（50 ep）**：

| 指标 | oracle_eps | multigoal_oracle |
|---|---|---|
| total_steps | 1571 | 21226 |
| mean episode length | 31 | 425 |
| hindsight max_share | 49.6% | 7.2% |

### 1a. Multigoal v1：`exclude_env_goal=True`（默认）

| Method | u0 | u1 | worst | gap |
|---|---|---|---|---|
| Baseline | 0.02 | 0.08 | 0.02 | 0.06 |
| Option 1 neighbor | 0.06 | 0.06 | 0.06 | 0.00 |

→ 全崩。

**额外跑 lam=0.05 验证**：

| Method | u0 | u1 | worst | gap |
|---|---|---|---|---|
| Baseline lam=0.05 | 0.08 | 0.02 | 0.02 | 0.06 |
| Option 1 neighbor lam=0.05 | 0.12 | 0.04 | 0.04 | 0.08 |

→ lam 不是瓶颈。

### 1b. Multigoal v2：`exclude_env_goal=False`

**修改文件**：`utils/multigoal_oracle.py` 默认值翻转。

| Method | u0 | u1 | worst | gap |
|---|---|---|---|---|
| Baseline | 0.22 | 0.08 | 0.08 | 0.14 |
| Option 1 neighbor | 0.12 | 0.10 | 0.10 | 0.02 |

→ 仍崩。

---

## 改动 2：terminal-state fix

**Bug**：`utils/offline_data.py::collect_episodes` 里构建 `episode_states[ep]` 只收每个 transition 的 `obs`，不收最后一个 transition 的 `next_obs`。导致 agent 走到的最终状态（含 env 真终点）从未进入 hindsight relabel 的候选池。

**修改文件**：`utils/offline_data.py::collect_episodes` 在 states 列表末尾追加 `last["next_obs"]`。

**新增文件**：`experiments/diag_envgoal_share.py` 针对 env_goal 占比的诊断。

**诊断（multigoal_oracle, exclude_env_goal=False）**：

| 指标 | 修前 | 修后 |
|---|---|---|
| reach_rate (u=0) | 0.84 | 0.84 |
| reach_rate (u=1) | 0.77 | 0.77 |
| env_goal (13,5) hindsight share | **0.00%** | **5.80%** |
| top cell | (7,9) 7.2% | (7,9) 6.8% |

### 2a. Multigoal v3（fix 后）

| Method | u0 | u1 | mean | worst | gap |
|---|---|---|---|---|---|
| Baseline | 0.76 | 0.32 | 0.54 | **0.32** | 0.44 |
| Option 1 min_neg | 0.72 | 0.38 | 0.55 | **0.38** | 0.34 |
| Option 1 neighbor | 0.74 | 0.22 | 0.48 | **0.22** | 0.52 |

→ multigoal 上 baseline 恢复正常（0.32），pessimism 没大幅提升。

### 2b. WindyCorridor + oracle_eps 重跑（fix 后）

| Method | u0 | u1 | mean | worst | gap | 对照原 |
|---|---|---|---|---|---|---|
| Baseline | 0.98 | 0.56 | 0.77 | **0.56** | 0.42 | 0.44 → 0.56 (+12 仅 fix) |
| Option 1 min_neg | 1.00 | 0.56 | 0.78 | **0.56** | 0.44 | 0.50 → 0.56（pessimism 边际收益消失） |
| **Option 1 neighbor** | **1.00** | **0.72** | **0.86** | **0.72** | **0.28** | **新跑，pessimism 真贡献 +16** |

---

## 改动 3：实现 Option 2（Thm 2 递归 sampler）

**新增文件**：
- `utils/propensity.py` — empirical action propensity 查表
- `utils/obs_transition.py` — empirical next-state 查表
- `utils/value_function.py` — V_f via `max_{a,g} min(critic1, critic2)`（C1 定义）
- `utils/causal_sampler.py` — Thm 2 递归 sampler 主体
- `experiments/train_causal_thm2.py` — 训练脚本
- `experiments/smoke_thm2_components.py` — 组件 smoke 测试

| Data + Method | u0 | u1 | mean | worst | gap |
|---|---|---|---|---|---|
| WindyCorridor + Thm 2 | 0.98 | 0.50 | 0.74 | 0.50 | 0.48 |
| multigoal v3 + Thm 2 | 0.34 | 0.06 | 0.20 | **0.06** | 0.28 |

训练时 sampler 诊断：
- WindyCorridor oracle_eps：obs:pess = 47.5 : 52.5，mean_depth = 18.9
- multigoal v3：obs:pess = 62 : 38，mean_depth = 18.9

→ Thm 2 在 oracle_eps 上不如 neighbor (0.50 vs 0.72)；在 multigoal 上崩。

---

## 改动 4：新增 `ConfoundedFork` 环境

**新增文件**：
- `envs/confounded_fork.py` — 11×11 单 fork confounded env
- `experiments/smoke_confounded_fork.py` — env smoke

**修改文件**：
- `envs/__init__.py` — 注册 `CausalContrastive-ConfoundedFork-11x11-v0` 等 4 个 id
- `envs/windy_corridor.py` — 放宽 size 约束（11 也允许）
- `utils/offline_data.py::_safe_walkable_for_regime` — 加 env-type 分派（调用 `env.walkable_for_regime` 如果存在）
- `utils/domain_knowledge.py::reachable_positions` — 接受 walkable 参数（不再硬绑 WindyCorridor）
- `utils/domain_knowledge.py::get_neighbor_states_for_batch` — 用 pos_index 推 walkable
- `experiments/train_causal_thm2.py` — 从 env 实例动态拿 walkable

设计：start (1,9)，goal (9,1)，fork 在 (1,8)（start 上 1 步）。

### 4a. ConfoundedFork v1：单边 lava（左路 (2,3..6) 4 格）+ 全方向均匀风 20%

```python
_GLOBAL_WIND_DIST = {0: (0,0,0,0,1), 1: (0.05,0.05,0.05,0.05,0.8)}
```

| Method | u0 | u1 | mean | worst | gap |
|---|---|---|---|---|---|
| Oracle 上界 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 |
| Baseline | 1.00 | 0.98 | 0.99 | **0.98** | 0.02 |
| Option 1 neighbor | 1.00 | 0.98 | 0.99 | **0.98** | 0.02 |
| Option 2 Thm 2 | 1.00 | 0.92 | 0.96 | **0.92** | 0.08 |

→ baseline 已经接近完美，env 太简单。

### 4b. ConfoundedFork v2：双边 lava（加右路 (8,4..5) 2 格）+ 风同 v1

**修改**：`envs/confounded_fork.py::hazard_cells` 加 `(8,4), (8,5)`。

| Method | u0 | u1 | mean | worst | gap |
|---|---|---|---|---|---|
| Oracle 上界 | 1.00 | 0.96 | 0.98 | **0.96** | 0.04 |
| Baseline | 1.00 | 0.96 | 0.98 | **0.96** | 0.04 |
| Option 1 neighbor | 1.00 | 0.94 | 0.97 | **0.94** | 0.06 |
| Option 2 Thm 2 | 1.00 | 0.88 | 0.94 | **0.88** | 0.12 |

→ 加 2 lava 仍不够"咬"baseline。

### 4c. ConfoundedFork v3：有向风 40%（U=0 东风 / U=1 西风）+ 双边 lava

**修改**：
- `envs/confounded_fork.py::_GLOBAL_WIND_DIST` 改成 `{0: (0.4,0,0,0,0.6), 1: (0,0,0.4,0,0.6)}`
- `envs/confounded_fork.py::walkable_for_regime` 翻转语义：U=0 强制 right（左路东风进 lava）、U=1 强制 left（右路西风进 lava）

| Method | u0 | u1 | mean | worst | gap |
|---|---|---|---|---|---|
| Oracle 上界 | 0.92 | 0.40 | 0.66 | **0.40** | 0.52 |
| Baseline | 0.86 | 0.40 | 0.63 | **0.40** | 0.46 |
| Option 1 neighbor | 0.82 | 0.38 | 0.60 | **0.38** | 0.44 |
| Option 2 Thm 2 | 0.40 | 0.76 | 0.58 | **0.40** | 0.36 |

→ confounding 起作用，所有方法触顶 ~0.40。理论上限（"always right" 策略）≈ 0.36。

---

## 最终文件清单

### 新增
- `envs/confounded_fork.py`
- `utils/multigoal_oracle.py`
- `utils/propensity.py`
- `utils/obs_transition.py`
- `utils/value_function.py`
- `utils/causal_sampler.py`
- `experiments/train_causal_thm2.py`
- `experiments/smoke_thm2_components.py`
- `experiments/smoke_confounded_fork.py`
- `experiments/diag_collector_coverage.py`
- `experiments/diag_envgoal_share.py`
- `EXPERIMENT_LOG.md`（本文件）

### 修改
- `envs/__init__.py` — 注册 ConfoundedFork ids
- `envs/windy_corridor.py` — 放宽 size 约束（11/15）
- `utils/offline_data.py` — terminal-state fix + multigoal dispatch + env-type 分派
- `utils/domain_knowledge.py` — reachable_positions 加 walkable 参数 + neighbor 函数用 pos_index 推 walkable
