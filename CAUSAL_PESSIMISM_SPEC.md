# Causal Pessimism Execution Protocol (Phase 9)

> **WARNING FOR CODE GENERATION (VIBE CODING):**
> This document is the strict execution protocol for implementing the causal pessimistic
> contrastive RL variant. Do NOT invent new mechanics outside the phased plan. Do NOT
> optimize for creativity. Follow phases strictly. A phase is NOT complete unless its
> verification script passes and all required log lines match.

**Repository:** `causal-contrastive-rl`  
**Companion specs:** `DEV_SPEC.md` (pipeline), `ENV_DEV_SPEC.md` (environment), `README.md`  
**Prerequisite:** Phase 8 actor-critic baseline completed; forced-U evaluation shows regime gap.

---

## 0. Motivation & Research Context

### 0.1 Problem Statement

The baseline contrastive RL critic learns:

$$f(s, a, s_g) = \phi(s, a)^T \psi(s_g) / \tau$$

This score approximates the log-probability of reaching $s_g$ from $(s, a)$ under the
**observational** data distribution, which includes the hidden confounder $U$. When the
actor maximizes this score for **all** candidate actions, it implicitly trusts the critic's
estimate even for actions $a \neq a_{obs}$ — actions that were never actually executed in
the data. Under confounding, these counterfactual estimates are unreliable because the
data distribution $p(s' \mid s, a)$ conflates the effect of $a$ with the effect of $U$.

### 0.2 Proposed Fix: Causal Pessimism

Split the algorithm into three layers:

1. **Critic (representation learning):** Keep the standard InfoNCE/sigmoid-BCE loss.
   The critic should faithfully fit the observational distribution — no pessimism here.

2. **Pessimistic Q-evaluation:** Define a new scoring function $\underline{Q}(s, a, s_g)$
   that trusts the critic only for observed actions and applies a pessimistic lower bound
   for counterfactual actions:

$$
\underline{Q}(s, a, s_g) =
\begin{cases}
\phi(s, a)^T \psi(s_g) & \text{if } a = a_{obs} \\
\min_{s_{neg} \in \mathcal{B}_{neg}} \phi(s, a)^T \psi(s_{neg}) & \text{if } a \neq a_{obs}
\end{cases}
$$

3. **Actor:** Maximize the pessimistic $\underline{Q}$ instead of the raw critic scores.

### 0.3 Physical Interpretation

- **$a = a_{obs}$:** The action was actually taken in the data. The critic's estimate is
  grounded in real transitions, so we trust it.
- **$a \neq a_{obs}$:** This is a counterfactual action. We don't know what would have
  happened. The pessimistic estimate says: "assume this action leads to the state most
  dissimilar to your goal." This prevents the actor from being optimistic about untested
  actions whose apparent value might be an artifact of the confounder.

### 0.4 Engineering Variants for $a \neq a_{obs}$

| Variant | Formula | Pros | Cons |
|---------|---------|------|------|
| **min-neg** | $\min_{s_{neg} \in \mathcal{B}_{neg}} \phi(s,a)^T \psi(s_{neg})$ | Data-driven, adapts to representation | Requires negative sampling |
| **constant-penalty** | $-M$ (fixed large negative) | Simple, no extra computation | Hyperparameter $M$; ignores representation |

Both variants will be implemented. The **min-neg** variant is the primary method; the
**constant-penalty** variant is an ablation.

---

## 1. Scope & Non-Goals

### 1.1 In Scope

- New agent module: `agents/causal_pessimistic_actor.py`
- New training script: `experiments/train_causal_pessimistic.py`
- New evaluation entry in `experiments/run.py` and `configs/run_config.py`
- Forced-U evaluation using the **existing** `evaluate_actor_forced_u.py`
- CSV results in `results/`

### 1.2 Out of Scope (FORBIDDEN)

- Modifying `agents/contrastive_critic_numpy.py` (critic stays unchanged)
- Modifying `agents/goal_conditioned_actor_numpy.py` (baseline actor stays unchanged)
- Modifying `envs/` (environment stays unchanged)
- Modifying `utils/offline_data.py` (data collection stays unchanged)
- Modifying `experiments/evaluate_actor_forced_u.py` (evaluation stays unchanged)
- Introducing PyTorch, CNNs, transformers, or any architecture change
- Changing the observation/state representation

---

## 2. Coding Principles (Inherited + Extended)

All principles from `DEV_SPEC.md` §2 apply. Additional rules for this phase:

| Principle | What it means |
|-----------|---------------|
| **Critic is frozen during actor training** | Gradients MUST NOT flow through $\phi$ or $\psi$ when computing actor loss. |
| **Same offline dataset** | The causal-pessimistic actor MUST be trained on the exact same offline episodes as the baseline. Use the same `collect_episodes()` call with identical parameters. |
| **Same evaluation** | Use the same `evaluate_actor_forced_u.py` with identical settings. |
| **Explicit variant labeling** | Every checkpoint, CSV row, and log line must identify the method as `causal_pessimistic_min_neg` or `causal_pessimistic_constant_M`. |

---

## 3. Phased Roadmap

### Phase 9.0 — Verify Baseline Still Works

**Task:** Confirm the existing baseline actor-critic training and evaluation still run
correctly before any code changes.

**Verification commands:**

```bash
# Re-train baseline (confounded)
python experiments/run.py
# with TASK = "train_confounded"

# Re-evaluate
python experiments/run.py
# with TASK = "eval_confounded"
```

**Success criteria (STRICT):**

- Training completes without error.
- Evaluation CSV matches previous results (within stochastic tolerance).
- No import errors, no `NaN`.

**Gate:** Do NOT proceed to Phase 9.1 until this passes.

---

### Phase 9.1 — Implement `CausalPessimisticActorNumpy`

**Task:** Create `agents/causal_pessimistic_actor.py` containing a new actor class that
uses the pessimistic Q-evaluation mechanism.

**File:** `agents/causal_pessimistic_actor.py`

**Class:** `CausalPessimisticActorNumpy`

**Architecture:** Identical MLP to `GoalConditionedActorNumpy` (same `state_dim * 2`
input, same hidden layer, same softmax output). The only difference is in the loss
computation.

**Constructor signature:**

```python
class CausalPessimisticActorNumpy:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        seed: int = 0,
        pessimism_mode: str = "min_neg",  # {"min_neg", "constant"}
        constant_M: float = 10.0,
    ) -> None:
```

**Required methods (same interface as `GoalConditionedActorNumpy`):**

- `action_probs(state, goal) -> (B, n_actions)` — identical to baseline
- `greedy_action(state, goal, valid_actions) -> int` — identical to baseline
- `sample_action(state, goal, valid_actions, rng, temperature) -> int` — identical to baseline
- `apply_sgd(grads, lr) -> None` — identical to baseline
- `state_dict() -> dict` — identical to baseline (plus `pessimism_mode`, `constant_M`)

**New method — the core change:**

```python
def loss_and_grads(
    self,
    state: np.ndarray,        # (B, state_dim)
    goal: np.ndarray,         # (B, state_dim)
    a_obs: np.ndarray,        # (B,) int — observed actions from dataset
    critic_scores: np.ndarray, # (B, n_actions) — f(s, a_i, sg) for all actions
    neg_goals: np.ndarray,    # (N, state_dim) — negative goal states for min-neg
    phi_fn,                   # callable: (s, a_batch) -> (B, emb_dim)
    psi_fn,                   # callable: (goals) -> (N, emb_dim)
    lam: float = 0.5,
) -> tuple[float, dict[str, np.ndarray]]:
```

**Pessimistic Q computation (STRICT — this is the core algorithm):**

```python
# For each sample i in batch, for each candidate action a_j:
#   if a_j == a_obs[i]:
#       Q_pessimistic[i, a_j] = critic_scores[i, a_j]   (trust the critic)
#   else:
#       if pessimism_mode == "min_neg":
#           h_sa = phi_fn(state[i:i+1], a_j)             # (1, emb_dim)
#           h_neg = psi_fn(neg_goals)                     # (N, emb_dim)
#           Q_pessimistic[i, a_j] = min(h_sa @ h_neg.T)  # scalar
#       elif pessimism_mode == "constant":
#           Q_pessimistic[i, a_j] = -constant_M
```

**Actor loss (same structure as baseline, but using pessimistic Q):**

$$\mathcal{L}_{Actor}(\theta) = -(1-\lambda) \cdot \mathbb{E}\left[\sum_a \pi_\theta(a|s,s_g) \cdot \underline{Q}(s,a,s_g)\right] - \lambda \cdot \mathbb{E}\left[\log \pi_\theta(a_{obs}|s,s_g)\right]$$

**Implementation constraints:**

- `phi_fn` and `psi_fn` are passed as callables from the frozen critic. They MUST NOT
  be part of the actor's computation graph. The actor's gradients flow only through
  $\pi_\theta$.
- The pessimistic Q values are treated as **detached constants** (like `critic_scores`
  in the baseline). Only the actor's softmax probabilities have gradients.
- The gradient computation for the actor MLP is identical to `GoalConditionedActorNumpy`
  — only the `critic_scores` input is replaced by `Q_pessimistic`.

**Verification command:**

```bash
python -c "
import numpy as np
from agents.causal_pessimistic_actor import CausalPessimisticActorNumpy

actor = CausalPessimisticActorNumpy(state_dim=10, n_actions=3, seed=0)
s = np.random.randn(4, 10)
g = np.random.randn(4, 10)
print('[Phase9.1] action_probs shape:', actor.action_probs(s, g).shape)
print('[Phase9.1] PASS')
"
```

**Success criteria (STRICT):**

- Class instantiates without error.
- `action_probs` returns shape `(B, n_actions)`.
- `state_dict()` includes `pessimism_mode` and `constant_M`.
- No imports from `agents/goal_conditioned_actor_numpy.py` (standalone implementation,
  may share logic but must be a separate file).

---

### Phase 9.2 — Implement Pessimistic Q Computation (Unit Test)

**Task:** Verify the pessimistic Q computation produces correct values in a controlled
setting.

**File:** `experiments/smoke_causal_pessimistic.py`

**Test design:**

1. Create a tiny critic (state_dim=4, n_actions=3, emb_dim=8).
2. Create a batch of 2 samples with known `a_obs = [0, 2]`.
3. Compute `critic_scores` for all actions (3 per sample).
4. Create 5 random negative goals.
5. Compute pessimistic Q for all actions.
6. **Verify:** For `a_obs` actions, pessimistic Q equals critic score exactly.
7. **Verify:** For non-`a_obs` actions, pessimistic Q ≤ critic score (pessimism).
8. **Verify (constant mode):** For non-`a_obs` actions, pessimistic Q = -M exactly.

**Required logging (exact prefixes):**

```text
[Phase9.2] critic_scores shape = <tuple>
[Phase9.2] pessimistic_Q shape = <tuple>
[Phase9.2] observed_action_match = <bool>
[Phase9.2] counterfactual_pessimism = <bool>
[Phase9.2] constant_mode_exact = <bool>
[Phase9.2] PASS
```

**Success criteria (STRICT):**

- All three boolean checks are `True`.
- `[Phase9.2] PASS` appears in stdout.

**Gate:** Do NOT proceed to Phase 9.3 until this passes.

---

### Phase 9.3 — Implement Training Script

**Task:** Create `experiments/train_causal_pessimistic.py` that trains the causal
pessimistic actor on the same offline data as the baseline.

**File:** `experiments/train_causal_pessimistic.py`

**Training loop structure (STRICT — must match baseline where possible):**

```text
1. collect_episodes()          — IDENTICAL to baseline (same seed, env, episodes, etc.)
2. Build twin critics          — IDENTICAL to baseline
3. Critic warmup loop          — IDENTICAL to baseline (same number of steps)
4. Joint training loop:
   a. Sample batch             — IDENTICAL to baseline (same _build_batch)
   b. Train critics            — IDENTICAL to baseline (same loss, same SGD)
   c. Compute critic_scores    — IDENTICAL to baseline (min of twin critics)
   d. Sample neg_goals         — NEW: sample N negative goals from buffer
   e. Compute pessimistic Q    — NEW: call actor.loss_and_grads with pessimistic Q
   f. Update actor             — NEW: using pessimistic loss
5. Save checkpoint             — same format, method tag = "causal_pessimistic_min_neg"
```

**New hyperparameters (add to `configs/run_config.py`):**

```python
# Causal pessimistic training
TRAIN_PESSIMISM_MODE = "min_neg"    # {"min_neg", "constant"}
TRAIN_CONSTANT_M = 10.0             # only used when mode = "constant"
TRAIN_NEG_GOALS_N = 16              # number of negative goals for min-neg
```

**Negative goal sampling (STRICT):**

- Sample `N` states uniformly from the replay buffer (same buffer used for critic training).
- These are **goal states** for the pessimistic lower bound, NOT the in-batch negatives
  used by the critic.
- Sampling happens **per training step** (fresh negatives each step).

**Passing phi_fn and psi_fn to the actor:**

```python
# Freeze critic embeddings — no gradient flow
def phi_fn(s, a_batch):
    """phi(s, a) from critic1 (or min of both)."""
    h1, _ = critic1._embed_sa(s, a_batch)
    h2, _ = critic2._embed_sa(s, a_batch)
    # Use critic1 embeddings (both critics share the same architecture;
    # the min-Q is already applied at the score level, not embedding level)
    return h1

def psi_fn(goals):
    """psi(s_g) from critic1."""
    h1, _ = critic1._embed_g(goals)
    return h1
```

**Required logging (exact prefixes):**

```text
[CausalPessimistic] env=<string>  pessimism_mode=<string>  λ=<float>
[CausalPessimistic] episodes=<int>  steps=<int>  warmup=<int>
[CausalPessimistic] neg_goals_N=<int>  constant_M=<float>
[CausalPessimistic] twin_critics=True  loss=inbatch_sigmoid_bce
[CausalPessimistic] state_dim=<int>  n_actions=<int>  total_transitions=<int>  batch=<int>  γ=<float>
[Step <int>] critic_loss=<float>  margin=<float>  actor_loss=<float>  bc_acc=<float>
```

**Checkpoint format:**

Same `.npz` structure as baseline, with additional fields:

- `method` = `"causal_pessimistic_min_neg"` or `"causal_pessimistic_constant_M"`
- `pessimism_mode` = `"min_neg"` or `"constant"`
- `constant_M` = float (only meaningful for constant mode)
- `neg_goals_N` = int

**Verification command:**

```bash
python experiments/train_causal_pessimistic.py --num-steps 500 --num-episodes 5
```

**Success criteria (STRICT):**

- Training completes without error or `NaN`.
- Checkpoint file is saved.
- All required log lines appear.
- Actor loss is finite throughout training.

---

### Phase 9.4 — Integration with `run.py`

**Task:** Add `train_causal_pessimistic` and `eval_causal_pessimistic` tasks to
`experiments/run.py` and `configs/run_config.py`.

**Changes to `configs/run_config.py`:**

```python
# Add these lines:
TRAIN_PESSIMISM_MODE = "min_neg"
TRAIN_CONSTANT_M = 10.0
TRAIN_NEG_GOALS_N = 16

EVAL_CAUSAL_PESSIMISTIC_CHECKPOINT = "checkpoints/causal_pessimistic_min_neg_seed0_<env_tag>.npz"
EVAL_CAUSAL_PESSIMISTIC_OUTPUT = "results/actor_causal_pessimistic_forced_u.csv"
```

**Changes to `experiments/run.py`:**

Add two new task options:

- `"train_causal_pessimistic"` — calls `train()` from `train_causal_pessimistic.py`
- `"eval_causal_pessimistic"` — calls `evaluate_checkpoint()` with the causal pessimistic
  checkpoint

**Required logging:**

```text
[run] 结果已写入: <path>
```

**Verification command:**

```bash
# In run.py, set TASK = "train_causal_pessimistic"
python experiments/run.py

# Then set TASK = "eval_causal_pessimistic"
python experiments/run.py
```

**Success criteria (STRICT):**

- Both tasks complete without error.
- Evaluation CSV is written with the same fields as baseline.

---

### Phase 9.5 — Fair Comparison Evaluation

**Task:** Run forced-U evaluation on the causal pessimistic checkpoint and compare with
baseline results.

**Evaluation protocol (STRICT — inherited from `DEV_SPEC.md` §7 Phase 8):**

- Same `eval_env_id` as baseline: `CausalContrastive-WindyCorridor-15x15-Lethal-v0`
- Same `episodes_per_regime` as baseline: 50
- Same `temperature` as baseline: 1.0
- Same evaluation script: `experiments/evaluate_actor_forced_u.py`

**Required output CSV fields (same as baseline):**

```text
checkpoint, eval_env_id, success_u0, success_u1, mean_success, worst_case_success, regime_gap
```

**Comparison table format (for README / report):**

| Method | success_u0 | success_u1 | mean_success | worst_case_success | regime_gap |
|--------|------------|------------|--------------|-------------------|------------|
| actor_critic_oracle | (from existing CSV) | | | | |
| actor_critic_confounded | (from existing CSV) | | | | |
| causal_pessimistic_min_neg | (new) | | | | |
| causal_pessimistic_constant_M | (ablation, optional) | | | | |

**Interpretation contract (STRICT):**

- If `worst_case_success` improves AND `regime_gap` decreases → evidence for Claim C
  (causal pessimism improves decisions under confounding).
- If `worst_case_success` does not improve → the pessimistic mechanism is too aggressive
  or too weak. Adjust `neg_goals_N` or `constant_M` and re-run.
- Do NOT claim success based on `mean_success` alone — `worst_case_success` and
  `regime_gap` are the primary metrics.

**Verification command:**

```bash
python experiments/run.py
# with TASK = "eval_causal_pessimistic"
```

**Success criteria (STRICT):**

- Evaluation CSV is produced.
- Results are comparable to baseline (same env, same episodes, same evaluation logic).
- No method-specific branches in evaluation code.

---

### Phase 9.6 — Ablation: Constant Penalty Mode

**Task:** Re-train with `pessimism_mode="constant"` and evaluate.

**Verification command:**

```bash
python experiments/train_causal_pessimistic.py --pessimism-mode constant --constant-M 10.0
python experiments/run.py
# with TASK = "eval_causal_pessimistic" (pointing to constant-mode checkpoint)
```

**Success criteria (STRICT):**

- Training and evaluation complete.
- Results added to comparison table.

---

## 4. File Change Summary

### New files

| File | Purpose |
|------|---------|
| `agents/causal_pessimistic_actor.py` | `CausalPessimisticActorNumpy` class |
| `experiments/train_causal_pessimistic.py` | Training script |
| `experiments/smoke_causal_pessimistic.py` | Unit test for pessimistic Q |

### Modified files

| File | Change |
|------|--------|
| `configs/run_config.py` | Add pessimism hyperparameters + checkpoint/output paths |
| `experiments/run.py` | Add `train_causal_pessimistic` and `eval_causal_pessimistic` tasks |

### Unchanged files (MUST NOT be modified)

| File | Reason |
|------|--------|
| `agents/contrastive_critic_numpy.py` | Critic stays objective |
| `agents/goal_conditioned_actor_numpy.py` | Baseline actor preserved |
| `envs/*` | Environment unchanged |
| `utils/offline_data.py` | Data collection unchanged |
| `utils/preprocess.py` | State representation unchanged |
| `experiments/evaluate_actor_forced_u.py` | Evaluation unchanged |
| `experiments/train_actor_critic.py` | Baseline training unchanged |
| `configs/training_defaults.py` | Global defaults unchanged |

---

## 5. Hyperparameter Defaults

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `pessimism_mode` | `"min_neg"` | Primary method; data-driven pessimism |
| `constant_M` | `10.0` | Large enough to make counterfactual actions unattractive |
| `neg_goals_N` | `16` | Matches `ROBUST_V1_M`; enough for stable min estimate |
| `lam` | `0.5` | Same as baseline; BC regularization prevents collapse |
| `critic_warmup` | `2000` | Same as baseline; critic must converge before actor uses it |
| All other params | Same as baseline | Fair comparison requires identical settings |

---

## 6. Gradient Flow Diagram

```text
Offline Data ──→ Critic (twin) ──→ phi(s,a), psi(s_g)  [FROZEN for actor]
                      │
                      ▼
              critic_scores(s, a_i, s_g)  ──→  Pessimistic Q  ──→  Actor Loss
                                                     │                  │
              neg_goals ──→ psi(neg) ──────────────→──┘                  │
                           [FROZEN]                                      ▼
                                                                   Actor MLP
                                                                   (gradients
                                                                    flow here
                                                                    ONLY)
```

**Critical:** The only trainable parameters in the actor update step are the actor MLP
weights (`W1, b1, W2, b2`). The critic embeddings are used as **read-only** scoring
functions.

---

## 7. Execution Order (MANDATORY)

```text
Phase 9.0  →  Verify baseline still works
Phase 9.1  →  Implement CausalPessimisticActorNumpy
Phase 9.2  →  Unit test pessimistic Q computation
Phase 9.3  →  Implement training script
Phase 9.4  →  Integrate with run.py
Phase 9.5  →  Fair comparison evaluation
Phase 9.6  →  Ablation (constant penalty mode)
```

Each phase MUST pass its verification command before proceeding to the next.

---

## 8. Forbidden

- Modifying the critic loss or architecture.
- Modifying the environment.
- Modifying the evaluation script.
- Modifying the baseline actor or training script.
- Using different offline data for the causal pessimistic method vs baseline.
- Using different evaluation settings for the causal pessimistic method vs baseline.
- Claiming improvement based on `mean_success` alone without reporting `worst_case_success`
  and `regime_gap`.
- Introducing PyTorch, CNNs, transformers, or any architecture beyond the existing MLP.
- Skipping phases or merging phases.
- Silent exception handling.

---

## 9. Debug Contract (MANDATORY)

Every new module MUST:

- Print input/output shapes at module boundaries.
- Print at least one concrete sample value (e.g., pessimistic Q for first batch element).
- Never silently return `None` where an array is expected.
- Log the number of counterfactual actions penalized per batch (diagnostic).

---

## 10. Document Maintenance

- **Canonical file:** `CAUSAL_PESSIMISM_SPEC.md`. Do not fork competing specs.
- When adding sub-phases (e.g., 9.7 for seed sweep), **append** with new numbers.
- If a log string must change, update here and in the script in the **same** commit.
- After Phase 9.5 evaluation, update `README.md` with the comparison table.
