# Execution Protocol & Development Specification


> **This document is an execution protocol, not a research explanation.**  
> Use `Contrastive319.pdf` only to scope **later** causal / pessimistic work (see §5).

**Repository:** `causal-contrastive-rl`  
**Companion note:** See `Contrastive319.pdf` for research motivation (contrastive RL under unobserved confounding, pessimistic targets, propensity-style weighting).

This document defines **how** we build and **how** we know each step is correct. It is the single source of truth for phased development, logging contracts, and review expectations.

---

## 1. Purpose

- **Goal:** A **minimal, controllable** codebase to validate three claims in order:
  1. Contrastive RL works in simple environments.
  2. Contrastive RL degrades under **confounding** (spurious positive futures).
  3. A **causal / pessimistic** modification improves behavior under confounding.
- **Non-goals:** SOTA performance, large models, or premature optimization.

---

## 2. Coding Principles (Process)

These rules make progress **inspectable** (logs, shapes, one clear change per phase).

| Principle | What it means |
|-----------|----------------|
| **Intent before code** | Each phase has one hypothesis and one deliverable file (or a named set). |
| **Phases are linear** | Complete phases in order: **0 → 1 → 2 → …**. No skipping, no “future” features. |
| **Verifiable by logs** | Success = **required log lines** + **strict criteria**. No “it seems fine.” |
| **Small surface area** | Prefer one new module or one script per phase when possible. |
| **Debuggability > elegance** | Print shapes, one sample value, and boundaries (empty buffer, done flag). |
| **Reproducibility** | Fixed seeds where randomness matters; configs live under `configs/` when introduced. |

**Definition of “done” for a phase:** All **Success Criteria (STRICT)** and **Required Logging** match. If logs do not match → the phase is **not** complete.

---

## 3. Development Philosophy (Content)

- Do **not** optimize for speed or scale first.
- Do **not** introduce complex architectures (see **Forbidden** below).
- Do **not** skip validation steps.

We prioritize:

- **Interpretability** — can we explain what each tensor is?
- **Debuggability** — if it fails, which module failed?
- **Reproducibility** — same command → same log signatures.

---

## 4. Project Structure (Strict)

```text
causal-contrastive-rl/
├── envs/              # environments (MiniGrid; confounding variants later)
├── agents/            # contrastive RL / causal variants (later phases)
├── buffers/           # replay and contrastive sampling
├── experiments/       # runnable scripts (one job, one purpose)
├── configs/           # hyperparameters and experiment presets (when needed)
├── utils/             # collector, preprocessing, shared helpers
├── main.py
├── requirements.txt
├── README.md          # project overview + pointer to this spec
├── DEV_SPEC.md        # this file (canonical execution protocol)
└── Contrastive319.pdf # research motivation (scope only; not early implementation)
```

---

## 5. Research Alignment (from `Contrastive319.pdf`)

Use this only to **scope** later phases—not to implement early.

- **Problem:** Under confounding, observational transitions need not match interventional ones:  
  \(p(s' \mid s, a) \neq p(s' \mid \mathrm{do}(a), s)\). Positive pairs from “future states” can be **biased**.
- **Contrastive RL:** Learning uses tuples \((s, a, s^+, s^-)\); confounding mainly hurts the **positive** distribution.
- **Direction for mitigation (later):** Pessimism or reweighting of positives (e.g. propensity-style weights \(w(s,a)\)), aligned with a **modified Bellman / contrastive** target—implement only after a working **non-causal baseline** (Phase 6).

---

## 6. Execution Rules (STRICT)

1. Complete phases **in order** (0 → 1 → 2 → …).
2. **Do not** implement later phases early.
3. After each phase, run the **Verification command** for that phase (below).
4. **Verify logs** using the **Logging contract (exact prefix)** in §6.1.

If logs do not match → **FAIL** (phase not complete).

---

## 6.1 Logging Contract (Exact Prefix Match)

**Required logging** means:

- Each required line MUST match the documented string as an **exact prefix** (case-sensitive), from the first character through the last fixed character before dynamic content.
- **Dynamic segments** (numbers, shapes, floats, short vectors) may vary **after** that prefix.
- **Forbidden:** paraphrasing labels, changing bracket style, changing spacing around `=`, different tag names, lowercasing tags, or omitting punctuation that is part of the prefix.

**Examples (invalid vs valid):**

| Invalid | Why |
|--------|-----|
| `[pipeline] trajectory length = 10` | Wrong case / tag |
| `[Pipeline] traj len: 10` | Paraphrased label |
| `[Pipeline] trajectory length=10` | Missing space before `=` (prefix must match `... length = `) |

| Valid prefix | Dynamic part |
|--------------|----------------|
| `[Pipeline] trajectory length = ` | `10` |

**Shape / sample lines:** Use NumPy-style tuples for shapes, e.g. `(12,)` not `[12]` unless explicitly documented otherwise.

---

## 7. Phased Roadmap

Each phase lists a **Verification command** you MUST run after implementation. Commands assume you run them from the **repository root** with your Python environment activated.

---

### Phase 0 — Environment Sanity Check

**Task:** Run a random agent in MiniGrid.

**File:** `experiments/run_random.py`

**Verification command:**

```bash
python experiments/run_random.py
```

**Expected behavior:**

- Environment resets.
- Step loop runs; rewards observable.
- Episode terminates normally.

**Required logging (exact prefix / line unless noted):**

- `Initial observation keys` — exact prefix (line may continue with keys).
- Each step: lines whose prefix is exactly `[Random] step = ` (dynamic remainder must include `action`, `reward`, and `done` in readable form, e.g. comma-separated fields).
- Exactly: `Episode finished.`
- `Total reward:` — exact prefix (line may continue with the numeric total).

**Success criteria (STRICT):**

- All required prefixes/lines appear in stdout (order may vary only if documented in the script header).
- No uncaught exceptions.

---

### Phase 1 — Trajectory Collection

**Task:** Implement a single-episode rollout that returns a trajectory.

**File:** `utils/collector.py`

**Verification command:**

```bash
python experiments/smoke_collector.py
```

*This smoke script imports `collector`, runs one rollout, and prints the required log. If it is not yet in the repo, add it before claiming Phase 1 complete.*

**Data contract:** `trajectory` is a `list` of `dict`, each transition:

```python
{
    "obs": ...,
    "action": ...,
    "reward": ...,
    "next_obs": ...,
    "done": ...,
}
```

**Required logging:**

```text
[Collector] trajectory length = <int>
```

Prefix must be exactly `[Collector] trajectory length = ` (space before the integer).

**Success criteria (STRICT):**

- `len(trajectory) > 0`.
- `trajectory[0]` contains keys: `obs`, `action`, `reward`, `next_obs`, `done`.
- The required log line is printed exactly once per collected trajectory (unless documented otherwise).

---

### Phase 2 — State Preprocessing

**Task:** Map raw observation → fixed-size vector (or fixed tensor shape).

**File:** `utils/preprocess.py`

**Verification command:**

```bash
python experiments/smoke_preprocess.py
```

*Add this smoke script if missing; it must exercise preprocess on real observations and print required logs.*

**Requirements:**

- Flatten image (or agreed observation fields).
- Append direction (or other minimal state if required by env).
- Output a **NumPy** array (or document if later upgraded to torch).

**Required logging:**

```text
[Preprocess] state shape: <tuple or shape repr>
```

Prefix must be exactly `[Preprocess] state shape: ` (space after colon).

**Success criteria (STRICT):**

- Output is a NumPy array (dtype documented; no silent casts to object).
- **Same shape** for every step in an episode.
- No `NaN` / `inf`.
- Required log line printed at least once per smoke run (document if also printed per step).

---

### Phase 3 — Replay Buffer

**Task:** Store transitions and support uniform sampling.

**File:** `buffers/replay_buffer.py`

**Verification command:**

```bash
python experiments/smoke_buffer.py
```

*Add this smoke script if missing; it fills the buffer and samples a batch.*

**Required API:**

- `add(item)`
- `sample(batch_size)`
- `__len__`

**Required logging (after a fill + sample test):**

```text
[Buffer] size: <int>
[Buffer] sample size: <int>
```

Prefixes must be exactly `[Buffer] size: ` and `[Buffer] sample size: ` (space after each colon).

**Success criteria (STRICT):**

- Size increases after `add` (until capacity if capped—document capacity).
- `sample(n)` returns exactly `n` items when buffer has ≥ `n` elements; behavior when undersized must be **explicit** (error or pad—no silent wrong batch).

---

### Phase 4 — Integrated Pipeline Test

**Task:** Connect env → collector → preprocess → buffer in one script. Catch mistakes where shapes look right but **wrong tensors** are stored (e.g. raw `obs` vs processed state).

**File:** `experiments/run_pipeline.py`

**Verification command:**

```bash
python experiments/run_pipeline.py
```

**Required steps:**

1. Run **one** episode (or fixed step budget—document which).
2. Preprocess states that actually go into the buffer (document whether `obs`, `next_obs`, or both).
3. Push **processed** transitions into the buffer (not raw dicts unless explicitly documented).
4. Sample one batch and assert batch element structure is consistent.

**Required logging:**

```text
[Pipeline] trajectory length = <int>
[Pipeline] processed state shape = <tuple>
[Pipeline] first processed state sample = <five floats or first five elements>
[Pipeline] buffer size = <int>
[Pipeline] sample batch size = <int>
```

Exact prefixes:

- `[Pipeline] trajectory length = `
- `[Pipeline] processed state shape = `
- `[Pipeline] first processed state sample = `
- `[Pipeline] buffer size = `
- `[Pipeline] sample batch size = `

The **first processed state sample** line MUST print **at least the first five numeric values** of the processed state vector (after flattening), so integration errors (wrong channel order, wrong observation branch) are visible without guessing.

**Success criteria (STRICT):**

- End-to-end run completes without error.
- All five log lines appear; shapes and batch sizes are **mutually consistent**.
- Sample values are finite (no `NaN` / `inf`).

**Gate:** Do **not** start Phase 5 until Phase 4 passes.

---

### Phase 5 — Contrastive Tuple Construction

**Gate:** **Do not** implement until Phase 4 is verified.

**Task:** Build training tuples \((s, a, s^+, s^-)\) from buffer + trajectories.

**Suggested file:** `buffers/contrastive_batch.py` or `utils/contrastive_sampling.py` (pick one; document in `README.md`).

**Verification command:**

```bash
python experiments/smoke_contrastive.py
```

**Rules (baseline):**

- **Positive \(s^+\):** future state in the **same** trajectory at offset \(+k\) (fixed \(k\), document default).
- **Negative \(s^-\):** random state from buffer (or random transition’s `next_obs`—document one rule and stick to it).

**Repository defaults (see `README.md`):** \(k=4\) (see `DEFAULT_K` in `utils/contrastive_sampling.py`); negative is the **`state`** field of a uniformly sampled buffer transition; anchor indices \(t\) are drawn **uniformly with replacement** from \(\{0,\ldots,T-1-k\}\) (not full enumeration). Implementation: `utils/contrastive_sampling.py`, smoke: `experiments/smoke_contrastive.py`.

**Required logging (exact lines):**

```text
[Contrastive] positive sampled from t+k
[Contrastive] negative sampled randomly
```

These two lines must appear **verbatim** (no extra words, no punctuation changes).

**Success criteria (STRICT):**

- Batch has valid indices (no out-of-range positives at trajectory tail—document truncation or mask).
- Required log lines appear in the smoke path (e.g. one batch construction).

---

### Phase 6 — Contrastive Critic (Baseline, Non-Causal)

**Gate:** Only after Phase 5 tuples are verified.

**Task:** Implement a **minimal** contrastive critic / encoder training loop **without** causal correction (baseline).

**Suggested layout:**

- `agents/contrastive_critic.py` — model forward + loss.
- `experiments/train_contrastive_baseline.py` — train for a **small** number of steps.

**Verification command:**

```bash
python experiments/train_contrastive_baseline.py
```

**Constraints:**

- Keep architecture minimal (e.g. small MLP on flattened state)—see **Forbidden** below.

**Required logging (minimum, exact prefixes):**

```text
[Train] step = <int>
[Train] loss = <float>
```

Prefixes: `[Train] step = ` and `[Train] loss = ` (spaces as shown).

**Recommended (exact prefixes) — contrastive separation:**

```text
[Train] mean_pos_logit = <float>
[Train] mean_neg_logit = <float>
[Train] mean_pos_minus_neg_logit = <float>
```

Use these to compare **clean vs confounded** (larger margin usually means easier pos/neg separation). Printed each step in `train_contrastive_baseline.py`.

**Configuration (STRICT for training scripts):**

- **Single source of truth:** `configs/training_defaults.py` — set **`TRAIN_SEED`**, **`TRAIN_ENV_ID`**, and other training hyperparameters there (no scattered seed literals in `train_contrastive_baseline.py`).
- **Exceptions** (keep local seeds / demos): `experiments/run_pipeline.py`, `experiments/smoke_contrastive.py`, `experiments/run_random.py`.
- **Logging:** Each training step prints a full **`[Config] ...` one-line block** (all keys) immediately before the `[Train]` lines; a multiline `[Config]` header runs at start; checkpoint save repeats the multiline `[Config]`. Checkpoints store **`train_config`** / **`train_config_json`**.

**Success criteria (STRICT):**

- Loss is finite; no `NaN` over a short run.
- Checkpoint or final weights can be saved/loaded (optional but recommended—document).

**Repository defaults:** `agents/contrastive_critic.py` + `experiments/train_contrastive_baseline.py`; defaults from `configs/training_defaults.py` (e.g. 200 train steps); checkpoint at `checkpoints/contrastive_baseline.pt`. If `import torch` fails (e.g. Windows CUDA DLL errors), the training script falls back to **`agents/contrastive_critic_numpy.py`** and saves **`checkpoints/contrastive_baseline.npz`**. See `README.md`.

---

### Phase 7 — Confounding Environment

**Gate:** Only after Phase 6 baseline runs.

**Task:** Introduce a **controlled** confounding mechanism in `envs/` (e.g. action or transition bias correlated with unobserved noise—document the generative story in `README.md`).

**Verification command:**

```bash
python experiments/run_pipeline.py --env <confounded_env_id>
```

*Use the actual CLI flag / env id implemented in code; document it in `README.md`. If no CLI yet, the verification command MUST be the single documented entrypoint for “pipeline smoke on confounded env”.*

**Requirements:**

- Confounders must be **toggleable** (flag or env id) so “with / without confounding” are comparable.
- Re-run Phase 4 pipeline smoke on the new env before training.

**Success criteria (STRICT):**

- Same logging contracts as Phase 0–4 still hold (including Phase 4’s five-line contract).
- A short experiment script documents **expected degradation** of the Phase 6 baseline under confounding (qualitative log or metric—document).

**Repository defaults:** `envs/hidden_regime_fork.py` — `HiddenRegimeForkEnv` with **goal** at top center, **long center trunk**, and **fork** that opens left vs right depending on hidden **`U`** (sampled in `_gen_grid`; **`U ∉ obs`**, `info["confounder"]` for debug). **`CausalContrastive-HiddenFork-15x15-v0`** (confounded) vs **`CausalContrastive-HiddenFork-15x15-Clean-v0`** (fixed `U=0`). See `README.md`.

---

### Phase 8+ — Causal / Pessimistic Variant (Optional, After Phase 7)

**Gate:** Only after confounding reproduces failure of the baseline.

**Direction (aligned with `Contrastive319.pdf`):** Modify how positives / negatives enter the loss (e.g. pessimistic surrogate pairs, weighted robust objective, documented \(w(s,a)\) proxy). Treat this as a **new** phase with its own success criteria and logs—do not fold into Phase 6 retroactively.

**Current repository implementation (Phase 8 v1):**

- `experiments/train_contrastive_robust_v1.py` — bidirectional robust contrastive objective (NumPy fallback path).
- Defaults live in `configs/training_defaults.py`:
  - `ROBUST_V1_NUM_EPISODES`
  - `ROBUST_V1_W`
  - `ROBUST_V1_LOGIT_SCALE`
  - `ROBUST_V1_P`
  - `ROBUST_V1_M`
  - `ROBUST_V1_COLLECTOR_MODE`
  - `ROBUST_V1_ORACLE_EPSILON`
- `experiments/run_hidden_fork_seed_sweep_robust_v1.py` — sweep seeds / regimes / weights and write CSV.
- `experiments/probe_hidden_u_robust_v1.py` — train one or more robust-v1 models, then run a **pre-divergence** linear probe for hidden `U` on paired `fixed_u=0/1` rollouts using both `action-only` and embedding `h(s,a)` features.
- `experiments/heatmap_hidden_fork_actions_robust_v1.py` — train one or more robust-v1 models, then export an action-score matrix from the last equal-observation anchor toward `U=0` / `U=1` futures; optionally writes a PNG if `matplotlib` is installed.

**Interpretation contract (STRICT):**

- Treat the current robust family as a **BCE / log-sigmoid** family, not as the original Phase 6 softmax-contrastive loss.
- `robust_v1` with `w=1.0` means **no-pessimism control inside the robust-v1 family**; it is **not automatically identical** to the original Phase 6 baseline.
- Until a same-data softmax baseline is implemented, do **not** label `robust_v1, w=1.0` as “the baseline” in tables or docs without an explicit qualifier.
- If both are present, name methods explicitly, e.g.:
  - `softmax_baseline`
  - `robust_v1_w1.0`
  - `robust_v1_w0.7`
  - `robust_v1_w0.5`

**Data-generation contract (STRICT):**

- If `collector_mode="oracle_eps"`, the demonstrator **may** read privileged env internals (e.g. `env.unwrapped.hidden_u`, `agent_pos`, `agent_dir`) **only to choose actions during rollout**.
- Stored replay items / training states **must remain partial-observation only** (e.g. `extract_state(obs)`); **never** append `hidden_u` or other privileged values to the learner input.
- Positive windows (`t+k .. t+k+P`) must stay **within the same episode** as the anchor.
- Multi-episode datasets are preferred over single-episode training for Phase 8 comparisons.
- For BCE / log-sigmoid variants, a documented **logit scale** may be applied before the sigmoid; if changed, record it in config/logs/CSV.

**Fair-comparison contract (STRICT):**

- Any claim of “robust vs baseline” under confounding must hold the **offline dataset recipe** fixed unless the comparison is explicitly labeled as a data+method change.
- For fair comparison runs, align at minimum:
  - `env_id`
  - `seed` list
  - `num_episodes`
  - `collector_mode`
  - `oracle_epsilon`
  - `num_train_steps`
  - replay/buffer seed behavior
- The preferred comparison order is:
  1. same-data `softmax_baseline` vs `robust_v1_w1.0` vs `robust_v1_w<1`
  2. then auxiliary ablations on weights (`w=0.5`, `0.7`, `1.0`; optional `0.9`)
- If a comparison is **not** fair yet (e.g. old random-policy baseline vs oracle multi-episode robust), the result must be labeled as **preliminary / non-fair** in both stdout summary and docs.

**Dataset contract (STRICT, INVALID if violated):**

- For any baseline-vs-robust comparison, the offline dataset MUST be generated once and then reused across methods.
- Dataset generation parameters MUST be identical across compared methods, including at minimum:
  - collector policy / `collector_mode`
  - `num_episodes`
  - seed list
  - exploration parameter such as `oracle_epsilon`
  - replay/buffer sampling seed behavior
- It is FORBIDDEN to regenerate a separate offline dataset per method unless the run is explicitly labeled **non-fair** or **data+method ablation**.
- If one method is trained on a different offline dataset recipe from another and the run is presented as a fair comparison, the experiment is **INVALID**.

**Checkpoint contract (STRICT):**

- All Phase 8 training entrypoints used for downstream evaluation must save a checkpoint by default, unless the caller explicitly disables saving for a sweep or smoke run.
- A checkpoint must store enough information to run evaluation **without retraining**:
  - model weights
  - `state_dim`
  - `n_actions`
  - backend / device family
  - `train_config` or `train_config_json`
  - method identifier (`softmax_baseline` vs `robust_v1`)
  - weight `w` when applicable
- Sweep scripts may disable checkpoint saving to avoid overwriting, but evaluation scripts must consume saved checkpoints rather than retraining silently.

**Evaluation protocol (STRICT, current priority):**

- The primary Phase 8 downstream test is **forced-regime policy evaluation** in `HiddenRegimeForkEnv` using:
  - `fixed_u=0`
  - `fixed_u=1`
- This evaluation is the main evidence for claims about **stronger / safer policy behavior**; training-set logits alone are insufficient.
- For each evaluated checkpoint, run the same policy extraction rule in both regimes and report:
  - `success_u0`
  - `success_u1`
  - `mean_success`
  - `worst_case_success = min(success_u0, success_u1)`
  - `regime_gap = abs(success_u0 - success_u1)`
- If per-episode reward is meaningful, also report:
  - `return_u0`
  - `return_u1`
  - `mean_return`
  - `worst_case_return`
- The minimum recommended weight set for robust-v1 forced-regime evaluation is:
  - `w=1.0`
  - `w=0.7`
  - `w=0.5`
  - optional `w=0.9`
- A Phase 8 report that claims improved safety should prioritize:
  1. `worst_case_success`
  2. `regime_gap`
  3. `mean_success`

**Policy-extraction contract (STRICT):**

- Any Phase 8 evaluation script that derives a control policy from the critic must document the exact scoring rule used to rank actions.
- Until a stronger theory-backed evaluator is added, it is acceptable to use a **heuristic goal-scoring greedy policy** based on critic scores; however, this must **not** be described as a proven optimal `Q` policy unless such an equivalence is explicitly established.
- The scoring rule, action subset, and goal/goal-bank construction must be written to CSV metadata or config output.
- If only a subset of actions is evaluated (e.g. `left/right/forward`), this restriction must be explicit in logs/docs.
- A single policy-extraction function MUST be implemented once and reused across all methods in the same comparison.
- Changing the policy-extraction rule between methods is FORBIDDEN unless the run is explicitly labeled as a policy-extraction ablation.
- If two methods are compared under different action-scoring rules without explicit ablation labeling, the comparison is **INVALID**.

**Evaluation integrity contract (STRICT, INVALID if violated):**

- The same evaluation script MUST be used for all methods in a comparison.
- Evaluation scripts MUST NOT contain method-specific branches that change rollout logic, action ranking, stopping rules, or metric computation.
- Metrics computation MUST be identical across runs for all compared methods.
- Any method-specific preprocessing that affects only evaluation must be made explicit and treated as a separate ablation, not folded into the main comparison.
- If robust and baseline results are produced with different evaluation logic and reported as a single fair comparison, the evaluation is **INVALID**.

**Claim-to-metric mapping (REQUIRED for docs / reports):**

- Claim A — baseline is misled by confounding:
  - compare `softmax_baseline` (or clearly labeled interim control) on clean vs confounded settings
  - inspect degradation, regime sensitivity, or widened forced-regime gap
- Claim B — robust objective reduces misleading signal dependence:
  - use surrogate diagnostics
  - use pre-divergence probe
  - use pre-fork action heatmap
- Claim C — this reduction improves decisions:
  - use forced-regime success / return metrics
  - emphasize `worst_case_success` and `regime_gap`

**Required result tables / CSV fields (Phase 8 evaluation):**

- Every Phase 8 evaluation CSV should include at least:
  - `method`
  - `loss_family`
  - `seed`
  - `env_id`
  - `eval_env_id`
  - `collector_mode`
  - `num_episodes`
  - `num_train_steps`
  - `w` (if applicable)
  - `success_u0`
  - `success_u1`
  - `mean_success`
  - `worst_case_success`
  - `regime_gap`
- If probe / heatmap / policy evaluation are produced from checkpoints, also log the checkpoint path or checkpoint id.

**Required logging (Phase 8 robust v1):**

```text
[Train] step = <int>
[Train] loss = <float>
[Train] mean_pos_logit = <float>
[Train] mean_neg_logit = <float>
[Train] mean_pos_minus_neg_logit = <float>
[Diag] frac_pos_logit_gt_neg_logit = <float>
[Diag] mean_pos_obs_minus_surr_logit = <float>
[Diag] mean_neg_surr_minus_obs_logit = <float>
```

**Required logging (Phase 8 evaluation):**

```text
[EvalConfig] method = <string>
[EvalConfig] loss_family = <string>
[EvalConfig] checkpoint = <path>
[EvalConfig] eval_env_id = <string>
[EvalConfig] action_subset = <string>
[Eval] success_u0 = <float>
[Eval] success_u1 = <float>
[Eval] mean_success = <float>
[Eval] worst_case_success = <float>
[Eval] regime_gap = <float>
```

**Verification commands:**

```bash
python experiments/train_contrastive_robust_v1.py
python experiments/run_hidden_fork_seed_sweep_robust_v1.py
python experiments/probe_hidden_u_robust_v1.py
python experiments/heatmap_hidden_fork_actions_robust_v1.py
```

**Planned execution order (MANDATORY unless explicitly waived):**

1. Update / enforce `DEV_SPEC.md` first.
2. Align baseline training settings with Phase 8 data-generation settings and save checkpoints.
3. Save robust checkpoints for the target weight set.
4. Run forced-regime evaluation (`fixed_u=0`, `fixed_u=1`) and produce the main summary table.
5. Run probe and heatmap as explanatory diagnostics.
6. Update `README.md` only after the evaluation protocol and outputs exist.

---

## 8. Debug Contract (MANDATORY)

Every module that transforms data MUST:

- Print **input/output shapes** (or counts for non-tensor structures).
- Print **at least one concrete sample value** (bounded—e.g. first element or small slice).
- **Never** silently return `None` where a tensor/array is expected.

If something fails:

1. Print intermediate values at module boundaries.
2. Isolate **which phase/module** fails.
3. Do **not** “fix” without a hypothesis tied to logs.

---

## 9. Forbidden (Unless Explicitly Unlocked in a Later Phase Doc)

- Convolutional stacks / heavy CNNs (default pipeline stays tabular or tiny MLP).
- Transformers or large sequence models.
- Large-batch training before baselines work.
- Skipping phases or merging phases without updating **this** document.
- Silent exception handling that hides failure (empty `except:`).

---

## 10. Review Checklist (for PRs / Self-Review)

- [ ] Changes belong to **one phase** (or doc-only / bugfix clearly labeled).
- [ ] **Required logs** for that phase appear in the PR description or sample stdout.
- [ ] New behavior is documented in `README.md` if it affects how to run experiments.
- [ ] Randomness: seeds noted for any new script.
- [ ] No new forbidden patterns introduced.
- [ ] **Verification command** for the phase was run and stdout matches §6.1.

---

## 11. Definition of Progress

You only move forward if:

- Logs match expectations for the current phase (including **exact prefixes**).
- Outputs are **interpretable** (shapes and at least one value trace).
- Behavior is **explainable** in one short paragraph (added to `README.md` or phase notes).

---

## 12. Document Maintenance

- **Canonical file:** `DEV_SPEC.md`. Do not fork competing specs; update this file.
- When you add a phase, **append** it with a new number; do not renumber completed phases without team agreement.
- If a log string must change, update **Required logging** here and in the script in the **same** commit.
