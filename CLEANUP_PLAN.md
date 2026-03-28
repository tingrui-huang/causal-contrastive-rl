# Code Cleanup Plan

> **WARNING FOR CODE GENERATION (VIBE CODING):**
> This document is the strict execution plan for reorganizing the repository before
> building new environments. Do NOT delete files without following this plan. Do NOT
> rename modules that are actively imported. Execute steps in order. Verify after each
> step.

---

## 1. Purpose

The repository has accumulated experiment scripts, config files, and result CSVs from
Phases 0–8 of `DEV_SPEC.md`. Before starting the wind-confounder extension
(`ENV_DEV_SPEC.md`), we clean up so that:

- Active code is clearly separated from archived code.
- The working tree is small and navigable.
- No accidental breakage of the still-needed pipeline.

---

## 2. Guiding Principles

| Principle | Rule |
|-----------|------|
| **Move, don't delete** | Old code goes to `oldversion/`, not the trash. We may need to reference it. |
| **One step at a time** | Each step below is one commit (or one logical change). Verify before next step. |
| **Import safety** | Before moving any `.py` file, grep for all imports of that module. If anything active imports it, do NOT move it. |
| **Results are cheap** | CSV files in `results/` can always be regenerated. Archive them freely. |
| **Docs stay** | `DEV_SPEC.md`, `ENV_DEV_SPEC.md`, `README.md` stay in root. |

---

## 3. What Stays (Active Code)

These files are **actively used** by the current pipeline and MUST NOT be moved:

### Core modules

| File | Role | Why it stays |
|------|------|-------------|
| `envs/__init__.py` | Registers Gymnasium env ids on import | Every experiment imports `envs` |
| `envs/hidden_regime_fork.py` | The environment itself | Core of Phase 7+ |
| `agents/contrastive_critic_numpy.py` | NumPy critic (primary backend on Windows) | Used by baseline + robust training |
| `buffers/replay_buffer.py` | Replay buffer | Used by all training scripts |
| `configs/training_defaults.py` | Single source of truth for seeds, env ids, hyperparams | Used by all training + sweep scripts |
| `utils/collector.py` | Episode rollout | Used by pipeline + offline data |
| `utils/preprocess.py` | `extract_state` | Used everywhere |
| `utils/contrastive_sampling.py` | Contrastive tuple builder | Used by all training scripts |
| `utils/offline_data.py` | Multi-episode dataset builder + oracle policy | Used by robust-v1 + baseline training |

### Active experiment scripts

| File | Role | Why it stays |
|------|------|-------------|
| `experiments/run_pipeline.py` | Phase 4 integration test | Still the primary pipeline smoke test |
| `experiments/run_random.py` | Phase 0 sanity check | Minimal, still useful |
| `experiments/smoke_contrastive.py` | Phase 5 smoke test | Still useful for tuple verification |
| `experiments/train_contrastive_baseline.py` | Phase 6 baseline training | Needed for fair comparison |
| `experiments/train_contrastive_robust_v1.py` | Phase 8 robust training | Current research method |
| `experiments/inspect_hidden_fork.py` | Env diagnostics (obs aliasing scan) | Useful for new env variants |

### Active config

| File | Role | Why it stays |
|------|------|-------------|
| `configs/training_defaults.py` | All training hyperparams | Single source of truth |

### Documents

| File | Stays |
|------|-------|
| `DEV_SPEC.md` | YES |
| `ENV_DEV_SPEC.md` | YES |
| `README.md` | YES |
| `requirements.txt` | YES |
| `.gitignore` | YES |

---

## 4. What Moves to `oldversion/`

### 4.1 Stale / superseded experiment scripts

These scripts were useful during earlier phases but are now superseded by the robust-v1
pipeline or the full-batch runner. They can be regenerated or adapted if needed later.

| File | Reason to archive |
|------|-------------------|
| `experiments/run_hidden_fork_seed_sweep.py` | Softmax-baseline-only sweep; superseded by robust-v1 sweep and full batch |
| `experiments/run_hidden_fork_seed_sweep_robust_v1.py` | Standalone robust sweep; superseded by `run_forced_u_full_batch.py` |
| `experiments/evaluate_hidden_fork_forced_u.py` | Forced-U evaluator; folded into `run_forced_u_full_batch.py` |
| `experiments/run_forced_u_full_batch.py` | Overnight batch runner for branch-wall map; will be replaced by wind-env experiments |
| `experiments/probe_hidden_u_robust_v1.py` | Linear probe diagnostic; Phase 8 specific, not needed for new env work |
| `experiments/heatmap_hidden_fork_actions_robust_v1.py` | Heatmap diagnostic; Phase 8 specific |

### 4.2 Stale config

| File | Reason to archive |
|------|-------------------|
| `configs/pipeline_defaults.py` | Only sets `DEFAULT_PIPELINE_ENV_ID`; `training_defaults.py` already has `TRAIN_ENV_ID`. Redundant. |

### 4.3 Stale top-level files

| File | Reason to archive |
|------|-------------------|
| `main.py` | Prints one line ("scaffold is ready"). No real function. |

### 4.4 PyTorch critic (conditional)

| File | Reason to archive |
|------|-------------------|
| `agents/contrastive_critic.py` | PyTorch version. If you are exclusively using the NumPy backend (Windows), this is dead code. **If you plan to use PyTorch later, keep it.** |

### 4.5 Result CSVs

| Path | Action |
|------|--------|
| `results/OldResult/` | Already archived. Leave as-is inside `oldversion/results/OldResult/`. |
| `results/*.csv` (top-level) | Move all to `oldversion/results/`. |

---

## 5. Execution Steps (STRICT ORDER)

### Step 0 — Verify current state

```bash
python experiments/run_pipeline.py --env CausalContrastive-HiddenForkHiddenTrap-15x15-v0
```

Must pass all Phase 4 log lines. If it fails, fix first — do NOT clean up a broken repo.

---

### Step 1 — Create `oldversion/` structure

```bash
mkdir oldversion
mkdir oldversion\experiments
mkdir oldversion\configs
mkdir oldversion\results
```

---

### Step 2 — Move stale experiment scripts

```bash
git mv experiments/run_hidden_fork_seed_sweep.py oldversion/experiments/
git mv experiments/run_hidden_fork_seed_sweep_robust_v1.py oldversion/experiments/
git mv experiments/evaluate_hidden_fork_forced_u.py oldversion/experiments/
git mv experiments/run_forced_u_full_batch.py oldversion/experiments/
git mv experiments/probe_hidden_u_robust_v1.py oldversion/experiments/
git mv experiments/heatmap_hidden_fork_actions_robust_v1.py oldversion/experiments/
```

---

### Step 3 — Move stale config

```bash
git mv configs/pipeline_defaults.py oldversion/configs/
```

---

### Step 4 — Move stale top-level

```bash
git mv main.py oldversion/
```

---

### Step 5 — Move PyTorch critic (OPTIONAL — skip if you want to keep it)

```bash
git mv agents/contrastive_critic.py oldversion/agents/
```

Only do this if you are certain you will not use PyTorch in the near term.

---

### Step 6 — Move result CSVs

```bash
git mv results oldversion/results
```

(This moves the entire `results/` directory including `OldResult/`.)

---

### Step 7 — Verify nothing is broken

```bash
python experiments/run_pipeline.py --env CausalContrastive-HiddenForkHiddenTrap-15x15-v0
python experiments/run_random.py
python experiments/smoke_contrastive.py
python experiments/train_contrastive_baseline.py
python experiments/train_contrastive_robust_v1.py
```

All must run without import errors. If any fail, check whether the moved file was still
imported by an active script and move it back.

---

### Step 8 — Update README.md

Remove or mark as archived any sections that reference moved scripts (e.g. the
`run_hidden_fork_seed_sweep.py` usage, the `run_forced_u_full_batch.py` section). Add a
note:

```markdown
## Archived code

Previous Phase 8 experiment scripts, diagnostics, and result CSVs have been moved to
`oldversion/`. They are preserved for reference but are not part of the active pipeline.
```

---

### Step 9 — Update DEV_SPEC.md (minimal)

In the Phase 8 section, add a note that the scripts referenced there have been archived to
`oldversion/experiments/` and are no longer the active entrypoints. Do NOT delete the
Phase 8 specification text — it documents the research history.

---

### Step 10 — Commit

```bash
git add -A
git commit -m "cleanup: archive Phase 8 experiment scripts and stale files to oldversion/"
```

---

## 6. Post-Cleanup File Tree (Expected)

```text
causal-contrastive-rl/
├── agents/
│   ├── contrastive_critic.py          # (kept if PyTorch planned; else in oldversion/)
│   └── contrastive_critic_numpy.py
├── buffers/
│   └── replay_buffer.py
├── configs/
│   └── training_defaults.py
├── envs/
│   ├── __init__.py
│   └── hidden_regime_fork.py
├── experiments/
│   ├── inspect_hidden_fork.py
│   ├── run_pipeline.py
│   ├── run_random.py
│   ├── smoke_contrastive.py
│   ├── train_contrastive_baseline.py
│   └── train_contrastive_robust_v1.py
├── oldversion/
│   ├── agents/
│   │   └── contrastive_critic.py      # (only if moved in Step 5)
│   ├── configs/
│   │   └── pipeline_defaults.py
│   ├── experiments/
│   │   ├── evaluate_hidden_fork_forced_u.py
│   │   ├── heatmap_hidden_fork_actions_robust_v1.py
│   │   ├── probe_hidden_u_robust_v1.py
│   │   ├── run_forced_u_full_batch.py
│   │   ├── run_hidden_fork_seed_sweep.py
│   │   └── run_hidden_fork_seed_sweep_robust_v1.py
│   ├── results/
│   │   ├── OldResult/
│   │   │   └── (33 archived CSVs)
│   │   └── (any top-level CSVs)
│   └── main.py
├── utils/
│   ├── collector.py
│   ├── contrastive_sampling.py
│   ├── offline_data.py
│   └── preprocess.py
├── DEV_SPEC.md
├── ENV_DEV_SPEC.md
├── CLEANUP_PLAN.md                    # this file (can be moved to oldversion/ after cleanup)
├── README.md
└── requirements.txt
```

---

## 7. Import Dependency Check (Reference)

Before moving any file, verify it is not imported by active code. The known dependency
graph for **active** files is:

```text
experiments/run_pipeline.py
  → envs (side-effect registration)
  → configs.pipeline_defaults        ← THIS IS BEING ARCHIVED (Step 3)
  → configs.training_defaults
  → buffers.replay_buffer
  → utils.collector
  → utils.preprocess

experiments/train_contrastive_baseline.py
  → envs
  → agents.contrastive_critic        ← PyTorch (optional)
  → agents.contrastive_critic_numpy
  → buffers.replay_buffer
  → configs.training_defaults
  → utils.offline_data
  → utils.contrastive_sampling

experiments/train_contrastive_robust_v1.py
  → envs
  → agents.contrastive_critic_numpy
  → buffers.replay_buffer
  → configs.training_defaults
  → utils.offline_data
  → utils.contrastive_sampling
```

**Action required for Step 3:** `experiments/run_pipeline.py` line 30 imports
`from configs.pipeline_defaults import DEFAULT_PIPELINE_ENV_ID`. Before (or immediately
after) moving `pipeline_defaults.py`, you MUST patch `run_pipeline.py` to read the default
env id from `configs/training_defaults.py` (`TRAIN_ENV_ID`) instead. Concretely:

```python
# OLD (line 30):
from configs.pipeline_defaults import DEFAULT_PIPELINE_ENV_ID

# NEW:
from configs.training_defaults import TRAIN_ENV_ID as DEFAULT_PIPELINE_ENV_ID
```

This is the **only known cross-dependency** between active and archived code. Fix it in
the same commit as Step 3.

---

## 8. Forbidden During Cleanup

- Deleting any file permanently (use `git mv` to `oldversion/`).
- Renaming active modules (e.g. renaming `contrastive_critic_numpy.py`).
- Changing any logic in active files (cleanup is structural only).
- Moving `DEV_SPEC.md` or `ENV_DEV_SPEC.md`.
- Moving `envs/`, `buffers/`, or `utils/` modules.

---

## 9. After Cleanup

Once the commit from Step 10 is done and all Step 7 verification passes:

1. This file (`CLEANUP_PLAN.md`) can be moved to `oldversion/` or kept in root as a
   record.
2. Proceed to `ENV_DEV_SPEC.md` Phase W0 (baseline snapshot).
3. New wind-environment code goes into `envs/hidden_regime_fork.py` (extension) or a new
   file under `envs/`.
4. New smoke scripts go into `experiments/smoke_env_wind.py`.
