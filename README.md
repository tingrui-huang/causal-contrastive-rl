# causal-contrastive-rl

Minimal experimental framework for **contrastive RL** and (later) **causal / pessimistic** variants under confounding.

## Execution protocol (canonical)

> **This repository’s rules for phased work, log contracts, and verification live in [`DEV_SPEC.md`](DEV_SPEC.md).**  
> That file is the execution protocol—not a research essay. Research motivation is in `Contrastive319.pdf` (scope for later phases only).

- Follow phases **0 → 1 → 2 → …** in order.
- Each phase has a **verification command** and **exact-prefix** log requirements—see `DEV_SPEC.md`.

**Legacy path:** `excuate.md` redirects here so older links keep working.

## Archived code

Previous Phase 8 batch scripts, diagnostics, and result CSVs may be moved to
`oldversion/` during cleanup. They are preserved for reference but are not part of the
active pipeline for new environment work.

## Directory layout

```text
causal-contrastive-rl/
├── envs/              # MiniGrid modifications (confounding later)
├── agents/            # contrastive RL / causal variants
├── buffers/           # replay + contrastive sampling
├── experiments/       # runnable scripts (smoke tests, training)
├── configs/           # hyperparameter configs (`training_defaults.py` = seed + env for training)
├── utils/             # collector, preprocessing, helpers
├── main.py
├── requirements.txt
├── README.md
├── DEV_SPEC.md        # execution protocol (read this for Cursor / contributors)
└── Contrastive319.pdf # research notes (reference only)
```

## Quick start

```bash
python experiments/run_random.py
```

(Phase 0 sanity check; see `DEV_SPEC.md` for the full command list per phase.)

## Phase 5 — contrastive tuples (baseline)

- **Code:** `utils/contrastive_sampling.py` (`build_contrastive_batch`), smoke: `experiments/smoke_contrastive.py`.
- **Fixed** \(k = 4\) (`DEFAULT_K` in `utils/contrastive_sampling.py`): positive \(s^+\) is the processed state at index \(t+k\) on the **same** trajectory as anchor \(t\).
- **Negative** \(s^-\): `state` from a uniformly random transition in the replay buffer.
- **Anchors:** minibatch indices \(t\) sampled **uniformly with replacement** from valid anchors \(\{0,\ldots,T-1-k\}\) (not full enumeration).

```bash
python experiments/smoke_contrastive.py
```

## Phase 6 — contrastive critic baseline (PyTorch or NumPy)

- **Torch:** `agents/contrastive_critic.py` — MLP on `concat(state, one_hot(action))`, L2-normalized embeddings, 2-way softmax contrastive loss.
- **NumPy fallback:** `agents/contrastive_critic_numpy.py` — same objective; SGD; used when `import torch` fails (common on Windows with broken CUDA DLLs).
- **Train:** `experiments/train_contrastive_baseline.py` — one episode → buffer → **200** steps with `build_contrastive_batch`, logs **`[Config] ...`** (full run settings: **seed**, **env_id**, inferred **has_hidden_confounder**, **k**, **lr**, etc.) before each `[Train]` block, then `[Train] step =` / `[Train] loss =`, plus **`[Train] mean_pos_logit`**, **`mean_neg_logit`**, **`mean_pos_minus_neg_logit`**. **Change `TRAIN_SEED` and `TRAIN_ENV_ID` only in `configs/training_defaults.py`** (exceptions: `run_pipeline.py`, `smoke_contrastive.py`, `run_random.py`).

```bash
python experiments/train_contrastive_baseline.py
```

**Batch comparison (HiddenFork clean vs confounded, 5 seeds 0–4 → CSV):**

```bash
python experiments/run_hidden_fork_seed_sweep.py
```

Writes `results/hidden_fork_seed_sweep.csv` and prints the same CSV to stdout. Use `--stdout-only` to skip the file; `-o path.csv` to change the path.

## Phase 8 — robust contrastive v1 (NumPy)

- **Train:** `experiments/train_contrastive_robust_v1.py` — bidirectional robust objective with shared **`w=0.9`**, positive window **`P=2`** (`t+k .. t+k+2`), negative candidate pool **`M=16`**, **`logit_scale=5.0`**, and a **multi-episode** dataset (**default `num_episodes=50`**).
- **Diagnostics:** prints usual `[Train]` metrics plus **`[Diag] frac_pos_logit_gt_neg_logit`**, **`[Diag] mean_pos_obs_minus_surr_logit`**, **`[Diag] mean_neg_surr_minus_obs_logit`**.
- **Reproducibility:** seeds now fix `env.reset`, `action_space.sample()`, replay-buffer sampling, and NumPy model init.
- **Collector:** default `collector_mode="oracle_eps"` uses a privileged demonstrator that may read `env.unwrapped.hidden_u` / pose for action selection, but replay items still store only partial-observation states from `obs`.
- **Configuration:** change `ROBUST_V1_NUM_EPISODES`, `ROBUST_V1_W`, `ROBUST_V1_SWEEP_WEIGHTS`, `ROBUST_V1_P`, `ROBUST_V1_M`, `ROBUST_V1_LOGIT_SCALE`, `ROBUST_V1_COLLECTOR_MODE`, `ROBUST_V1_ORACLE_EPSILON` in `configs/training_defaults.py`.
- **Extra diagnostics:** `experiments/probe_hidden_u_robust_v1.py` runs a pre-divergence linear probe for hidden `U` using both `action-only` and learned `h(s,a)` features; `experiments/heatmap_hidden_fork_actions_robust_v1.py` exports per-action scores from the same pre-divergence anchor toward `U=0` / `U=1` futures, and also writes a PNG heatmap if `matplotlib` is available.

```bash
python experiments/train_contrastive_robust_v1.py
```

**Batch comparison (robust v1, HiddenFork clean vs confounded, 5 seeds 0–4 → CSV):**

```bash
python experiments/run_hidden_fork_seed_sweep_robust_v1.py
```

Writes `results/hidden_fork_seed_sweep_robust_v1.csv` and prints the same CSV to stdout.
To tune the shared robust weight, sweep multiple values in one run:

```bash
python experiments/run_hidden_fork_seed_sweep_robust_v1.py --weights 0.5 0.7 0.9
```

The current default sweep in config is fixed at `w=0.9`; use `--weights ...` only when you explicitly want a new ablation.

```bash
python experiments/probe_hidden_u_robust_v1.py
python experiments/heatmap_hidden_fork_actions_robust_v1.py
```

## Forced-U evaluation batch

For the current overnight comparison entrypoint, use:

```bash
python experiments/run_forced_u_full_batch.py
```

This script now defaults to the **merge-shared closed-book** configuration:

- seeds: `0,1`
- robust weights: `0.5, 0.8, 1.0`
- `goal_mode=merge_shared`
- planner: `plan_depth=2`
- `collision_penalty=3.0`
- `turn_penalty=0.05`
- `progress_bonus=0.75`
- `success_bonus=5.0`

It writes four CSV files incrementally while running:

- `results/hidden_fork_confounded_baseline_2seed.csv`
- `results/hidden_fork_confounded_robust_v1_merge_shared_2seed.csv`
- `results/hidden_fork_forced_u_eval_merge_shared_2seed.csv`
- `results/hidden_fork_forced_u_main_table_merge_shared_2seed.csv`

Use this batch for the current closed-book branch-choice check on the existing
branch-wall HiddenFork map.

### Planned next environment: visually symmetric hidden-trap fork

The current `merge_shared` evaluator removes regime-specific future-goal leakage,
but the map still has a simpler cue: one side is visibly blocked and the other is
visibly open. That means a policy may still solve the fork by reading geometry,
not by handling hidden confounding.

The planned stricter next step is therefore a **visually symmetric hidden-trap**
version of `HiddenFork`:

- at the fork, **left** and **right** should both look like ordinary floor in the observation
- the visible one-side wall cue should be removed
- the branch hazard becomes latent:
  - if `U=0`, right is the hidden failure branch
  - if `U=1`, left is the hidden failure branch
- stepping on the unsafe branch should terminate with failure
- the oracle collector still knows `U` and should avoid the trap perfectly
- evaluation should continue using the same **shared merge target** so the test remains closed-book

Expected use:

- current evaluator = **open-book / tracking** sanity check
- weak-planner evaluator = **planner sensitivity** check
- merge-goal evaluator on branch-wall map = **current closed-book branch-choice** check
- symmetric hidden-trap map = **stricter causal robustness** check

This redesign is planned and documented first before implementation so the next
environment change can be interpreted cleanly.

- If PyTorch loads: uses **PyTorch**; device is **`cuda`** when available and a tiny CUDA alloc succeeds, else **`cpu`**. Checkpoint: **`checkpoints/contrastive_baseline.pt`**.
- If `import torch` **raises** (e.g. `torch_cuda.dll` / WinError 127): automatically uses **NumPy** backend (CPU). Checkpoint: **`checkpoints/contrastive_baseline.npz`**.

### PyTorch on Windows (`torch_cuda.dll` / `WinError 127`)

若 **`import torch` 就报错**：这与脚本内「选 CPU/GPU」无关；可先不管 PyTorch，直接跑训练脚本，会走 **NumPy**。

想修好 PyTorch 再用 GPU/CPU 版 torch，可卸载后装 CPU 轮：

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

需要 GPU 时，请从 [pytorch.org](https://pytorch.org) 选与驱动匹配的 CUDA 构建，并安装 **Visual C++ Redistributable**。

## Phase 7 — hidden regime fork (map-level confounding)

Custom MiniGrid: **`envs/hidden_regime_fork.py`** (`HiddenRegimeForkEnv`). At each `reset`, a binary **`U`** is drawn in **`_gen_grid`** (only if `confound=True`); **`U` is not in `obs`**, only in **`info["confounder"]`** for debugging.

- **Goal:** top-center `(cx, 1)`. **Agent:** bottom-center `(cx, size-2)` facing up.
- **Long trunk:** full **center column** `(cx, y)` for `y = 1 … bottom` so the fork is far from the start. **Fork** at `fork_row` (default **4**): **left** detour `(cx−1,f),(cx−1,f−1)` vs **right** `(cx+1,f),(cx+1,f−1)`; **exactly one** side open depending on `U`. With **size ≥ 15**, the fork sits **outside** the 7×7 view for reset + 1–2 `forward` (see `inspect_hidden_fork.py`).
- **Clean baseline:** `confound=False` fixes `U=0` every episode.

Registered ids (import **`envs`** before `gym.make` — `run_pipeline.py` already does):

| Env id | Meaning |
|--------|--------|
| `CausalContrastive-HiddenFork-15x15-v0` | `confound=True`, random `U` |
| `CausalContrastive-HiddenFork-15x15-Clean-v0` | `confound=False`, fixed layout |

**Phase 4 pipeline:**

```bash
python experiments/run_pipeline.py --env CausalContrastive-HiddenFork-15x15-v0
python experiments/run_pipeline.py --env CausalContrastive-HiddenFork-15x15-Clean-v0
```

**Research wording:** interpret as **latent map regime** changing reachable futures (not textbook policy confounding unless you add a behavioral policy). Phase 8+ is for pessimistic / causal fixes.

**Default env without typing CLI:** edit **`configs/pipeline_defaults.py`** → `DEFAULT_PIPELINE_ENV_ID`, then run:

```bash
python experiments/run_pipeline.py
```

`--env` still overrides that default.

**Diagnostics** (initial obs, 1–2 forward, **scan** `k=0..12` for first **U0 vs U1** image divergence — aligns with contrastive `k`; reset seed = **`configs/training_defaults.TRAIN_SEED`**):

```bash
python experiments/inspect_hidden_fork.py
```

If the scan says divergence starts at step `m` but `DEFAULT_K` in `utils/contrastive_sampling.py` is smaller, early positives may still be regime-agnostic; adjust `k` or the map.

`HiddenRegimeForkEnv` accepts **`fixed_u=0` or `fixed_u=1`** via `gym.make(..., fixed_u=0)` for tests (forces regime when `confound=True`). Pipeline prints **`[Pipeline] confounder = ...`** when the env exposes `hidden_u`.
