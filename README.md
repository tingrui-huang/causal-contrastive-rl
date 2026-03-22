# causal-contrastive-rl

Minimal experimental framework for **contrastive RL** and (later) **causal / pessimistic** variants under confounding.

## Execution protocol (canonical)

> **This repository’s rules for phased work, log contracts, and verification live in [`DEV_SPEC.md`](DEV_SPEC.md).**  
> That file is the execution protocol—not a research essay. Research motivation is in `Contrastive319.pdf` (scope for later phases only).

- Follow phases **0 → 1 → 2 → …** in order.
- Each phase has a **verification command** and **exact-prefix** log requirements—see `DEV_SPEC.md`.

**Legacy path:** `excuate.md` redirects here so older links keep working.

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
- **Train:** `experiments/train_contrastive_baseline.py` — one episode → buffer → **50** steps with `build_contrastive_batch`, logs **`[Config] ...`** (full run settings: **seed**, **env_id**, inferred **has_hidden_confounder**, **k**, **lr**, etc.) before each `[Train]` block, then `[Train] step =` / `[Train] loss =`, plus **`[Train] mean_pos_logit`**, **`mean_neg_logit`**, **`mean_pos_minus_neg_logit`**. **Change `TRAIN_SEED` and `TRAIN_ENV_ID` only in `configs/training_defaults.py`** (exceptions: `run_pipeline.py`, `smoke_contrastive.py`, `run_random.py`).

```bash
python experiments/train_contrastive_baseline.py
```

**Batch comparison (HiddenFork clean vs confounded, 5 seeds 0–4 → CSV):**

```bash
python experiments/run_hidden_fork_seed_sweep.py
```

Writes `results/hidden_fork_seed_sweep.csv` and prints the same CSV to stdout. Use `--stdout-only` to skip the file; `-o path.csv` to change the path.

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
