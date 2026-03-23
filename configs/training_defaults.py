"""
Unified defaults for **training** scripts (contrastive baseline, etc.).

Change **`TRAIN_SEED`** and **`TRAIN_ENV_ID`** here — do not scatter literals.

**Exceptions** (intentionally not wired here; keep local seeds / demos):
`experiments/run_pipeline.py`, `experiments/smoke_contrastive.py`, `experiments/run_random.py`.
"""

from __future__ import annotations

from typing import Any

# --- Single global seed for training (env reset, batch RNG, torch/numpy model init) ---
TRAIN_SEED = 0

# --- Environment ---
# Examples:
# TRAIN_ENV_ID = "MiniGrid-Empty-5x5-v0"
# TRAIN_ENV_ID = "CausalContrastive-HiddenFork-15x15-Clean-v0"
# TRAIN_ENV_ID = "CausalContrastive-HiddenFork-15x15-v0"
TRAIN_ENV_ID = "MiniGrid-Empty-5x5-v0"

TRAIN_MAX_EPISODE_STEPS = 500
TRAIN_NUM_EPISODES = 5
TRAIN_NUM_STEPS = 1000
TRAIN_HIDDEN = 128
TRAIN_EMB_DIM = 64
TRAIN_TAU = 0.07
TRAIN_TORCH_LR = 1e-3
TRAIN_NUMPY_LR = 1e-3
TRAIN_REPLAY_CAPACITY = 10000

# --- Phase 8 robust-v1 defaults ---
ROBUST_V1_NUM_EPISODES = 50
ROBUST_V1_P = 2
ROBUST_V1_M = 16
ROBUST_V1_W = 0.9
ROBUST_V1_SWEEP_WEIGHTS = (0.9,)
ROBUST_V1_COLLECTOR_MODE = "oracle_eps"  # {"random", "oracle_eps"}
ROBUST_V1_ORACLE_EPSILON = 0.2
ROBUST_V1_LOGIT_SCALE = 5.0


def infer_hidden_confounder(env_id: str) -> bool | None:
    """
    Whether the env uses an **unobserved** confounder in the HiddenFork sense.

    Returns:
        ``True`` / ``False`` when known from naming; ``None`` if unknown custom id.
    """
    if "CausalContrastive-HiddenFork" in env_id:
        return "Clean" not in env_id
    if env_id.startswith("MiniGrid-") or env_id.startswith("BabyAI-"):
        return False
    return None


def build_train_config(
    *,
    backend: str,
    device: str | None = None,
    batch_size: int | None = None,
    k: int | None = None,
    lr: float | None = None,
    seed: int | None = None,
    env_id: str | None = None,
    num_episodes: int | None = None,
) -> dict[str, Any]:
    """Flat dict for logging and checkpointing."""
    from utils.contrastive_sampling import DEFAULT_K

    k_val = DEFAULT_K if k is None else k
    seed_val = TRAIN_SEED if seed is None else seed
    env_val = TRAIN_ENV_ID if env_id is None else env_id
    num_episodes_val = TRAIN_NUM_EPISODES if num_episodes is None else num_episodes
    conf = infer_hidden_confounder(env_val)
    if lr is None:
        lr = TRAIN_NUMPY_LR if backend == "numpy" else TRAIN_TORCH_LR
    return {
        "seed": seed_val,
        "env_id": env_val,
        "has_hidden_confounder": conf,
        "has_hidden_confounder_note": _confounder_note(conf, env_val),
        "contrastive_k": k_val,
        "max_episode_steps": TRAIN_MAX_EPISODE_STEPS,
        "num_episodes": num_episodes_val,
        "num_train_steps": TRAIN_NUM_STEPS,
        "hidden": TRAIN_HIDDEN,
        "emb_dim": TRAIN_EMB_DIM,
        "tau": TRAIN_TAU,
        "lr": lr,
        "replay_capacity": TRAIN_REPLAY_CAPACITY,
        "backend": backend,
        "device": device,
        "batch_size": batch_size,
    }


def _confounder_note(conf: bool | None, env_id: str) -> str:
    if conf is True:
        return "hidden U present (not in obs); confound=True fork"
    if conf is False:
        return "no hidden confounder (clean MiniGrid or HiddenFork-Clean)"
    return f"unknown env id for confounder inference: {env_id!r}"


def format_train_config_compact(cfg: dict[str, Any]) -> str:
    """Single-line ``[Config]`` for repeating on every metric line (full key set)."""
    parts = [f"{k}={cfg[k]!r}" for k in sorted(cfg.keys())]
    return "[Config] " + " ".join(parts)


def format_train_config_lines(cfg: dict[str, Any]) -> list[str]:
    """Lines with prefix ``[Config] `` for stdout."""
    # Stable key order for readability
    keys = [
        "seed",
        "env_id",
        "has_hidden_confounder",
        "has_hidden_confounder_note",
        "contrastive_k",
        "max_episode_steps",
        "num_episodes",
        "num_train_steps",
        "batch_size",
        "hidden",
        "emb_dim",
        "tau",
        "lr",
        "backend",
        "device",
        "replay_capacity",
    ]
    lines = []
    for key in keys:
        if key not in cfg:
            continue
        lines.append(f"[Config] {key} = {cfg[key]!r}")
    # Any extra keys (e.g. future flags)
    for key in sorted(cfg.keys()):
        if key in keys:
            continue
        lines.append(f"[Config] {key} = {cfg[key]!r}")
    return lines
