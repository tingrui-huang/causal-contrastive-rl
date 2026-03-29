"""
Phase 6 — train minimal contrastive critic (non-causal baseline).

Uses **PyTorch** when ``import torch`` succeeds; otherwise falls back to a **NumPy**
implementation (no GPU) so Phase 6 runs even if the PyTorch install is broken
(e.g. ``torch_cuda.dll`` / WinError 127 on Windows).

**Seeds and env:** set ``TRAIN_SEED`` / ``TRAIN_ENV_ID`` in ``configs/training_defaults.py`` only.

Programmatic runs (e.g. batch sweeps) can pass ``seed`` / ``env_id`` to ``train_torch`` /
``train_numpy`` — they default to the config file.

Verification: ``python experiments/train_contrastive_baseline.py``
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

import envs  # noqa: F401 — register HiddenFork ids when TRAIN_ENV_ID uses them

_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except OSError:
    torch = None  # type: ignore

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from buffers.replay_buffer import ReplayBuffer
from configs.training_defaults import (
    TRAIN_EMB_DIM,
    TRAIN_COLLECTOR_MODE,
    TRAIN_ENV_ID,
    TRAIN_HIDDEN,
    TRAIN_MAX_EPISODE_STEPS,
    TRAIN_NUM_EPISODES,
    TRAIN_NUMPY_LR,
    TRAIN_NUM_STEPS,
    TRAIN_ORACLE_EPSILON,
    TRAIN_SEED,
    TRAIN_TAU,
    TRAIN_TORCH_LR,
    build_train_config,
    format_train_config_compact,
    format_train_config_lines,
)
from utils.offline_data import collect_episodes

if _TORCH_AVAILABLE:
    from agents.contrastive_critic import ContrastiveCritic


RESULT_FIELDS = [
    "seed",
    "env_id",
    "has_hidden_confounder",
    "trajectory_length",
    "contrastive_k",
    "batch_size",
    "num_train_steps",
    "backend",
    "last_loss",
    "last_mean_pos_logit",
    "last_mean_neg_logit",
    "last_mean_pos_minus_neg_logit",
    "mean_loss",
    "mean_mean_pos_logit",
    "mean_mean_neg_logit",
    "mean_mean_pos_minus_neg_logit",
    "checkpoint_path",
]

DEFAULT_LOSS_FAMILY = "sigmoid_bce_weight1"


def _method_name(loss_family: str) -> str:
    if loss_family == "sigmoid_bce_weight1":
        return "sigmoid_baseline"
    if loss_family == "softmax_ce":
        return "softmax_baseline"
    raise ValueError(f"Unknown loss_family: {loss_family!r}")


def _build_multi_episode_batch(
    *,
    episode_states: list[np.ndarray],
    episode_actions: list[np.ndarray],
    valid_anchors: list[tuple[int, int]],
    buffer: ReplayBuffer,
    state_dim: int,
    k: int,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    anchor_ids = rng.integers(low=0, high=len(valid_anchors), size=batch_size)
    anchor_pairs = [valid_anchors[int(i)] for i in anchor_ids]
    s_batch = np.stack(
        [episode_states[ep_idx][t] for ep_idx, t in anchor_pairs], axis=0
    ).astype(np.float32, copy=False)
    a_batch = np.array(
        [episode_actions[ep_idx][t] for ep_idx, t in anchor_pairs], dtype=np.int64
    )
    s_pos_batch = np.stack(
        [episode_states[ep_idx][t + k] for ep_idx, t in anchor_pairs], axis=0
    ).astype(np.float32, copy=False)

    s_neg_batch = np.empty((batch_size, state_dim), dtype=np.float32)
    for i in range(batch_size):
        neg_item = buffer.sample(1)[0]
        s_neg_batch[i] = np.asarray(neg_item["state"], dtype=np.float32)

    return {
        "s": s_batch,
        "a": a_batch,
        "s_pos": s_pos_batch,
        "s_neg": s_neg_batch,
    }


def _select_torch_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        _ = torch.empty(1, device="cuda")
    except (RuntimeError, AssertionError):
        return torch.device("cpu")
    return torch.device("cuda")


def _print_config_header(lines: list[str]) -> None:
    print()
    for line in lines:
        print(line)
    print()


def _summarize_run(
    *,
    cfg: dict[str, Any],
    trajectory_length: int,
    losses: list[float],
    pos_logits: list[float],
    neg_logits: list[float],
    margins: list[float],
    backend: str,
) -> dict[str, Any]:
    n = len(losses)
    return {
        "seed": cfg["seed"],
        "env_id": cfg["env_id"],
        "has_hidden_confounder": cfg["has_hidden_confounder"],
        "trajectory_length": trajectory_length,
        "contrastive_k": cfg["contrastive_k"],
        "batch_size": cfg["batch_size"],
        "num_train_steps": n,
        "backend": backend,
        "last_loss": losses[-1],
        "last_mean_pos_logit": pos_logits[-1],
        "last_mean_neg_logit": neg_logits[-1],
        "last_mean_pos_minus_neg_logit": margins[-1],
        "mean_loss": float(np.mean(losses)),
        "mean_mean_pos_logit": float(np.mean(pos_logits)),
        "mean_mean_neg_logit": float(np.mean(neg_logits)),
        "mean_mean_pos_minus_neg_logit": float(np.mean(margins)),
        "train_config": cfg,
    }


def _result_row(r: dict[str, Any]) -> dict[str, Any]:
    return {field: r.get(field) for field in RESULT_FIELDS}


def _default_results_path(*, seed: int, env_id: str, backend: str) -> Path:
    env_tag = env_id.replace("/", "_").replace("\\", "_").replace(":", "").replace("-", "_")
    return ROOT / "results" / f"softmax_baseline_seed{seed}_{env_tag}_{backend}.csv"


def _write_result_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def train_torch(
    *,
    seed: int | None = None,
    env_id: str | None = None,
    num_episodes: int | None = None,
    num_steps: int | None = None,
    collector_mode: str = TRAIN_COLLECTOR_MODE,
    oracle_epsilon: float = TRAIN_ORACLE_EPSILON,
    loss_family: str = DEFAULT_LOSS_FAMILY,
    verbose: bool = True,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    seed = TRAIN_SEED if seed is None else seed
    env_id = TRAIN_ENV_ID if env_id is None else env_id
    num_episodes = TRAIN_NUM_EPISODES if num_episodes is None else num_episodes
    num_steps = TRAIN_NUM_STEPS if num_steps is None else num_steps

    torch.manual_seed(seed)
    device = _select_torch_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f"Using backend: PyTorch  device={device}")

    (
        trajectories,
        episode_states,
        episode_actions,
        valid_anchors,
        buffer,
        state_dim,
        n_actions,
        total_steps,
        k,
        rng,
    ) = collect_episodes(
        seed=seed,
        env_id=env_id,
        num_episodes=num_episodes,
        positive_window=0,
        collector_mode=collector_mode,
        oracle_epsilon=oracle_epsilon,
        max_episode_steps=TRAIN_MAX_EPISODE_STEPS,
    )

    batch_size = min(32, len(valid_anchors))
    cfg = build_train_config(
        backend="torch",
        device=str(device),
        batch_size=batch_size,
        k=k,
        lr=TRAIN_TORCH_LR,
        seed=seed,
        env_id=env_id,
        num_episodes=num_episodes,
    )
    cfg["collector_mode"] = collector_mode
    cfg["oracle_epsilon"] = oracle_epsilon if collector_mode == "oracle_eps" else None
    cfg["num_train_steps"] = num_steps
    cfg["method"] = _method_name(loss_family)
    cfg["loss_family"] = loss_family
    if verbose:
        _print_config_header(format_train_config_lines(cfg))

    model = ContrastiveCritic(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=TRAIN_HIDDEN,
        emb_dim=TRAIN_EMB_DIM,
        tau=TRAIN_TAU,
        loss_family=loss_family,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_TORCH_LR)

    losses: list[float] = []
    pos_logits: list[float] = []
    neg_logits: list[float] = []
    margins: list[float] = []

    model.train()
    for step in range(1, num_steps + 1):
        batch = _build_multi_episode_batch(
            episode_states=episode_states,
            episode_actions=episode_actions,
            valid_anchors=valid_anchors,
            buffer=buffer,
            state_dim=state_dim,
            k=k,
            batch_size=batch_size,
            rng=rng,
        )
        s = torch.from_numpy(batch["s"]).float().to(device)
        a = torch.from_numpy(batch["a"]).long().to(device)
        s_pos = torch.from_numpy(batch["s_pos"]).float().to(device)
        s_neg = torch.from_numpy(batch["s_neg"]).float().to(device)

        opt.zero_grad()
        loss = model(s, a, s_pos, s_neg)
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu().item())
        if not np.isfinite(loss_val):
            raise RuntimeError(f"Non-finite loss at step {step}")

        with torch.no_grad():
            pl, nl = model.logits(s, a, s_pos, s_neg)
            m_pos = float(pl.mean().cpu())
            m_neg = float(nl.mean().cpu())
            m_margin = float((pl - nl).mean().cpu())

        losses.append(loss_val)
        pos_logits.append(m_pos)
        neg_logits.append(m_neg)
        margins.append(m_margin)

        if verbose:
            print(format_train_config_compact(cfg))
            print(f"[Train] step = {step}")
            print(f"[Train] loss = {loss_val}")
            print(f"[Train] mean_pos_logit = {m_pos}")
            print(f"[Train] mean_neg_logit = {m_neg}")
            print(f"[Train] mean_pos_minus_neg_logit = {m_margin}")

    out = _summarize_run(
        cfg=cfg,
        trajectory_length=total_steps,
        losses=losses,
        pos_logits=pos_logits,
        neg_logits=neg_logits,
        margins=margins,
        backend="torch",
    )

    if save_checkpoint:
        env_tag = env_id.replace("/", "_").replace("\\", "_").replace(":", "").replace("-", "_")
        collector_tag = collector_mode.replace("-", "_")
        ckpt_path = (
            ROOT
            / "checkpoints"
            / f"{cfg['method']}_seed{seed}_{env_tag}_{collector_tag}.pt"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "state_dim": state_dim,
                "n_actions": n_actions,
                "train_device": str(device),
                "backend": "torch",
                "method": cfg["method"],
                "loss_family": loss_family,
                "train_config": cfg,
                "train_config_json": json.dumps(cfg, default=str),
            },
            ckpt_path,
        )
        out["checkpoint_path"] = str(ckpt_path)
        if verbose:
            print()
            print("[Checkpoint] saved:", ckpt_path)
            _print_config_header(format_train_config_lines(cfg))

    return out


def train_numpy(
    *,
    seed: int | None = None,
    env_id: str | None = None,
    num_episodes: int | None = None,
    num_steps: int | None = None,
    collector_mode: str = TRAIN_COLLECTOR_MODE,
    oracle_epsilon: float = TRAIN_ORACLE_EPSILON,
    loss_family: str = DEFAULT_LOSS_FAMILY,
    verbose: bool = True,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    seed = TRAIN_SEED if seed is None else seed
    env_id = TRAIN_ENV_ID if env_id is None else env_id
    num_episodes = TRAIN_NUM_EPISODES if num_episodes is None else num_episodes
    num_steps = TRAIN_NUM_STEPS if num_steps is None else num_steps

    if verbose:
        print(
            "Using backend: NumPy — PyTorch could not be imported "
            "(e.g. torch_cuda.dll / WinError 127). Training still runs without fixing PyTorch."
        )

    (
        trajectories,
        episode_states,
        episode_actions,
        valid_anchors,
        buffer,
        state_dim,
        n_actions,
        total_steps,
        k,
        rng,
    ) = collect_episodes(
        seed=seed,
        env_id=env_id,
        num_episodes=num_episodes,
        positive_window=0,
        collector_mode=collector_mode,
        oracle_epsilon=oracle_epsilon,
        max_episode_steps=TRAIN_MAX_EPISODE_STEPS,
    )

    batch_size = min(32, len(valid_anchors))
    cfg = build_train_config(
        backend="numpy",
        device="cpu",
        batch_size=batch_size,
        k=k,
        lr=TRAIN_NUMPY_LR,
        seed=seed,
        env_id=env_id,
        num_episodes=num_episodes,
    )
    cfg["collector_mode"] = collector_mode
    cfg["oracle_epsilon"] = oracle_epsilon if collector_mode == "oracle_eps" else None
    cfg["num_train_steps"] = num_steps
    cfg["method"] = _method_name(loss_family)
    cfg["loss_family"] = loss_family
    if verbose:
        _print_config_header(format_train_config_lines(cfg))

    model = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=TRAIN_HIDDEN,
        emb_dim=TRAIN_EMB_DIM,
        tau=TRAIN_TAU,
        seed=seed,
        loss_family=loss_family,
    )

    losses: list[float] = []
    pos_logits: list[float] = []
    neg_logits: list[float] = []
    margins: list[float] = []

    for step in range(1, num_steps + 1):
        batch = _build_multi_episode_batch(
            episode_states=episode_states,
            episode_actions=episode_actions,
            valid_anchors=valid_anchors,
            buffer=buffer,
            state_dim=state_dim,
            k=k,
            batch_size=batch_size,
            rng=rng,
        )
        loss, grads = model.loss_and_grads(
            batch["s"],
            batch["a"],
            batch["s_pos"],
            batch["s_neg"],
        )
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")

        model.apply_sgd(grads, lr=TRAIN_NUMPY_LR)

        pl, nl = model.logits(
            batch["s"],
            batch["a"],
            batch["s_pos"],
            batch["s_neg"],
        )
        m_pos = float(np.mean(pl))
        m_neg = float(np.mean(nl))
        m_margin = float(np.mean(pl - nl))

        losses.append(float(loss))
        pos_logits.append(m_pos)
        neg_logits.append(m_neg)
        margins.append(m_margin)

        if verbose:
            print(format_train_config_compact(cfg))
            print(f"[Train] step = {step}")
            print(f"[Train] loss = {loss}")
            print(f"[Train] mean_pos_logit = {m_pos}")
            print(f"[Train] mean_neg_logit = {m_neg}")
            print(f"[Train] mean_pos_minus_neg_logit = {m_margin}")

    out = _summarize_run(
        cfg=cfg,
        trajectory_length=total_steps,
        losses=losses,
        pos_logits=pos_logits,
        neg_logits=neg_logits,
        margins=margins,
        backend="numpy",
    )

    if save_checkpoint:
        env_tag = env_id.replace("/", "_").replace("\\", "_").replace(":", "").replace("-", "_")
        collector_tag = collector_mode.replace("-", "_")
        ckpt_path = (
            ROOT
            / "checkpoints"
            / f"{cfg['method']}_seed{seed}_{env_tag}_{collector_tag}.npz"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        sd = model.state_dict()
        np.savez(
            ckpt_path,
            W1=sd["W1"],
            b1=sd["b1"],
            W2=sd["W2"],
            b2=sd["b2"],
            state_dim=sd["state_dim"],
            n_actions=sd["n_actions"],
            tau=sd["tau"],
            method=np.array(cfg["method"]),
            loss_family=np.array(loss_family),
            train_config_json=np.array(json.dumps(cfg, default=str)),
        )
        out["checkpoint_path"] = str(ckpt_path)
        if verbose:
            print()
            print("[Checkpoint] saved:", ckpt_path)
            _print_config_header(format_train_config_lines(cfg))

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Train the contrastive baseline.")
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument("--env-id", type=str, default=TRAIN_ENV_ID)
    p.add_argument("--num-episodes", type=int, default=TRAIN_NUM_EPISODES)
    p.add_argument("--num-steps", type=int, default=TRAIN_NUM_STEPS)
    p.add_argument(
        "--collector-mode",
        type=str,
        default=TRAIN_COLLECTOR_MODE,
        choices=["random", "oracle_eps"],
    )
    p.add_argument("--oracle-epsilon", type=float, default=TRAIN_ORACLE_EPSILON)
    p.add_argument(
        "--loss-family",
        type=str,
        default=DEFAULT_LOSS_FAMILY,
        choices=["sigmoid_bce_weight1", "softmax_ce"],
    )
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "torch", "numpy"],
        help="Choose backend explicitly or auto-select",
    )
    p.add_argument(
        "--no-save-checkpoint",
        action="store_true",
        help="Skip writing a checkpoint file",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for saving the single-run result summary",
    )
    p.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print summary CSV to stdout only; do not write a result file",
    )
    p.add_argument("--quiet", action="store_true", help="Disable per-step printing")
    args = p.parse_args()

    use_torch = _TORCH_AVAILABLE if args.backend == "auto" else args.backend == "torch"
    if use_torch and not _TORCH_AVAILABLE:
        raise RuntimeError("Torch backend requested but torch import is unavailable.")

    train = train_torch if use_torch else train_numpy
    result = train(
        seed=args.seed,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        collector_mode=args.collector_mode,
        oracle_epsilon=args.oracle_epsilon,
        loss_family=args.loss_family,
        verbose=not args.quiet,
        save_checkpoint=not args.no_save_checkpoint,
    )
    row = _result_row(result)
    output_path = args.output or _default_results_path(
        seed=args.seed,
        env_id=args.env_id,
        backend=result["backend"],
    )
    if not args.stdout_only:
        _write_result_csv(output_path, row)
    writer = csv.DictWriter(sys.stdout, fieldnames=RESULT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)


if __name__ == "__main__":
    main()
