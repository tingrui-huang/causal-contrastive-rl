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
    TRAIN_ENV_ID,
    TRAIN_HIDDEN,
    TRAIN_MAX_EPISODE_STEPS,
    TRAIN_NUMPY_LR,
    TRAIN_NUM_STEPS,
    TRAIN_REPLAY_CAPACITY,
    TRAIN_SEED,
    TRAIN_TAU,
    TRAIN_TORCH_LR,
    build_train_config,
    format_train_config_compact,
    format_train_config_lines,
)
from utils.collector import rollout_episode
from utils.contrastive_sampling import DEFAULT_K, build_contrastive_batch
from utils.preprocess import extract_state

if _TORCH_AVAILABLE:
    from agents.contrastive_critic import ContrastiveCritic

def build_random_policy(env):
    def policy(_obs):
        return env.action_space.sample()

    return policy


def processed_transition(trans: dict) -> dict:
    return {
        "state": extract_state(trans["obs"]),
        "next_state": extract_state(trans["next_obs"]),
        "action": trans["action"],
        "reward": trans["reward"],
        "done": trans["done"],
    }


def collect_episode(seed: int, env_id: str):
    """Roll out one episode, fill replay buffer; return trajectory + dims."""
    rng = np.random.default_rng(seed)
    env = gym.make(env_id, render_mode="rgb_array")
    env.action_space.seed(seed)
    n_actions = int(env.action_space.n)
    policy = build_random_policy(env)
    trajectory = rollout_episode(
        env, policy, max_steps=TRAIN_MAX_EPISODE_STEPS, seed=seed
    )

    buffer = ReplayBuffer(capacity=TRAIN_REPLAY_CAPACITY, seed=seed)
    for trans in trajectory:
        buffer.add(processed_transition(trans))

    T = len(trajectory)
    k = DEFAULT_K
    if T < k + 1:
        raise RuntimeError(
            f"Need trajectory length >= k+1={k+1}, got T={T}. "
            f"Increase TRAIN_MAX_EPISODE_STEPS in configs/training_defaults.py or change TRAIN_SEED / TRAIN_ENV_ID."
        )

    state_dim = int(extract_state(trajectory[0]["obs"]).shape[0])
    return trajectory, buffer, state_dim, n_actions, T, k, rng


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


def train_torch(
    *,
    seed: int | None = None,
    env_id: str | None = None,
    verbose: bool = True,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    seed = TRAIN_SEED if seed is None else seed
    env_id = TRAIN_ENV_ID if env_id is None else env_id

    torch.manual_seed(seed)
    device = _select_torch_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if verbose:
        print(f"Using backend: PyTorch  device={device}")

    trajectory, buffer, state_dim, n_actions, T, k, rng = collect_episode(seed, env_id)

    batch_size = min(32, T - k)
    cfg = build_train_config(
        backend="torch",
        device=str(device),
        batch_size=batch_size,
        k=k,
        lr=TRAIN_TORCH_LR,
        seed=seed,
        env_id=env_id,
    )
    if verbose:
        _print_config_header(format_train_config_lines(cfg))

    model = ContrastiveCritic(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=TRAIN_HIDDEN,
        emb_dim=TRAIN_EMB_DIM,
        tau=TRAIN_TAU,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_TORCH_LR)

    losses: list[float] = []
    pos_logits: list[float] = []
    neg_logits: list[float] = []
    margins: list[float] = []

    model.train()
    for step in range(1, TRAIN_NUM_STEPS + 1):
        batch = build_contrastive_batch(
            trajectory,
            buffer,
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
        trajectory_length=T,
        losses=losses,
        pos_logits=pos_logits,
        neg_logits=neg_logits,
        margins=margins,
        backend="torch",
    )

    if save_checkpoint:
        ckpt_path = ROOT / "checkpoints" / "contrastive_baseline.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "state_dim": state_dim,
                "n_actions": n_actions,
                "train_device": str(device),
                "backend": "torch",
                "train_config": cfg,
                "train_config_json": json.dumps(cfg, default=str),
            },
            ckpt_path,
        )
        if verbose:
            print()
            print("[Checkpoint] saved:", ckpt_path)
            _print_config_header(format_train_config_lines(cfg))

    return out


def train_numpy(
    *,
    seed: int | None = None,
    env_id: str | None = None,
    verbose: bool = True,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    seed = TRAIN_SEED if seed is None else seed
    env_id = TRAIN_ENV_ID if env_id is None else env_id

    if verbose:
        print(
            "Using backend: NumPy — PyTorch could not be imported "
            "(e.g. torch_cuda.dll / WinError 127). Training still runs without fixing PyTorch."
        )

    trajectory, buffer, state_dim, n_actions, T, k, rng = collect_episode(seed, env_id)

    batch_size = min(32, T - k)
    cfg = build_train_config(
        backend="numpy",
        device="cpu",
        batch_size=batch_size,
        k=k,
        lr=TRAIN_NUMPY_LR,
        seed=seed,
        env_id=env_id,
    )
    if verbose:
        _print_config_header(format_train_config_lines(cfg))

    model = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=TRAIN_HIDDEN,
        emb_dim=TRAIN_EMB_DIM,
        tau=TRAIN_TAU,
        seed=seed,
    )

    losses: list[float] = []
    pos_logits: list[float] = []
    neg_logits: list[float] = []
    margins: list[float] = []

    for step in range(1, TRAIN_NUM_STEPS + 1):
        batch = build_contrastive_batch(
            trajectory,
            buffer,
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
        trajectory_length=T,
        losses=losses,
        pos_logits=pos_logits,
        neg_logits=neg_logits,
        margins=margins,
        backend="numpy",
    )

    if save_checkpoint:
        ckpt_path = ROOT / "checkpoints" / "contrastive_baseline.npz"
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
            train_config_json=np.array(json.dumps(cfg, default=str)),
        )
        if verbose:
            print()
            print("[Checkpoint] saved:", ckpt_path)
            _print_config_header(format_train_config_lines(cfg))

    return out


def main() -> None:
    if _TORCH_AVAILABLE:
        train_torch()
    else:
        train_numpy()


if __name__ == "__main__":
    main()
