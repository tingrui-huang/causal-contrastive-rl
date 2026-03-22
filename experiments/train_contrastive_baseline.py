"""
Phase 6 — train minimal contrastive critic (non-causal baseline).

Uses **PyTorch** when ``import torch`` succeeds; otherwise falls back to a **NumPy**
implementation (no GPU) so Phase 6 runs even if the PyTorch install is broken
(e.g. ``torch_cuda.dll`` / WinError 127 on Windows).

Verification: ``python experiments/train_contrastive_baseline.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except OSError:
    torch = None  # type: ignore

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from buffers.replay_buffer import ReplayBuffer
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


def collect_episode(seed: int = 0):
    """Roll out one episode, fill replay buffer; return trajectory + dims."""
    rng = np.random.default_rng(seed)
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    n_actions = int(env.action_space.n)
    policy = build_random_policy(env)
    trajectory = rollout_episode(env, policy, max_steps=500, seed=seed)

    buffer = ReplayBuffer(capacity=10000)
    for trans in trajectory:
        buffer.add(processed_transition(trans))

    T = len(trajectory)
    k = DEFAULT_K
    if T < k + 1:
        raise RuntimeError(
            f"Need trajectory length >= k+1={k+1}, got T={T}. Increase max_steps or change seed."
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


def train_torch() -> None:
    torch.manual_seed(0)
    device = _select_torch_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    print(f"Using backend: PyTorch  device={device}")

    trajectory, buffer, state_dim, n_actions, T, k, rng = collect_episode(0)

    model = ContrastiveCritic(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=128,
        emb_dim=64,
        tau=0.07,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = min(32, T - k)
    num_steps = 50

    model.train()
    for step in range(1, num_steps + 1):
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

        print(f"[Train] step = {step}")
        print(f"[Train] loss = {loss_val}")

    ckpt_path = ROOT / "checkpoints" / "contrastive_baseline.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "n_actions": n_actions,
            "train_device": str(device),
            "backend": "torch",
        },
        ckpt_path,
    )


def train_numpy() -> None:
    print(
        "Using backend: NumPy — PyTorch could not be imported "
        "(e.g. torch_cuda.dll / WinError 127). Training still runs without fixing PyTorch."
    )

    trajectory, buffer, state_dim, n_actions, T, k, rng = collect_episode(0)

    model = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=128,
        emb_dim=64,
        tau=0.07,
        seed=0,
    )

    batch_size = min(32, T - k)
    num_steps = 50
    lr = 1e-2

    for step in range(1, num_steps + 1):
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

        model.apply_sgd(grads, lr=lr)

        print(f"[Train] step = {step}")
        print(f"[Train] loss = {loss}")

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
    )


def main() -> None:
    if _TORCH_AVAILABLE:
        train_torch()
    else:
        train_numpy()


if __name__ == "__main__":
    main()
