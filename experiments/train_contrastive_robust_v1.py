"""
Phase 8 v1 (minimal) — Bidirectional robust contrastive objective (NumPy).

Goal: handle confounding where both observed positives and sampled negatives
may be unreliable. We therefore use:

- single critic
- shared constant weight w = 0.5
- pessimism on both sides via surrogates:
  - robust positive surrogate: worst candidate in a future window of size P
    (P=2 => candidates at t+k, t+k+1, t+k+2; choose argmin score)
  - robust negative surrogate: hardest candidate in a random negative pool of size M
    (M=16; choose argmax score)

This script is designed to be runnable even when PyTorch cannot be imported
(e.g. broken torch CUDA DLL on Windows). It uses ContrastiveCriticNumpy only.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs  # noqa: F401  # register custom HiddenFork ids  # noqa: E402

from agents.contrastive_critic_numpy import (  # noqa: E402
    ContrastiveCriticNumpy,
    _embed_backward,
)
from buffers.replay_buffer import ReplayBuffer  # noqa: E402
from configs.training_defaults import (  # noqa: E402
    ROBUST_V1_M,
    ROBUST_V1_P,
    ROBUST_V1_W,
    TRAIN_EMB_DIM,
    TRAIN_ENV_ID,
    TRAIN_HIDDEN,
    TRAIN_MAX_EPISODE_STEPS,
    TRAIN_NUM_EPISODES,
    TRAIN_NUMPY_LR,
    TRAIN_NUM_STEPS,
    TRAIN_REPLAY_CAPACITY,
    TRAIN_SEED,
    TRAIN_TAU,
    build_train_config,
    format_train_config_compact,
    format_train_config_lines,
)
from utils.collector import rollout_episode  # noqa: E402
from utils.contrastive_sampling import DEFAULT_K  # noqa: E402
from utils.preprocess import extract_state  # noqa: E402


def _build_random_policy(env: gym.Env):
    def policy(_obs):
        return env.action_space.sample()

    return policy


def _processed_transition(trans: dict) -> dict:
    return {
        "state": extract_state(trans["obs"]),
        "next_state": extract_state(trans["next_obs"]),
        "action": trans["action"],
        "reward": trans["reward"],
        "done": trans["done"],
    }


def collect_episodes(seed: int, env_id: str, num_episodes: int, P: int):
    """Roll out multiple episodes; positives stay within an episode, negatives use global buffer."""
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity=TRAIN_REPLAY_CAPACITY, seed=seed)
    k = DEFAULT_K
    trajectories: list[list[dict]] = []
    episode_states: list[np.ndarray] = []
    episode_actions: list[np.ndarray] = []
    valid_anchors: list[tuple[int, int]] = []
    n_actions: int | None = None
    state_dim: int | None = None

    for ep_idx in range(num_episodes):
        ep_seed = seed + ep_idx
        env = gym.make(env_id, render_mode="rgb_array")
        env.action_space.seed(ep_seed)
        if n_actions is None:
            n_actions = int(env.action_space.n)
        policy = _build_random_policy(env)
        trajectory = rollout_episode(
            env, policy, max_steps=TRAIN_MAX_EPISODE_STEPS, seed=ep_seed
        )
        trajectories.append(trajectory)

        for trans in trajectory:
            buffer.add(_processed_transition(trans))

        states = np.stack([extract_state(tr["obs"]) for tr in trajectory], axis=0)
        actions = np.array([tr["action"] for tr in trajectory], dtype=np.int64)
        episode_states.append(states)
        episode_actions.append(actions)

        if state_dim is None:
            state_dim = int(states.shape[1])

        T = len(trajectory)
        num_valid = T - k - P
        for t in range(max(0, num_valid)):
            valid_anchors.append((ep_idx, t))

    if not valid_anchors:
        raise RuntimeError(
            f"No valid anchors across {num_episodes} episode(s). Need some episode with "
            f"T >= k+P+1={k+P+1}. Change TRAIN_SEED/ENV or TRAIN_MAX_EPISODE_STEPS."
        )

    total_steps = int(sum(len(traj) for traj in trajectories))
    return (
        trajectories,
        episode_states,
        episode_actions,
        valid_anchors,
        buffer,
        int(state_dim),
        int(n_actions),
        total_steps,
        k,
        rng,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """log(sigmoid(x)) with stable numpy."""
    # log(sigmoid(x)) = -softplus(-x) = -logaddexp(0, -x)
    return -np.logaddexp(0.0, -x)


def _log_one_minus_sigmoid(x: np.ndarray) -> np.ndarray:
    """log(1-sigmoid(x)) with stable numpy."""
    # log(1-sigmoid(x)) = -softplus(x) = -logaddexp(0, x)
    return -np.logaddexp(0.0, x)


def train_numpy_robust_v1(
    *,
    seed: int,
    env_id: str,
    num_episodes: int,
    num_steps: int,
    lr: float,
    P: int,
    M: int,
    w: float,
    verbose: bool = True,
) -> dict[str, Any]:
    # ---- collect data (multi-episode dataset) ----
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
        seed=seed, env_id=env_id, num_episodes=num_episodes, P=P
    )

    model = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=TRAIN_HIDDEN,
        emb_dim=TRAIN_EMB_DIM,
        tau=TRAIN_TAU,
        seed=seed,
    )

    batch_size = min(32, len(valid_anchors))
    cfg = build_train_config(
        backend="numpy",
        device="cpu",
        batch_size=batch_size,
        k=k,
        lr=lr,
        seed=seed,
        env_id=env_id,
        num_episodes=num_episodes,
    )
    if verbose:
        print()
        for line in format_train_config_lines(cfg):
            print(line)
        print()

    losses: list[float] = []
    pos_means: list[float] = []
    neg_means: list[float] = []
    margins: list[float] = []
    frac_pos_gt_neg_vals: list[float] = []
    pos_obs_minus_surr_vals: list[float] = []
    neg_surr_minus_obs_vals: list[float] = []

    w_pos = float(w)
    w_neg = float(w)

    for step in range(1, num_steps + 1):
        # ---- sample anchors ----
        anchor_ids = rng.integers(low=0, high=len(valid_anchors), size=batch_size)
        anchor_pairs = [valid_anchors[int(i)] for i in anchor_ids]
        s_batch = np.stack(
            [episode_states[ep_idx][t] for ep_idx, t in anchor_pairs], axis=0
        ).astype(np.float64, copy=False)
        a_batch = np.array(
            [episode_actions[ep_idx][t] for ep_idx, t in anchor_pairs], dtype=np.int64
        )

        # observed positive
        s_pos_obs = np.stack(
            [episode_states[ep_idx][t + k] for ep_idx, t in anchor_pairs], axis=0
        ).astype(np.float64, copy=False)

        # positive surrogate window candidates
        pos_cand_states = np.stack(
            [
                np.stack(
                    [episode_states[ep_idx][t + k + p] for ep_idx, t in anchor_pairs],
                    axis=0,
                ).astype(np.float64, copy=False)
                for p in range(P + 1)
            ],
            axis=1,
        )  # (B, P+1, state_dim)

        # observed negative: random from replay buffer
        s_neg_obs = np.empty((batch_size, state_dim), dtype=np.float64)
        for i in range(batch_size):
            neg_item = buffer.sample(1)[0]
            s_neg_obs[i] = np.asarray(neg_item["state"], dtype=np.float64)

        # negative candidate pool: random from replay buffer
        neg_pool_items = buffer.sample(M)
        neg_pool_states = np.stack(
            [np.asarray(it["state"], dtype=np.float64) for it in neg_pool_items],
            axis=0,
        )  # (M, state_dim)

        # ---- embeddings for anchors/observed ----
        ha, cache_ha = model._embed(s_batch, a_batch)  # (B, emb)
        hpos_obs, cache_pos_obs = model._embed(s_pos_obs, a_batch)
        hneg_obs, cache_neg_obs = model._embed(s_neg_obs, a_batch)

        # For the BCE-style robust objective, keep scores in their natural
        # cosine-like range instead of amplifying them by 1/tau.
        f_pos_obs = np.sum(ha * hpos_obs, axis=1)  # (B,)
        f_neg_obs = np.sum(ha * hneg_obs, axis=1)  # (B,)

        # ---- positive surrogate: worst over window ----
        pos_cand_h: list[np.ndarray] = []
        pos_cand_cache: list[Any] = []
        f_pos_cand = np.empty((batch_size, P + 1), dtype=np.float64)
        for p in range(P + 1):
            hp, cache_p = model._embed(pos_cand_states[:, p, :], a_batch)
            pos_cand_h.append(hp)
            pos_cand_cache.append(cache_p)
            f_pos_cand[:, p] = np.sum(ha * hp, axis=1)

        idx_pos_surr = np.argmin(f_pos_cand, axis=1)  # (B,)
        f_pos_surr = f_pos_cand[np.arange(batch_size), idx_pos_surr]

        # ---- negative surrogate: hardest over pool ----
        neg_cand_h: list[np.ndarray] = []
        neg_cand_cache: list[Any] = []
        f_neg_cand = np.empty((batch_size, M), dtype=np.float64)
        for j in range(M):
            s_neg_cand = np.repeat(neg_pool_states[j][None, :], batch_size, axis=0)
            hn, cache_j = model._embed(s_neg_cand, a_batch)
            neg_cand_h.append(hn)
            neg_cand_cache.append(cache_j)
            f_neg_cand[:, j] = np.sum(ha * hn, axis=1)

        idx_neg_surr = np.argmax(f_neg_cand, axis=1)  # (B,)
        f_neg_surr = f_neg_cand[np.arange(batch_size), idx_neg_surr]

        # ---- robust objective (pdf / updated323) ----
        # pos: log sigma(f)
        # neg: log (1-sigma(f))
        Lpos = -(
            w_pos * _log_sigmoid(f_pos_obs) + (1.0 - w_pos) * _log_sigmoid(f_pos_surr)
        )  # (B,)
        Lneg = -(
            w_neg * _log_one_minus_sigmoid(f_neg_obs)
            + (1.0 - w_neg) * _log_one_minus_sigmoid(f_neg_surr)
        )  # (B,)
        loss = float(np.mean(Lpos + Lneg))
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite robust loss at step {step}: {loss}")

        # ---- compute analytic gradients w.r.t logits f ----
        # derivative for -log sigma(x):  d/dx = -sigmoid(-x)
        # derivative for -log(1-sigmoid(x)): d/dx = sigmoid(x)
        batch_size_f = float(batch_size)
        dL_df_pos_obs = -w_pos * _sigmoid(-f_pos_obs) / batch_size_f
        dL_df_pos_surr = -(1.0 - w_pos) * _sigmoid(-f_pos_surr) / batch_size_f
        dL_df_neg_obs = w_neg * _sigmoid(f_neg_obs) / batch_size_f
        dL_df_neg_surr = (1.0 - w_neg) * _sigmoid(f_neg_surr) / batch_size_f

        # ---- backprop to embeddings ----
        dh_ha = np.zeros_like(ha)

        # observed pos
        dh_hpos_obs = ha * dL_df_pos_obs[:, None]
        dh_ha += hpos_obs * dL_df_pos_obs[:, None]

        # pos surrogate candidates
        dh_pos_cand: list[np.ndarray] = [
            np.zeros_like(pos_cand_h[p]) for p in range(P + 1)
        ]
        for p in range(P + 1):
            mask = idx_pos_surr == p
            if not np.any(mask):
                continue
            dh_pos_cand[p][mask] = ha[mask] * dL_df_pos_surr[mask, None]
            dh_ha[mask] += pos_cand_h[p][mask] * dL_df_pos_surr[mask, None]

        # observed neg
        dh_hneg_obs = ha * dL_df_neg_obs[:, None]
        dh_ha += hneg_obs * dL_df_neg_obs[:, None]

        # negative surrogate candidates
        dh_neg_cand: list[np.ndarray] = [
            np.zeros_like(neg_cand_h[j]) for j in range(M)
        ]
        for j in range(M):
            mask = idx_neg_surr == j
            if not np.any(mask):
                continue
            dh_neg_cand[j][mask] = ha[mask] * dL_df_neg_surr[mask, None]
            dh_ha[mask] += neg_cand_h[j][mask] * dL_df_neg_surr[mask, None]

        # ---- accumulate parameter grads through embed backward ----
        gW1 = np.zeros_like(model.W1)
        gb1 = np.zeros_like(model.b1)
        gW2 = np.zeros_like(model.W2)
        gb2 = np.zeros_like(model.b2)

        def _accum_grads(dh: np.ndarray, cache: Any) -> None:
            nonlocal gW1, gb1, gW2, gb2
            dW1, db1, dW2, db2, _dx = _embed_backward(dh, cache, model.W1, model.W2)
            gW1 += dW1
            gb1 += db1
            gW2 += dW2
            gb2 += db2

        _accum_grads(dh_ha, cache_ha)
        _accum_grads(dh_hpos_obs, cache_pos_obs)
        for p in range(P + 1):
            _accum_grads(dh_pos_cand[p], pos_cand_cache[p])
        _accum_grads(dh_hneg_obs, cache_neg_obs)
        for j in range(M):
            _accum_grads(dh_neg_cand[j], neg_cand_cache[j])

        grads = {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}
        model.apply_sgd(grads, lr=lr)

        # ---- metrics (observed pos/neg) ----
        m_pos = float(np.mean(f_pos_obs))
        m_neg = float(np.mean(f_neg_obs))
        m_margin = float(np.mean(f_pos_obs - f_neg_obs))
        frac_pos_gt_neg = float(np.mean(f_pos_obs > f_neg_obs))
        m_pos_obs_minus_surr = float(np.mean(f_pos_obs - f_pos_surr))
        m_neg_surr_minus_obs = float(np.mean(f_neg_surr - f_neg_obs))

        losses.append(loss)
        pos_means.append(m_pos)
        neg_means.append(m_neg)
        margins.append(m_margin)
        frac_pos_gt_neg_vals.append(frac_pos_gt_neg)
        pos_obs_minus_surr_vals.append(m_pos_obs_minus_surr)
        neg_surr_minus_obs_vals.append(m_neg_surr_minus_obs)

        if verbose:
            print(format_train_config_compact(cfg))
            print(f"[Train] step = {step}")
            print(f"[Train] loss = {loss}")
            print(f"[Train] mean_pos_logit = {m_pos}")
            print(f"[Train] mean_neg_logit = {m_neg}")
            print(f"[Train] mean_pos_minus_neg_logit = {m_margin}")
            print(f"[Diag] frac_pos_logit_gt_neg_logit = {frac_pos_gt_neg}")
            print(f"[Diag] mean_pos_obs_minus_surr_logit = {m_pos_obs_minus_surr}")
            print(f"[Diag] mean_neg_surr_minus_obs_logit = {m_neg_surr_minus_obs}")

    out = {
        "seed": seed,
        "env_id": env_id,
        "backend": "numpy",
        "trajectory_length": total_steps,
        "num_episodes": num_episodes,
        "valid_anchor_count": len(valid_anchors),
        "contrastive_k": k,
        "P": P,
        "M": M,
        "w": w,
        "num_train_steps": num_steps,
        "last_loss": losses[-1] if losses else math.nan,
        "last_mean_pos_logit": pos_means[-1] if pos_means else math.nan,
        "last_mean_neg_logit": neg_means[-1] if neg_means else math.nan,
        "last_mean_pos_minus_neg_logit": margins[-1] if margins else math.nan,
        "last_frac_pos_logit_gt_neg_logit": (
            frac_pos_gt_neg_vals[-1] if frac_pos_gt_neg_vals else math.nan
        ),
        "last_mean_pos_obs_minus_surr_logit": (
            pos_obs_minus_surr_vals[-1] if pos_obs_minus_surr_vals else math.nan
        ),
        "last_mean_neg_surr_minus_obs_logit": (
            neg_surr_minus_obs_vals[-1] if neg_surr_minus_obs_vals else math.nan
        ),
        "mean_loss": float(np.mean(losses)) if losses else math.nan,
        "mean_mean_pos_logit": float(np.mean(pos_means)) if pos_means else math.nan,
        "mean_mean_neg_logit": float(np.mean(neg_means)) if neg_means else math.nan,
        "mean_mean_pos_minus_neg_logit": float(np.mean(margins)) if margins else math.nan,
        "mean_frac_pos_logit_gt_neg_logit": (
            float(np.mean(frac_pos_gt_neg_vals)) if frac_pos_gt_neg_vals else math.nan
        ),
        "mean_mean_pos_obs_minus_surr_logit": (
            float(np.mean(pos_obs_minus_surr_vals))
            if pos_obs_minus_surr_vals
            else math.nan
        ),
        "mean_mean_neg_surr_minus_obs_logit": (
            float(np.mean(neg_surr_minus_obs_vals))
            if neg_surr_minus_obs_vals
            else math.nan
        ),
        "train_config": cfg,
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Phase 8 v1 bidirectional robust contrastive objective (NumPy)."
    )
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument("--env-id", type=str, default=TRAIN_ENV_ID)
    p.add_argument("--num-episodes", type=int, default=TRAIN_NUM_EPISODES)
    p.add_argument("--num-steps", type=int, default=TRAIN_NUM_STEPS)
    p.add_argument("--lr", type=float, default=TRAIN_NUMPY_LR)
    p.add_argument(
        "--P",
        type=int,
        default=ROBUST_V1_P,
        help="Positive window half offset (P=2 => t+k..t+k+2)",
    )
    p.add_argument(
        "--M",
        type=int,
        default=ROBUST_V1_M,
        help="Negative candidate pool size",
    )
    p.add_argument("--w", type=float, default=ROBUST_V1_W, help="Shared weight")
    p.add_argument("--quiet", action="store_true", help="Disable per-step printing")
    args = p.parse_args()

    train_numpy_robust_v1(
        seed=args.seed,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        lr=args.lr,
        P=args.P,
        M=args.M,
        w=args.w,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

