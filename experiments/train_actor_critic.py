"""
Joint Actor-Critic training (paper Section 5.5, offline RL variant).

Alternates:
  1. Critic update  — contrastive loss (Eq. 6, sigmoid BCE)
  2. Actor  update  — advantage + BC loss (Eq. 8)

Saves a combined checkpoint (critic weights + actor weights).

Usage:
  python experiments/train_actor_critic.py \
      --env-id CausalContrastive-WindyCorridor-15x15-Lethal-v0 \
      --num-episodes 50 --num-steps 20000 --lam 0.5

  # Oracle (U exposed):
  python experiments/train_actor_critic.py \
      --env-id CausalContrastive-WindyCorridor-15x15-Lethal-v0 \
      --oracle-state --num-episodes 50 --num-steps 20000 --lam 0.5
"""
from __future__ import annotations

import argparse
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

import envs  # noqa: F401

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from agents.goal_conditioned_actor_numpy import GoalConditionedActorNumpy
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
)
from utils.offline_data import collect_episodes
from utils.preprocess import extract_state_oracle

LOSS_FAMILY = "sigmoid_bce_weight1"
GAMMA = 0.95


def _geometric_offset(rng: np.random.Generator, max_val: int) -> int:
    """Sample offset from Geometric(1-γ), clipped to [1, max_val]."""
    offset = int(rng.geometric(1.0 - GAMMA))
    return min(max(offset, 1), max_val)


def _compute_critic_scores(
    critic: ContrastiveCriticNumpy,
    states: np.ndarray,
    goals: np.ndarray,
) -> np.ndarray:
    """f(s, a_i, sg) for all actions → (B, n_actions)."""
    B = states.shape[0]
    s64 = np.asarray(states, dtype=np.float64)
    g64 = np.asarray(goals, dtype=np.float64)
    scores = np.zeros((B, critic.n_actions), dtype=np.float64)
    for a_i in range(critic.n_actions):
        a_batch = np.full(B, a_i, dtype=np.int64)
        h_sa, _ = critic._embed(s64, a_batch)
        h_sg, _ = critic._embed(g64, a_batch)
        scores[:, a_i] = np.sum(h_sa * h_sg, axis=1) / critic.tau
    return scores


def _build_valid_anchors(
    episode_states: list[np.ndarray],
) -> list[tuple[int, int]]:
    """Any time step with at least 1 future state is a valid anchor."""
    anchors = []
    for ep_idx, states in enumerate(episode_states):
        for t in range(len(states) - 1):
            anchors.append((ep_idx, t))
    return anchors


def _build_batch(
    *,
    episode_states: list[np.ndarray],
    episode_actions: list[np.ndarray],
    valid_anchors: list[tuple[int, int]],
    buffer,
    state_dim: int,
    batch_size: int,
    rng: np.random.Generator,
    episode_regimes: list[int] | None = None,
    regime_buffers: dict | None = None,
    oracle_state: bool = False,
) -> dict[str, np.ndarray]:
    """Sample batch with geometric positive offset (paper Sec. 3, Eq. 3)."""
    anchor_ids = rng.integers(low=0, high=len(valid_anchors), size=batch_size)
    anchor_pairs = [valid_anchors[int(i)] for i in anchor_ids]

    s = np.stack(
        [episode_states[ep][t] for ep, t in anchor_pairs], axis=0
    ).astype(np.float32, copy=False)
    a = np.array(
        [episode_actions[ep][t] for ep, t in anchor_pairs], dtype=np.int64
    )

    s_pos_list = []
    for ep, t in anchor_pairs:
        max_off = len(episode_states[ep]) - t - 1
        off = _geometric_offset(rng, max_off)
        s_pos_list.append(episode_states[ep][t + off])
    s_pos = np.stack(s_pos_list, axis=0).astype(np.float32, copy=False)

    use_regime_neg = oracle_state and episode_regimes is not None and regime_buffers is not None
    s_neg = np.empty((batch_size, state_dim), dtype=np.float32)
    for i, (ep, _) in enumerate(anchor_pairs):
        if use_regime_neg:
            regime = episode_regimes[ep]
            neg_item = regime_buffers[regime].sample(1)[0]
        else:
            neg_item = buffer.sample(1)[0]
        s_neg[i] = np.asarray(neg_item["state"], dtype=np.float32)

    return {"s": s, "a": a, "s_pos": s_pos, "s_neg": s_neg}


def train(
    *,
    seed: int = TRAIN_SEED,
    env_id: str = TRAIN_ENV_ID,
    num_episodes: int = TRAIN_NUM_EPISODES,
    num_steps: int = TRAIN_NUM_STEPS,
    collector_mode: str = TRAIN_COLLECTOR_MODE,
    oracle_epsilon: float = TRAIN_ORACLE_EPSILON,
    oracle_state: bool = False,
    lam: float = 0.5,
    critic_lr: float = TRAIN_NUMPY_LR,
    actor_lr: float = TRAIN_NUMPY_LR,
    critic_warmup: int = 1000,
    verbose: bool = True,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    if verbose:
        print(f"[ActorCritic] env={env_id}  oracle_state={oracle_state}  λ={lam}")
        print(f"[ActorCritic] episodes={num_episodes}  steps={num_steps}  warmup={critic_warmup}")

    state_fn = extract_state_oracle if oracle_state else None
    (
        _trajectories,
        episode_states,
        episode_actions,
        valid_anchors,
        buffer,
        state_dim,
        n_actions,
        total_steps,
        k,
        rng,
        episode_regimes,
        regime_buffers,
    ) = collect_episodes(
        seed=seed,
        env_id=env_id,
        num_episodes=num_episodes,
        positive_window=0,
        collector_mode=collector_mode,
        oracle_epsilon=oracle_epsilon,
        max_episode_steps=TRAIN_MAX_EPISODE_STEPS,
        state_fn=state_fn,
    )

    valid_anchors = _build_valid_anchors(episode_states)
    batch_size = min(64, len(valid_anchors))
    if verbose:
        print(f"[ActorCritic] state_dim={state_dim}  n_actions={n_actions}  "
              f"total_transitions={total_steps}  batch={batch_size}  γ={GAMMA}")

    critic = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=TRAIN_HIDDEN,
        emb_dim=TRAIN_EMB_DIM,
        tau=TRAIN_TAU,
        seed=seed,
        loss_family=LOSS_FAMILY,
    )
    actor = GoalConditionedActorNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=TRAIN_HIDDEN,
        seed=seed + 1,
    )

    c_losses, a_losses = [], []
    margins, bc_accs = [], []

    for step in range(1, num_steps + 1):
        batch = _build_batch(
            episode_states=episode_states,
            episode_actions=episode_actions,
            valid_anchors=valid_anchors,
            buffer=buffer,
            state_dim=state_dim,
            batch_size=batch_size,
            rng=rng,
            episode_regimes=episode_regimes,
            regime_buffers=regime_buffers,
            oracle_state=oracle_state,
        )

        c_loss, c_grads = critic.loss_and_grads(
            batch["s"], batch["a"], batch["s_pos"], batch["s_neg"]
        )
        critic.apply_sgd(c_grads, lr=critic_lr)
        c_losses.append(float(c_loss))

        pl, nl = critic.logits(batch["s"], batch["a"], batch["s_pos"], batch["s_neg"])
        margin = float(np.mean(pl - nl))
        margins.append(margin)

        if step > critic_warmup:
            f_all = _compute_critic_scores(critic, batch["s"], batch["s_pos"])
            a_loss, a_grads = actor.loss_and_grads(
                batch["s"], batch["s_pos"], batch["a"], f_all, lam=lam
            )
            actor.apply_sgd(a_grads, lr=actor_lr)
            a_losses.append(float(a_loss))

            probs = actor.action_probs(
                batch["s"].astype(np.float64),
                batch["s_pos"].astype(np.float64),
            )
            pred = np.argmax(probs, axis=1)
            bc_acc = float(np.mean(pred == batch["a"]))
            bc_accs.append(bc_acc)

        if verbose and step % 2000 == 0:
            a_loss_str = f"{np.mean(a_losses[-200:]):.4f}" if a_losses else "N/A"
            bc_str = f"{np.mean(bc_accs[-200:]):.3f}" if bc_accs else "N/A"
            print(
                f"[Step {step:>6d}] critic_loss={np.mean(c_losses[-200:]):.4f}  "
                f"margin={np.mean(margins[-200:]):.3f}  "
                f"actor_loss={a_loss_str}  bc_acc={bc_str}"
            )

    tag = "actor_critic_oracle" if oracle_state else "actor_critic_confounded"
    out: dict[str, Any] = {
        "method": tag,
        "final_critic_loss": float(np.mean(c_losses[-500:])),
        "final_margin": float(np.mean(margins[-500:])),
        "final_actor_loss": float(np.mean(a_losses[-500:])) if a_losses else None,
        "final_bc_acc": float(np.mean(bc_accs[-500:])) if bc_accs else None,
    }

    if save_checkpoint:
        env_tag = env_id.replace("/", "_").replace("\\", "_").replace(":", "").replace("-", "_")
        collector_tag = collector_mode.replace("-", "_")
        ckpt_path = (
            ROOT / "checkpoints"
            / f"{tag}_seed{seed}_{env_tag}_{collector_tag}.npz"
        )
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        cfg = {
            "seed": seed, "env_id": env_id, "oracle_state": oracle_state,
            "lam": lam, "num_episodes": num_episodes, "num_steps": num_steps,
            "critic_warmup": critic_warmup, "loss_family": LOSS_FAMILY,
            "collector_mode": collector_mode, "oracle_epsilon": oracle_epsilon,
            "state_dim": state_dim, "n_actions": n_actions,
            "critic_lr": critic_lr, "actor_lr": actor_lr,
            "hidden": TRAIN_HIDDEN, "emb_dim": TRAIN_EMB_DIM, "tau": TRAIN_TAU,
            "method": tag, "has_actor": True,
        }

        sd_critic = critic.state_dict()
        sd_actor = actor.state_dict()

        np.savez(
            ckpt_path,
            W1=sd_critic["W1"], b1=sd_critic["b1"],
            W2=sd_critic["W2"], b2=sd_critic["b2"],
            actor_W1=sd_actor["actor_W1"], actor_b1=sd_actor["actor_b1"],
            actor_W2=sd_actor["actor_W2"], actor_b2=sd_actor["actor_b2"],
            state_dim=state_dim, n_actions=n_actions,
            tau=TRAIN_TAU, actor_hidden=TRAIN_HIDDEN,
            method=np.array(tag),
            loss_family=np.array(LOSS_FAMILY),
            train_config_json=np.array(json.dumps(cfg, default=str)),
        )
        out["checkpoint_path"] = str(ckpt_path)
        if verbose:
            print(f"\n[Checkpoint] {ckpt_path}")

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Joint Actor-Critic training (offline RL).")
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument("--env-id", type=str, default=TRAIN_ENV_ID)
    p.add_argument("--num-episodes", type=int, default=TRAIN_NUM_EPISODES)
    p.add_argument("--num-steps", type=int, default=TRAIN_NUM_STEPS)
    p.add_argument("--collector-mode", type=str, default=TRAIN_COLLECTOR_MODE)
    p.add_argument("--oracle-epsilon", type=float, default=TRAIN_ORACLE_EPSILON)
    p.add_argument("--oracle-state", action="store_true")
    p.add_argument("--lam", type=float, default=0.5, help="BC coefficient (1=pure BC, 0=pure advantage)")
    p.add_argument("--critic-lr", type=float, default=TRAIN_NUMPY_LR)
    p.add_argument("--actor-lr", type=float, default=TRAIN_NUMPY_LR)
    p.add_argument("--critic-warmup", type=int, default=1000)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    result = train(
        seed=args.seed,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        collector_mode=args.collector_mode,
        oracle_epsilon=args.oracle_epsilon,
        oracle_state=args.oracle_state,
        lam=args.lam,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        critic_warmup=args.critic_warmup,
        verbose=not args.quiet,
    )
    print(f"\n[Done] {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
