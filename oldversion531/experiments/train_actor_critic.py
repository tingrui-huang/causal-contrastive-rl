"""
Joint Actor-Critic training (paper Section 5.5, offline RL variant).

Architecture aligned with Eysenbach et al. (2022) Algorithm 1:
  - Dual-encoder critic: f(s,a,s_g) = phi(s,a)^T psi(s_g) / tau
  - In-batch negatives: B x B logit matrix, sigmoid BCE
  - Twin critics with min-Q for actor scoring (Section 5.5)

Actor update — advantage + BC loss (Eq. 7-8):
  max_pi  E[(1-lam) * f(s,a,sg) + lam * log pi(a_orig | s, sg)]

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

LOSS_FAMILY = "inbatch_sigmoid_bce"
GAMMA = 0.95


def _geometric_offset(rng: np.random.Generator, max_val: int) -> int:
    """Sample offset from Geometric(1-gamma), clipped to [1, max_val]."""
    offset = int(rng.geometric(1.0 - GAMMA))
    return min(max(offset, 1), max_val)


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
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample batch: (s, a, s_future) with geometric positive offset.

    Negatives are handled in-batch by the critic (B x B logit matrix),
    so no explicit s_neg is needed.
    """
    anchor_ids = rng.integers(low=0, high=len(valid_anchors), size=batch_size)
    anchor_pairs = [valid_anchors[int(i)] for i in anchor_ids]

    s = np.stack(
        [episode_states[ep][t] for ep, t in anchor_pairs], axis=0
    ).astype(np.float32, copy=False)
    a = np.array(
        [episode_actions[ep][t] for ep, t in anchor_pairs], dtype=np.int64
    )

    s_future_list = []
    for ep, t in anchor_pairs:
        max_off = len(episode_states[ep]) - t - 1
        off = _geometric_offset(rng, max_off)
        s_future_list.append(episode_states[ep][t + off])
    s_future = np.stack(s_future_list, axis=0).astype(np.float32, copy=False)

    return {"s": s, "a": a, "s_future": s_future}


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
    checkpoint_tag: str | None = None,
    verbose: bool = True,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    if verbose:
        print(f"[ActorCritic] env={env_id}  oracle_state={oracle_state}  λ={lam}")
        print(f"[ActorCritic] episodes={num_episodes}  steps={num_steps}  warmup={critic_warmup}")
        print(f"[ActorCritic] twin_critics=True  loss={LOSS_FAMILY}")

    state_fn = extract_state_oracle if oracle_state else None
    (
        _trajectories,
        episode_states,
        episode_actions,
        valid_anchors,
        _buffer,
        state_dim,
        n_actions,
        total_steps,
        _k,
        rng,
        _episode_regimes,
        _regime_buffers,
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

    # --- Twin critics (Section 5.5: min-Q for offline RL) -----------------
    critic1 = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=TRAIN_HIDDEN,
        emb_dim=TRAIN_EMB_DIM,
        tau=TRAIN_TAU,
        seed=seed,
    )
    critic2 = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=TRAIN_HIDDEN,
        emb_dim=TRAIN_EMB_DIM,
        tau=TRAIN_TAU,
        seed=seed + 100,
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
            batch_size=batch_size,
            rng=rng,
        )

        # Train both critics on the same batch
        c1_loss, c1_grads = critic1.loss_and_grads(
            batch["s"], batch["a"], batch["s_future"]
        )
        critic1.apply_sgd(c1_grads, lr=critic_lr)

        c2_loss, c2_grads = critic2.loss_and_grads(
            batch["s"], batch["a"], batch["s_future"]
        )
        critic2.apply_sgd(c2_grads, lr=critic_lr)

        c_loss = 0.5 * (c1_loss + c2_loss)
        c_losses.append(c_loss)

        m1 = critic1.margin(batch["s"], batch["a"], batch["s_future"])
        m2 = critic2.margin(batch["s"], batch["a"], batch["s_future"])
        margins.append(0.5 * (m1 + m2))

        if step > critic_warmup:
            # Min-Q: pessimistic action scoring from twin critics
            scores1 = critic1.score_actions(batch["s"], batch["s_future"])
            scores2 = critic2.score_actions(batch["s"], batch["s_future"])
            f_all = np.minimum(scores1, scores2)

            a_loss, a_grads = actor.loss_and_grads(
                batch["s"], batch["s_future"], batch["a"], f_all, lam=lam
            )
            actor.apply_sgd(a_grads, lr=actor_lr)
            a_losses.append(float(a_loss))

            probs = actor.action_probs(
                batch["s"].astype(np.float64),
                batch["s_future"].astype(np.float64),
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
        auto_tag = checkpoint_tag
        if auto_tag is None and (num_steps < 1000 or num_episodes < 10):
            auto_tag = f"smoke_steps{num_steps}_eps{num_episodes}"
        suffix = f"_{auto_tag}" if auto_tag else ""
        ckpt_path = (
            ROOT / "checkpoints"
            / f"{tag}_seed{seed}_{env_tag}_{collector_tag}{suffix}.npz"
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
            "method": tag, "has_actor": True, "twin_critics": True,
            "checkpoint_tag": auto_tag,
        }

        sd_c1 = critic1.state_dict()
        sd_c2 = critic2.state_dict()
        sd_actor = actor.state_dict()

        np.savez(
            ckpt_path,
            # Critic 1 (sa_encoder + g_encoder)
            c1_sa_W1=sd_c1["sa_W1"], c1_sa_b1=sd_c1["sa_b1"],
            c1_sa_W2=sd_c1["sa_W2"], c1_sa_b2=sd_c1["sa_b2"],
            c1_g_W1=sd_c1["g_W1"], c1_g_b1=sd_c1["g_b1"],
            c1_g_W2=sd_c1["g_W2"], c1_g_b2=sd_c1["g_b2"],
            # Critic 2 (sa_encoder + g_encoder)
            c2_sa_W1=sd_c2["sa_W1"], c2_sa_b1=sd_c2["sa_b1"],
            c2_sa_W2=sd_c2["sa_W2"], c2_sa_b2=sd_c2["sa_b2"],
            c2_g_W1=sd_c2["g_W1"], c2_g_b1=sd_c2["g_b1"],
            c2_g_W2=sd_c2["g_W2"], c2_g_b2=sd_c2["g_b2"],
            # Actor
            actor_W1=sd_actor["actor_W1"], actor_b1=sd_actor["actor_b1"],
            actor_W2=sd_actor["actor_W2"], actor_b2=sd_actor["actor_b2"],
            # Metadata
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
    p.add_argument("--checkpoint-tag", type=str, default=None)
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
        checkpoint_tag=args.checkpoint_tag,
        verbose=not args.quiet,
    )
    print(f"\n[Done] {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
