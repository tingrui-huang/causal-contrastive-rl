"""
Train a goal-conditioned contrastive critic + actor on WindyCorridor offline data.

Pipeline:
  1. Load .npz produced by collect_corridor_data.py
  2. HER-style positive sampling: s_future ~ Geom(1-γ) offset within trajectory
  3. Critic loss: in-batch sigmoid BCE (Eysenbach Alg. 1) — vanilla baseline,
     no Manski/propensity weighting (that's task #11)
  4. Actor loss: (1-λ)·advantage + λ·BC (Eysenbach Eq. 7-8)
  5. Save checkpoint with weights + config

Why HER even though we have a single fixed goal:
  Contrastive learning needs a non-degenerate s_g distribution for the in-batch
  negatives to make sense. We train the critic over all (s, a, s_future) triples
  and only ever query it at the fixed goal during evaluation.

Usage::
    python experiments/train_corridor.py \\
        --data data/corridor_expert_n1000.npz \\
        --num-steps 20000 \\
        --output checkpoints/corridor_baseline_seed0.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from agents.goal_conditioned_actor_numpy import GoalConditionedActorNumpy


GAMMA = 0.99
DEFAULT_HIDDEN = 128
DEFAULT_EMB_DIM = 64
DEFAULT_TAU = 0.07


def _geometric_offset(rng: np.random.Generator, max_val: int) -> int:
    """Sample positive offset ~ Geom(1-γ), clipped to [1, max_val]."""
    offset = int(rng.geometric(1.0 - GAMMA))
    return min(max(offset, 1), max_val)


def _build_valid_anchors(episode_states: list[np.ndarray]) -> list[tuple[int, int]]:
    """All (ep, t) such that t < len(states[ep]) - 1 (has at least one future)."""
    anchors: list[tuple[int, int]] = []
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
    """Sample anchors + HER future states."""
    anchor_ids = rng.integers(low=0, high=len(valid_anchors), size=batch_size)
    pairs = [valid_anchors[int(i)] for i in anchor_ids]

    s = np.stack([episode_states[ep][t] for ep, t in pairs], axis=0).astype(np.float32, copy=False)
    a = np.array([episode_actions[ep][t] for ep, t in pairs], dtype=np.int64)

    s_future_list = []
    for ep, t in pairs:
        max_off = len(episode_states[ep]) - t - 1
        off = _geometric_offset(rng, max_off)
        s_future_list.append(episode_states[ep][t + off])
    s_future = np.stack(s_future_list, axis=0).astype(np.float32, copy=False)
    return {"s": s, "a": a, "s_future": s_future}


def train(
    *,
    data_path: Path,
    output_path: Path,
    seed: int = 0,
    num_steps: int = 20000,
    batch_size: int = 64,
    critic_lr: float = 2e-3,
    actor_lr: float = 2e-3,
    critic_warmup: int = 1000,
    lam: float = 0.5,
    hidden: int = DEFAULT_HIDDEN,
    emb_dim: int = DEFAULT_EMB_DIM,
    tau: float = DEFAULT_TAU,
    log_interval: int = 500,
    verbose: bool = True,
) -> dict[str, Any]:
    data = np.load(data_path, allow_pickle=True)
    episode_states = list(data["episodes_states"])
    episode_actions = list(data["episodes_actions"])
    goal_state = np.asarray(data["goal_state"], dtype=np.float32)
    state_dim = int(data["state_dim"])
    n_actions = int(data["n_actions"])
    n_episodes = int(data["n_episodes"])

    valid_anchors = _build_valid_anchors(episode_states)
    if verbose:
        print(f"[Train] data={data_path.name}  episodes={n_episodes}  anchors={len(valid_anchors)}")
        print(f"[Train] state_dim={state_dim}  n_actions={n_actions}  γ={GAMMA}  τ={tau}")
        print(f"[Train] steps={num_steps}  batch={batch_size}  critic_lr={critic_lr}  λ={lam}")

    rng = np.random.default_rng(seed)
    critic = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=hidden,
        emb_dim=emb_dim,
        tau=tau,
        seed=seed,
    )
    actor = GoalConditionedActorNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=hidden,
        seed=seed + 1,
    )

    c_losses: list[float] = []
    a_losses: list[float] = []
    margins: list[float] = []
    bc_accs: list[float] = []

    t0 = time.time()
    for step in range(1, num_steps + 1):
        batch = _build_batch(
            episode_states=episode_states,
            episode_actions=episode_actions,
            valid_anchors=valid_anchors,
            batch_size=batch_size,
            rng=rng,
        )

        c_loss, c_grads = critic.loss_and_grads(batch["s"], batch["a"], batch["s_future"])
        critic.apply_sgd(c_grads, lr=critic_lr)
        c_losses.append(float(c_loss))
        margins.append(critic.margin(batch["s"], batch["a"], batch["s_future"]))

        if step > critic_warmup:
            scores = critic.score_actions(batch["s"], batch["s_future"])
            a_loss, a_grads = actor.loss_and_grads(
                batch["s"], batch["s_future"], batch["a"], scores, lam=lam,
            )
            actor.apply_sgd(a_grads, lr=actor_lr)
            a_losses.append(float(a_loss))

            probs = actor.action_probs(
                batch["s"].astype(np.float64), batch["s_future"].astype(np.float64),
            )
            bc_accs.append(float(np.mean(np.argmax(probs, axis=1) == batch["a"])))

        if verbose and (step % log_interval == 0 or step == 1):
            recent_c = float(np.mean(c_losses[-log_interval:]))
            recent_m = float(np.mean(margins[-log_interval:]))
            if a_losses:
                recent_a = float(np.mean(a_losses[-log_interval:]))
                recent_bc = float(np.mean(bc_accs[-log_interval:]))
                print(f"  step {step:6d}  c_loss={recent_c:.4f}  margin={recent_m:+.3f}  "
                      f"a_loss={recent_a:+.4f}  bc_acc={recent_bc:.3f}")
            else:
                print(f"  step {step:6d}  c_loss={recent_c:.4f}  margin={recent_m:+.3f}  (warmup)")

    if verbose:
        print(f"[Train] done in {time.time() - t0:.1f}s")

    config = {
        "seed": seed,
        "data_path": str(data_path),
        "num_steps": num_steps,
        "batch_size": batch_size,
        "critic_lr": critic_lr,
        "actor_lr": actor_lr,
        "critic_warmup": critic_warmup,
        "lam": lam,
        "gamma": GAMMA,
        "hidden": hidden,
        "emb_dim": emb_dim,
        "tau": tau,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "n_episodes": n_episodes,
    }

    # Merge state dicts, deduping shared scalars (state_dim, n_actions)
    critic_sd = critic.state_dict()
    actor_sd = actor.state_dict()
    for k in ("state_dim", "n_actions"):
        actor_sd.pop(k, None)  # critic version wins; they're identical anyway

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        **critic_sd,
        **actor_sd,
        goal_state=goal_state,
        c_losses=np.array(c_losses, dtype=np.float32),
        a_losses=np.array(a_losses, dtype=np.float32),
        margins=np.array(margins, dtype=np.float32),
        bc_accs=np.array(bc_accs, dtype=np.float32),
        config_json=np.array(json.dumps(config)),
    )
    if verbose:
        print(f"[Train] checkpoint → {output_path}")
    return {"critic": critic, "actor": actor, "config": config}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train contrastive critic + actor on corridor data.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "corridor_expert_n1000.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "corridor_baseline_seed0.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--critic-lr", type=float, default=2e-3)
    parser.add_argument("--actor-lr", type=float, default=2e-3)
    parser.add_argument("--critic-warmup", type=int, default=1000)
    parser.add_argument("--lam", type=float, default=0.5)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_path=args.output,
        seed=args.seed,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        critic_warmup=args.critic_warmup,
        lam=args.lam,
    )


if __name__ == "__main__":
    main()
