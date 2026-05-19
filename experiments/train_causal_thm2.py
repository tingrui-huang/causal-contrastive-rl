"""Train the contrastive critic with the Theorem 2 recursive sampler.

Pessimism is built into the critic via the sampling design of positives p+:
samples are drawn from d̲^π (the pessimistic causal reachability distribution)
instead of d^π. After training, the critic ≈ log(d̲^π / p) (Theorem 3) and the
actor uses raw critic scores with no additional pessimism layer.

This is Option 2 from the implementation plan — distinct from Option 1
(actor-side pessimism, see train_causal_pessimistic.py).
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

import numpy as np

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from agents.goal_conditioned_actor_numpy import GoalConditionedActorNumpy
from configs.training_defaults import (
    TRAIN_EMB_DIM,
    TRAIN_HIDDEN,
    TRAIN_MAX_EPISODE_STEPS,
    TRAIN_NUMPY_LR,
    TRAIN_NUM_EPISODES,
    TRAIN_NUM_STEPS,
    TRAIN_SEED,
    TRAIN_TAU,
)
import gymnasium as gym

from experiments.train_actor_critic import (
    GAMMA,
    LOSS_FAMILY,
    _build_batch,
    _build_valid_anchors,
)
from utils.causal_sampler import diag_summary, fresh_diag, sample_Sf_thm2
from utils.domain_knowledge import build_position_index, deduplicate_position_index
from utils.obs_transition import build_obs_transition_table
from utils.offline_data import collect_episodes
from utils.preprocess import extract_state, extract_state_oracle
from utils.propensity import build_propensity_table
from utils.value_function import compute_v_f_table, v_f_table_summary


def train(
    *,
    seed: int = TRAIN_SEED,
    env_id: str = "CausalContrastive-WindyCorridor-15x15-Lethal-v0",
    num_episodes: int = TRAIN_NUM_EPISODES,
    num_steps: int = TRAIN_NUM_STEPS,
    collector_mode: str = "multigoal_oracle",
    oracle_epsilon: float = 0.0,
    oracle_state: bool = False,
    lam: float = 0.5,
    critic_lr: float = TRAIN_NUMPY_LR,
    actor_lr: float = TRAIN_NUMPY_LR,
    critic_warmup: int = 2000,
    thm2_gamma: float = GAMMA,
    thm2_max_manhattan: int = 2,
    thm2_max_depth: int = 100,
    thm2_v_f_refresh: int = 200,
    checkpoint_tag: str | None = None,
    verbose: bool = True,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    if verbose:
        print(f"[Thm2] env={env_id}  collector={collector_mode}  oracle_state={oracle_state}  λ={lam}")
        print(f"[Thm2] episodes={num_episodes}  steps={num_steps}  warmup={critic_warmup}")
        print(f"[Thm2] gamma={thm2_gamma}  max_manhattan={thm2_max_manhattan}  "
              f"max_depth={thm2_max_depth}  vf_refresh={thm2_v_f_refresh}")

    state_fn = extract_state_oracle if oracle_state else extract_state
    (
        trajectories,
        episode_states,
        episode_actions,
        _valid_anchors_unused,
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
        state_fn=None if not oracle_state else extract_state_oracle,
    )

    valid_anchors = _build_valid_anchors(episode_states)
    batch_size = min(64, len(valid_anchors))

    # ─── one-shot construction of the Thm 2 support tables ───────────────
    propensity_table = build_propensity_table(episode_states, episode_actions, n_actions)
    obs_transition_table = build_obs_transition_table(trajectories, state_fn=state_fn)
    raw_index = build_position_index(episode_states)
    pos_index = deduplicate_position_index(raw_index, max_per_position=8, rng=rng)
    # Pull walkable cells from the env class — supports both WindyCorridor and
    # ConfoundedFork (or any subclass exposing walkable_cells()).
    probe_env = gym.make(env_id)
    walkable = set(probe_env.unwrapped.walkable_cells())
    probe_env.close()

    if verbose:
        print(f"[Thm2] propensity_cells={len(propensity_table)}  "
              f"obs_keys={len(obs_transition_table)}  pos_index_cells={len(pos_index)}  "
              f"walkable_cells={len(walkable)}")
        print(f"[Thm2] state_dim={state_dim}  n_actions={n_actions}  "
              f"total_transitions={total_steps}  batch={batch_size}")

    critic1 = ContrastiveCriticNumpy(state_dim=state_dim, n_actions=n_actions,
                                     hidden=TRAIN_HIDDEN, emb_dim=TRAIN_EMB_DIM,
                                     tau=TRAIN_TAU, seed=seed)
    critic2 = ContrastiveCriticNumpy(state_dim=state_dim, n_actions=n_actions,
                                     hidden=TRAIN_HIDDEN, emb_dim=TRAIN_EMB_DIM,
                                     tau=TRAIN_TAU, seed=seed + 100)
    actor = GoalConditionedActorNumpy(state_dim=state_dim, n_actions=n_actions,
                                      hidden=TRAIN_HIDDEN, seed=seed + 1)

    v_f_table: dict[tuple[int, int], float] = {}
    sampler_diag = fresh_diag()
    c_losses, a_losses, margins, bc_accs = [], [], [], []

    for step in range(1, num_steps + 1):
        anchor_batch = _build_batch(
            episode_states=episode_states,
            episode_actions=episode_actions,
            valid_anchors=valid_anchors,
            batch_size=batch_size,
            rng=rng,
        )

        # ── decide how to produce s_future ────────────────────────────────
        use_thm2 = (step > critic_warmup) and bool(v_f_table)
        if use_thm2:
            sf_list = [
                sample_Sf_thm2(
                    anchor_batch["s"][i],
                    int(anchor_batch["a"][i]),
                    propensity_table=propensity_table,
                    obs_transition_table=obs_transition_table,
                    v_f_table=v_f_table,
                    pos_index=pos_index,
                    walkable_cells=walkable,
                    n_actions=n_actions,
                    rng=rng,
                    gamma=thm2_gamma,
                    max_manhattan=thm2_max_manhattan,
                    max_depth=thm2_max_depth,
                    diag=sampler_diag,
                )
                for i in range(batch_size)
            ]
            s_future = np.stack(sf_list, axis=0).astype(np.float32, copy=False)
        else:
            s_future = anchor_batch["s_future"]

        # ── critic update ────────────────────────────────────────────────
        c1_loss, c1_grads = critic1.loss_and_grads(anchor_batch["s"], anchor_batch["a"], s_future)
        critic1.apply_sgd(c1_grads, lr=critic_lr)
        c2_loss, c2_grads = critic2.loss_and_grads(anchor_batch["s"], anchor_batch["a"], s_future)
        critic2.apply_sgd(c2_grads, lr=critic_lr)
        c_losses.append(0.5 * (c1_loss + c2_loss))
        margins.append(0.5 * (
            critic1.margin(anchor_batch["s"], anchor_batch["a"], s_future)
            + critic2.margin(anchor_batch["s"], anchor_batch["a"], s_future)
        ))

        # ── V_f table refresh ────────────────────────────────────────────
        if step >= critic_warmup and (step == critic_warmup or step % thm2_v_f_refresh == 0):
            v_f_table = compute_v_f_table(critic1, critic2, list(walkable), pos_index)

        # ── actor update ─────────────────────────────────────────────────
        if step > critic_warmup:
            scores1 = critic1.score_actions(anchor_batch["s"], s_future)
            scores2 = critic2.score_actions(anchor_batch["s"], s_future)
            f_all = np.minimum(scores1, scores2)
            a_loss, a_grads = actor.loss_and_grads(
                anchor_batch["s"], s_future, anchor_batch["a"], f_all, lam=lam
            )
            actor.apply_sgd(a_grads, lr=actor_lr)
            a_losses.append(float(a_loss))

            probs = actor.action_probs(
                anchor_batch["s"].astype(np.float64),
                s_future.astype(np.float64),
            )
            pred = np.argmax(probs, axis=1)
            bc_accs.append(float(np.mean(pred == anchor_batch["a"])))

        if verbose and (step == 1 or step % 1000 == 0 or step == num_steps):
            recent = slice(-200, None)
            a_str = f"{np.mean(a_losses[recent]):.4f}" if a_losses else "N/A"
            bc_str = f"{np.mean(bc_accs[recent]):.3f}" if bc_accs else "N/A"
            vf_str = v_f_table_summary(v_f_table)
            samp = diag_summary(sampler_diag) if sampler_diag.get("calls", 0) > 0 else None
            print(f"[Step {step:>6d}] critic_loss={np.mean(c_losses[recent]):.4f}  "
                  f"margin={np.mean(margins[recent]):.3f}  actor_loss={a_str}  bc_acc={bc_str}")
            print(f"[Step {step:>6d}] V_f n={vf_str['n']}  "
                  f"mean={vf_str['mean']:.3f}  min={vf_str['min']:.3f}  max={vf_str['max']:.3f}")
            if samp is not None:
                print(f"[Step {step:>6d}] sampler calls={samp['calls']}  "
                      f"mean_depth={samp['mean_depth']:.1f}  "
                      f"obs={samp['obs_branch_rate']:.2f}  pess={samp['pess_branch_rate']:.2f}  "
                      f"max_depth_hits={samp['max_depth_hit_rate']:.4f}")

    tag = "causal_thm2_oracle" if oracle_state else "causal_thm2"
    out: dict[str, Any] = {
        "method": tag,
        "final_critic_loss": float(np.mean(c_losses[-500:])),
        "final_margin": float(np.mean(margins[-500:])),
        "final_actor_loss": float(np.mean(a_losses[-500:])) if a_losses else None,
        "final_bc_acc": float(np.mean(bc_accs[-500:])) if bc_accs else None,
        "sampler_summary": diag_summary(sampler_diag),
        "v_f_summary": v_f_table_summary(v_f_table),
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
            "thm2_gamma": thm2_gamma, "thm2_max_manhattan": thm2_max_manhattan,
            "thm2_max_depth": thm2_max_depth, "thm2_v_f_refresh": thm2_v_f_refresh,
            "checkpoint_tag": auto_tag,
        }

        sd_c1 = critic1.state_dict()
        sd_c2 = critic2.state_dict()
        sd_actor = actor.state_dict()
        np.savez(
            ckpt_path,
            c1_sa_W1=sd_c1["sa_W1"], c1_sa_b1=sd_c1["sa_b1"],
            c1_sa_W2=sd_c1["sa_W2"], c1_sa_b2=sd_c1["sa_b2"],
            c1_g_W1=sd_c1["g_W1"], c1_g_b1=sd_c1["g_b1"],
            c1_g_W2=sd_c1["g_W2"], c1_g_b2=sd_c1["g_b2"],
            c2_sa_W1=sd_c2["sa_W1"], c2_sa_b1=sd_c2["sa_b1"],
            c2_sa_W2=sd_c2["sa_W2"], c2_sa_b2=sd_c2["sa_b2"],
            c2_g_W1=sd_c2["g_W1"], c2_g_b1=sd_c2["g_b1"],
            c2_g_W2=sd_c2["g_W2"], c2_g_b2=sd_c2["g_b2"],
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
    p = argparse.ArgumentParser(description="Train contrastive critic with Thm 2 sampler.")
    p.add_argument("--seed", type=int, default=TRAIN_SEED)
    p.add_argument("--env-id", type=str, default="CausalContrastive-WindyCorridor-15x15-Lethal-v0")
    p.add_argument("--num-episodes", type=int, default=TRAIN_NUM_EPISODES)
    p.add_argument("--num-steps", type=int, default=TRAIN_NUM_STEPS)
    p.add_argument("--collector-mode", type=str, default="multigoal_oracle")
    p.add_argument("--oracle-epsilon", type=float, default=0.0)
    p.add_argument("--oracle-state", action="store_true")
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--critic-lr", type=float, default=TRAIN_NUMPY_LR)
    p.add_argument("--actor-lr", type=float, default=TRAIN_NUMPY_LR)
    p.add_argument("--critic-warmup", type=int, default=2000)
    p.add_argument("--thm2-gamma", type=float, default=GAMMA)
    p.add_argument("--thm2-max-manhattan", type=int, default=2)
    p.add_argument("--thm2-max-depth", type=int, default=100)
    p.add_argument("--thm2-v-f-refresh", type=int, default=200)
    p.add_argument("--checkpoint-tag", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out = train(
        seed=args.seed, env_id=args.env_id,
        num_episodes=args.num_episodes, num_steps=args.num_steps,
        collector_mode=args.collector_mode, oracle_epsilon=args.oracle_epsilon,
        oracle_state=args.oracle_state, lam=args.lam,
        critic_lr=args.critic_lr, actor_lr=args.actor_lr,
        critic_warmup=args.critic_warmup,
        thm2_gamma=args.thm2_gamma,
        thm2_max_manhattan=args.thm2_max_manhattan,
        thm2_max_depth=args.thm2_max_depth,
        thm2_v_f_refresh=args.thm2_v_f_refresh,
        checkpoint_tag=args.checkpoint_tag,
        verbose=not args.quiet,
    )
    print(f"\n[Done] {json.dumps(out, indent=2, default=str)}")


if __name__ == "__main__":
    main()
