"""Smoke test for the Thm 2 sampler stack.

Collects a small multigoal_oracle dataset, builds the four supporting tables
(propensity, obs_transition, pos_index, V_f), then exercises sample_Sf_thm2
across the batch and prints diagnostics. Confirms wiring before launching a
full training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import envs  # noqa: F401

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from configs.training_defaults import (
    TRAIN_EMB_DIM,
    TRAIN_HIDDEN,
    TRAIN_MAX_EPISODE_STEPS,
    TRAIN_TAU,
)
from envs.windy_corridor import WindyCorridorEnv
from utils.causal_sampler import diag_summary, fresh_diag, sample_Sf_thm2
from utils.domain_knowledge import build_position_index, deduplicate_position_index
from utils.obs_transition import build_obs_transition_table, fallback_diagnostics as obs_diag
from utils.offline_data import collect_episodes
from utils.preprocess import extract_state
from utils.propensity import build_propensity_table, fallback_diagnostics as prop_diag
from utils.value_function import compute_v_f_table, v_f_table_summary


def main() -> None:
    env_id = "CausalContrastive-WindyCorridor-15x15-Lethal-v0"
    seed = 0

    (
        trajectories,
        episode_states,
        episode_actions,
        _valid_anchors,
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
        num_episodes=10,
        positive_window=0,
        collector_mode="multigoal_oracle",
        oracle_epsilon=0.0,
        max_episode_steps=TRAIN_MAX_EPISODE_STEPS,
    )
    print(f"[smoke] collected {len(trajectories)} episodes, {total_steps} transitions")

    # --- build the four tables ---
    propensity_table = build_propensity_table(episode_states, episode_actions, n_actions)
    pd = prop_diag(propensity_table, episode_states)
    print(f"[smoke] propensity table: cells={pd['unique_cells']}  fallback_rate={pd['fallback_rate']:.3f}")

    obs_table = build_obs_transition_table(trajectories, state_fn=extract_state)
    od = obs_diag(obs_table, trajectories, state_fn=extract_state)
    print(f"[smoke] obs_transition table: keys={od['unique_keys']}  fallback_rate={od['fallback_rate']:.3f}  "
          f"mean_samples/key={od['mean_samples_per_key']:.1f}  min_samples/key={od['min_samples_per_key']}")

    raw_index = build_position_index(episode_states)
    pos_index = deduplicate_position_index(raw_index, max_per_position=8, rng=rng)
    print(f"[smoke] pos_index: {len(pos_index)} cells with state samples")

    # --- build critic, compute V_f ---
    critic1 = ContrastiveCriticNumpy(state_dim=state_dim, n_actions=n_actions,
                                     hidden=TRAIN_HIDDEN, emb_dim=TRAIN_EMB_DIM, tau=TRAIN_TAU, seed=0)
    critic2 = ContrastiveCriticNumpy(state_dim=state_dim, n_actions=n_actions,
                                     hidden=TRAIN_HIDDEN, emb_dim=TRAIN_EMB_DIM, tau=TRAIN_TAU, seed=100)

    walkable = WindyCorridorEnv.walkable_cells()
    v_f_table = compute_v_f_table(critic1, critic2, list(walkable), pos_index)
    print(f"[smoke] V_f table: {v_f_table_summary(v_f_table)}")

    # --- exercise the sampler ---
    diag = fresh_diag()
    n_samples = 64
    sf_list = []
    anchor_eps = rng.integers(0, len(episode_states), size=n_samples)
    anchor_ts = [int(rng.integers(0, len(episode_actions[ep]))) for ep in anchor_eps]
    for ep, t in zip(anchor_eps, anchor_ts):
        s = episode_states[ep][t]
        a = int(episode_actions[ep][t])
        sf = sample_Sf_thm2(
            s, a,
            propensity_table=propensity_table,
            obs_transition_table=obs_table,
            v_f_table=v_f_table,
            pos_index=pos_index,
            walkable_cells=set(walkable),
            n_actions=n_actions,
            rng=rng,
            gamma=0.95,
            max_manhattan=2,
            max_depth=100,
            diag=diag,
        )
        sf_list.append(sf)

    sf_arr = np.stack(sf_list, axis=0)
    print(f"[smoke] Sf samples shape: {sf_arr.shape}  any NaN: {bool(np.isnan(sf_arr).any())}")
    print(f"[smoke] sampler diagnostics: {diag_summary(diag)}")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
