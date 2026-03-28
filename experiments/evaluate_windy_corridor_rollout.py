"""Ordinary rollout evaluation for WindyCorridor checkpoints.

Unlike `evaluate_windy_corridor_forced_u.py`, this script does NOT override `fixed_u`.
It measures the actual success rate of the extracted policy on the chosen env id.
"""
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from minigrid.core.actions import Actions

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs  # noqa: F401

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from utils.offline_data import build_policy
from utils.preprocess import extract_state

ACTION_IDS = [int(Actions.left), int(Actions.right), int(Actions.forward)]
CSV_FIELDS = [
    "checkpoint",
    "train_env_id",
    "eval_env_id",
    "success_rate",
    "mean_return",
    "mean_steps",
]


def _load_numpy_checkpoint(path: Path) -> tuple[ContrastiveCriticNumpy, dict[str, Any]]:
    ckpt = np.load(path, allow_pickle=True)
    state_dim = int(ckpt["state_dim"])
    n_actions = int(ckpt["n_actions"])
    tau = float(ckpt["tau"])
    hidden = int(ckpt["W1"].shape[1])
    emb_dim = int(ckpt["W2"].shape[1])
    model = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=hidden,
        emb_dim=emb_dim,
        tau=tau,
        seed=0,
    )
    model.W1[...] = ckpt["W1"]
    model.b1[...] = ckpt["b1"]
    model.W2[...] = ckpt["W2"]
    model.b2[...] = ckpt["b2"]
    cfg = json.loads(str(ckpt["train_config_json"]))
    return model, cfg


def _reference_trajectory(
    *,
    env_id: str,
    seed: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    env = gym.make(env_id, render_mode="rgb_array")
    env.action_space.seed(seed)
    policy = build_policy(env, collector_mode="oracle_eps", oracle_epsilon=0.0, seed=seed)

    obs, _ = env.reset(seed=seed)
    states = [extract_state(obs)]
    positions = [tuple(int(v) for v in env.unwrapped.agent_pos)]
    for _ in range(env.unwrapped.max_steps):
        action = int(policy(obs))
        next_obs, _, terminated, truncated, _ = env.step(action)
        states.append(extract_state(next_obs))
        positions.append(tuple(int(v) for v in env.unwrapped.agent_pos))
        obs = next_obs
        if terminated or truncated:
            break
    env.close()
    return np.stack(states, axis=0).astype(np.float64, copy=False), positions


def _score_action(
    model: ContrastiveCriticNumpy,
    state: np.ndarray,
    action_id: int,
    future_bank: np.ndarray,
) -> float:
    bank_count = future_bank.shape[0]
    s_batch = np.repeat(state[None, :], bank_count, axis=0)
    a_batch = np.full(bank_count, action_id, dtype=np.int64)
    h_sa, _ = model._embed(s_batch, a_batch)
    h_goal, _ = model._embed(future_bank, a_batch)
    return float(np.max(np.sum(h_sa * h_goal, axis=1)))


def _future_window_bank(
    ref_states: np.ndarray,
    *,
    ref_idx: int,
    lookahead_k: int,
    future_window: int,
) -> np.ndarray:
    start = min(ref_idx + lookahead_k, len(ref_states) - 1)
    stop = min(start + future_window + 1, len(ref_states))
    return ref_states[start:stop]


def _goal_bank(
    ref_states: np.ndarray,
    *,
    ref_idx: int,
    lookahead_k: int,
    future_window: int,
    goal_bank_mode: str,
) -> np.ndarray:
    if goal_bank_mode == "waypoint":
        return _future_window_bank(
            ref_states,
            ref_idx=ref_idx,
            lookahead_k=lookahead_k,
            future_window=future_window,
        )
    if goal_bank_mode == "final_only":
        return ref_states[-1:]
    raise ValueError(f"Unknown goal_bank_mode: {goal_bank_mode!r}")


def _reference_index(
    *,
    step_idx: int,
    state: np.ndarray,
    ref_states: np.ndarray,
    alignment_mode: str,
) -> int:
    if alignment_mode == "time":
        return min(step_idx, len(ref_states) - 1)
    if alignment_mode == "nearest":
        d2 = np.sum((ref_states - state[None, :]) ** 2, axis=1)
        return int(np.argmin(d2))
    raise ValueError(f"Unknown alignment_mode: {alignment_mode!r}")


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _plan_action(
    *,
    env,
    obs,
    model: ContrastiveCriticNumpy,
    ref_states: np.ndarray,
    ref_positions: list[tuple[int, int]],
    step_idx: int,
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_bank_mode: str,
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
) -> int:
    best_value = -np.inf
    best_action = ACTION_IDS[0]

    for action_id in ACTION_IDS:
        for suffix in itertools.product(ACTION_IDS, repeat=max(0, plan_depth - 1)):
            seq = (action_id, *suffix)
            sim_env = copy.deepcopy(env)
            sim_obs = obs
            seq_value = 0.0
            for depth_idx, seq_action in enumerate(seq):
                sim_raw_before = sim_env.unwrapped
                pos_before = tuple(int(v) for v in sim_raw_before.agent_pos)
                sim_state = extract_state(sim_obs).astype(np.float64, copy=False)
                sim_ref_idx = _reference_index(
                    step_idx=step_idx + depth_idx,
                    state=sim_state,
                    ref_states=ref_states,
                    alignment_mode=alignment_mode,
                )
                future_bank = _goal_bank(
                    ref_states,
                    ref_idx=sim_ref_idx,
                    lookahead_k=lookahead_k,
                    future_window=future_window,
                    goal_bank_mode=goal_bank_mode,
                )
                if goal_bank_mode == "waypoint":
                    progress_target_pos = ref_positions[
                        min(sim_ref_idx + lookahead_k, len(ref_positions) - 1)
                    ]
                elif goal_bank_mode == "final_only":
                    progress_target_pos = ref_positions[-1]
                else:
                    raise ValueError(f"Unknown goal_bank_mode: {goal_bank_mode!r}")
                seq_value += _score_action(model, sim_state, seq_action, future_bank)

                sim_next_obs, reward, sim_terminated, sim_truncated, _ = sim_env.step(seq_action)
                sim_raw_after = sim_env.unwrapped
                pos_after = tuple(int(v) for v in sim_raw_after.agent_pos)
                seq_value += progress_bonus * float(
                    _manhattan(pos_before, progress_target_pos)
                    - _manhattan(pos_after, progress_target_pos)
                )
                if seq_action == int(Actions.forward) and pos_after == pos_before:
                    seq_value -= collision_penalty
                elif seq_action in (int(Actions.left), int(Actions.right)):
                    seq_value -= turn_penalty
                if sim_terminated and reward > 0:
                    seq_value += success_bonus

                sim_obs = sim_next_obs
                if sim_terminated or sim_truncated:
                    break

            if seq_value > best_value:
                best_value = seq_value
                best_action = action_id

    return best_action


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    eval_env_id: str,
    episodes: int,
    lookahead_k: int,
    future_window: int,
    alignment_mode: str,
    goal_bank_mode: str,
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
) -> dict[str, Any]:
    model, cfg = _load_numpy_checkpoint(checkpoint)
    successes = 0
    returns: list[float] = []
    steps_list: list[int] = []

    for ep in range(episodes):
        seed = 100 + ep
        ref_states, ref_positions = _reference_trajectory(env_id=eval_env_id, seed=seed)
        env = gym.make(eval_env_id, render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        done = False
        episode_return = 0.0
        steps = 0
        while not done and steps < 120:
            action = _plan_action(
                env=env,
                obs=obs,
                model=model,
                ref_states=ref_states,
                ref_positions=ref_positions,
                step_idx=steps,
                lookahead_k=lookahead_k,
                future_window=future_window,
                alignment_mode=alignment_mode,
                goal_bank_mode=goal_bank_mode,
                plan_depth=plan_depth,
                collision_penalty=collision_penalty,
                turn_penalty=turn_penalty,
                progress_bonus=progress_bonus,
                success_bonus=success_bonus,
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
            steps += 1
            if terminated and reward > 0:
                successes += 1
        env.close()
        returns.append(episode_return)
        steps_list.append(steps)

    success_rate = successes / float(episodes)
    mean_return = float(np.mean(returns))
    mean_steps = float(np.mean(steps_list))

    print(f"[RolloutEval] success_rate = {success_rate:.3f}")
    print(f"[RolloutEval] mean_return = {mean_return:.3f}")
    print(f"[RolloutEval] mean_steps = {mean_steps:.3f}")

    return {
        "checkpoint": str(checkpoint),
        "train_env_id": cfg.get("env_id"),
        "eval_env_id": eval_env_id,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_steps": mean_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout evaluation for WindyCorridor checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-env-id", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--lookahead-k", type=int, default=4)
    parser.add_argument("--future-window", type=int, default=2)
    parser.add_argument(
        "--goal-bank-mode",
        type=str,
        default="waypoint",
        choices=["waypoint", "final_only"],
    )
    parser.add_argument(
        "--alignment-mode",
        type=str,
        default="time",
        choices=["time", "nearest"],
    )
    parser.add_argument("--plan-depth", type=int, default=2)
    parser.add_argument("--collision-penalty", type=float, default=2.0)
    parser.add_argument("--turn-penalty", type=float, default=0.05)
    parser.add_argument("--progress-bonus", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "windy_corridor_rollout_eval.csv",
    )
    args = parser.parse_args()

    row = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        eval_env_id=args.eval_env_id,
        episodes=args.episodes,
        lookahead_k=args.lookahead_k,
        future_window=args.future_window,
        alignment_mode=args.alignment_mode,
        goal_bank_mode=args.goal_bank_mode,
        plan_depth=args.plan_depth,
        collision_penalty=args.collision_penalty,
        turn_penalty=args.turn_penalty,
        progress_bonus=args.progress_bonus,
        success_bonus=args.success_bonus,
    )

    if not args.stdout_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow(row)

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)


if __name__ == "__main__":
    main()
