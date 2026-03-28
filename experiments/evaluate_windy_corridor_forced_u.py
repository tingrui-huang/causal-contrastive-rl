"""Forced-U evaluation for WindyCorridor NumPy checkpoints."""
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
    "success_u0",
    "success_u1",
    "mean_success",
    "worst_case_success",
    "regime_gap",
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


def _goal_bank_from_clean_env(eval_env_id: str, seed: int) -> np.ndarray:
    clean_env_id = eval_env_id.replace("-15x15-v0", "-15x15-Clean-v0")
    env = gym.make(clean_env_id, render_mode="rgb_array", fixed_u=0)
    env.action_space.seed(seed)
    policy = build_policy(env, collector_mode="oracle_eps", oracle_epsilon=0.0, seed=seed)

    obs, _ = env.reset(seed=seed)
    states = [extract_state(obs)]
    for _ in range(env.unwrapped.max_steps):
        action = policy(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        states.append(extract_state(next_obs))
        obs = next_obs
        if terminated or truncated:
            break
    env.close()

    bank = np.stack(states[-6:], axis=0).astype(np.float64, copy=False)
    return bank


def _score_action(
    model: ContrastiveCriticNumpy,
    state: np.ndarray,
    action_id: int,
    goal_bank: np.ndarray,
) -> float:
    s_batch = np.repeat(state[None, :], goal_bank.shape[0], axis=0)
    a_batch = np.full(goal_bank.shape[0], action_id, dtype=np.int64)
    h_sa, _ = model._embed(s_batch, a_batch)
    h_goal, _ = model._embed(goal_bank, a_batch)
    return float(np.max(np.sum(h_sa * h_goal, axis=1)))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _plan_action(
    *,
    env,
    obs,
    model: ContrastiveCriticNumpy,
    goal_bank: np.ndarray,
    goal_pos: tuple[int, int],
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
) -> int:
    state = extract_state(obs).astype(np.float64, copy=False)
    best_value = -np.inf
    best_action = ACTION_IDS[0]

    for action_id in ACTION_IDS:
        for suffix in itertools.product(ACTION_IDS, repeat=max(0, plan_depth - 1)):
            seq = (action_id, *suffix)
            sim_env = copy.deepcopy(env)
            sim_obs = obs
            seq_value = 0.0
            for seq_action in seq:
                sim_raw_before = sim_env.unwrapped
                pos_before = tuple(int(v) for v in sim_raw_before.agent_pos)
                sim_state = extract_state(sim_obs).astype(np.float64, copy=False)
                seq_value += _score_action(model, sim_state, seq_action, goal_bank)

                sim_next_obs, reward, sim_terminated, sim_truncated, _ = sim_env.step(seq_action)
                sim_raw_after = sim_env.unwrapped
                pos_after = tuple(int(v) for v in sim_raw_after.agent_pos)

                seq_value += progress_bonus * float(
                    _manhattan(pos_before, goal_pos) - _manhattan(pos_after, goal_pos)
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


def _run_regime(
    *,
    model: ContrastiveCriticNumpy,
    eval_env_id: str,
    fixed_u: int,
    episodes: int,
    goal_bank: np.ndarray,
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
) -> float:
    successes = 0
    for ep in range(episodes):
        seed = 100 + fixed_u * 1000 + ep
        env = gym.make(eval_env_id, render_mode="rgb_array", fixed_u=fixed_u)
        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        goal_pos = env.unwrapped.goal_pos()
        while not done and steps < 120:
            best_action = _plan_action(
                env=env,
                obs=obs,
                model=model,
                goal_bank=goal_bank,
                goal_pos=goal_pos,
                plan_depth=plan_depth,
                collision_penalty=collision_penalty,
                turn_penalty=turn_penalty,
                progress_bonus=progress_bonus,
                success_bonus=success_bonus,
            )
            obs, reward, terminated, truncated, _ = env.step(best_action)
            done = terminated or truncated
            if reward > 0:
                successes += 1
            steps += 1
        env.close()
    return successes / float(episodes)


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    eval_env_id: str,
    episodes_per_regime: int,
    plan_depth: int,
    collision_penalty: float,
    turn_penalty: float,
    progress_bonus: float,
    success_bonus: float,
) -> dict[str, Any]:
    model, cfg = _load_numpy_checkpoint(checkpoint)
    goal_bank = _goal_bank_from_clean_env(eval_env_id, seed=int(cfg.get("seed", 0)))

    success_u0 = _run_regime(
        model=model,
        eval_env_id=eval_env_id,
        fixed_u=0,
        episodes=episodes_per_regime,
        goal_bank=goal_bank,
        plan_depth=plan_depth,
        collision_penalty=collision_penalty,
        turn_penalty=turn_penalty,
        progress_bonus=progress_bonus,
        success_bonus=success_bonus,
    )
    success_u1 = _run_regime(
        model=model,
        eval_env_id=eval_env_id,
        fixed_u=1,
        episodes=episodes_per_regime,
        goal_bank=goal_bank,
        plan_depth=plan_depth,
        collision_penalty=collision_penalty,
        turn_penalty=turn_penalty,
        progress_bonus=progress_bonus,
        success_bonus=success_bonus,
    )
    mean_success = 0.5 * (success_u0 + success_u1)
    worst_case_success = min(success_u0, success_u1)
    regime_gap = abs(success_u0 - success_u1)

    print(f"[Eval] success_u0 = {success_u0:.3f}")
    print(f"[Eval] success_u1 = {success_u1:.3f}")
    print(f"[Eval] mean_success = {mean_success:.3f}")
    print(f"[Eval] worst_case_success = {worst_case_success:.3f}")
    print(f"[Eval] regime_gap = {regime_gap:.3f}")

    return {
        "checkpoint": str(checkpoint),
        "train_env_id": cfg.get("env_id"),
        "eval_env_id": eval_env_id,
        "success_u0": success_u0,
        "success_u1": success_u1,
        "mean_success": mean_success,
        "worst_case_success": worst_case_success,
        "regime_gap": regime_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Forced-U evaluation for WindyCorridor checkpoints.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-env-id", type=str, default="CausalContrastive-WindyCorridor-15x15-v0")
    parser.add_argument("--episodes-per-regime", type=int, default=20)
    parser.add_argument("--plan-depth", type=int, default=2)
    parser.add_argument("--collision-penalty", type=float, default=2.0)
    parser.add_argument("--turn-penalty", type=float, default=0.05)
    parser.add_argument("--progress-bonus", type=float, default=0.5)
    parser.add_argument("--success-bonus", type=float, default=5.0)
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "windy_corridor_forced_u_eval.csv",
    )
    args = parser.parse_args()

    row = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        eval_env_id=args.eval_env_id,
        episodes_per_regime=args.episodes_per_regime,
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
