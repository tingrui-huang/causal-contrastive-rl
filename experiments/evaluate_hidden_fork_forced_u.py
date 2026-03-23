"""
Forced-regime policy evaluation for HiddenFork checkpoints.

The same heuristic policy-extraction rule is reused for every compared method:
- action subset fixed once for the whole run
- score(s, a) = max_g cosine(h(s, a), h(g, a)) over a shared goal bank

Primary metrics:
- success_u0
- success_u1
- mean_success
- worst_case_success
- regime_gap
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs  # noqa: F401

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from utils.offline_data import build_policy
from utils.preprocess import extract_state

ENV_CONF = "CausalContrastive-HiddenFork-15x15-v0"
ACTION_NAME_TO_ID = {
    "left": 0,
    "right": 1,
    "forward": 2,
}

CSV_FIELDS = [
    "method",
    "loss_family",
    "checkpoint",
    "seed",
    "env_id",
    "eval_env_id",
    "collector_mode",
    "num_episodes",
    "num_train_steps",
    "w",
    "action_subset",
    "goal_bank_size",
    "episodes_per_regime",
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
    meta = {
        "method": str(ckpt["method"]) if "method" in ckpt else str(cfg.get("method", "unknown")),
        "loss_family": str(ckpt["loss_family"]) if "loss_family" in ckpt else str(cfg.get("loss_family", "unknown")),
        "w": float(ckpt["w"]) if "w" in ckpt else cfg.get("w"),
        "train_config": cfg,
    }
    return model, meta


def _goal_bank_from_fixed_u(
    *,
    env_id: str,
    fixed_u: int,
    goal_bank_size: int,
) -> np.ndarray:
    env = gym.make(env_id, render_mode="rgb_array", fixed_u=fixed_u)
    env.action_space.seed(0)
    policy = build_policy(
        env,
        collector_mode="oracle_eps",
        oracle_epsilon=0.0,
        seed=0,
    )
    obs, _ = env.reset(seed=0)
    states = [extract_state(obs)]
    for _ in range(env.unwrapped.max_steps):
        action = policy(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        states.append(extract_state(next_obs))
        obs = next_obs
        if terminated or truncated:
            break
    tail = states[-goal_bank_size:]
    return np.stack(tail, axis=0).astype(np.float64, copy=False)


def _score_action(
    model: ContrastiveCriticNumpy,
    state: np.ndarray,
    action_id: int,
    goal_bank: np.ndarray,
) -> float:
    goal_count = goal_bank.shape[0]
    s_batch = np.repeat(state[None, :], goal_count, axis=0)
    a_batch = np.full(goal_count, action_id, dtype=np.int64)
    h_sa, _ = model._embed(s_batch, a_batch)
    h_goal, _ = model._embed(goal_bank, a_batch)
    return float(np.max(np.sum(h_sa * h_goal, axis=1)))


def _run_regime(
    *,
    model: ContrastiveCriticNumpy,
    eval_env_id: str,
    fixed_u: int,
    episodes: int,
    action_ids: list[int],
    goal_bank: np.ndarray,
) -> float:
    successes = 0
    for ep in range(episodes):
        env = gym.make(eval_env_id, render_mode="rgb_array", fixed_u=fixed_u)
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            state = extract_state(obs).astype(np.float64, copy=False)
            scores = [
                _score_action(model, state, action_id, goal_bank)
                for action_id in action_ids
            ]
            action = action_ids[int(np.argmax(scores))]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward > 0:
                successes += 1
            obs = next_obs
    return successes / float(episodes)


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    eval_env_id: str,
    episodes_per_regime: int,
    action_ids: list[int],
    action_subset_label: str,
    goal_bank_size: int,
) -> dict[str, object]:
    model, meta = _load_numpy_checkpoint(checkpoint_path)
    cfg = meta["train_config"]

    goal_bank_u0 = _goal_bank_from_fixed_u(
        env_id=eval_env_id,
        fixed_u=0,
        goal_bank_size=goal_bank_size,
    )
    goal_bank_u1 = _goal_bank_from_fixed_u(
        env_id=eval_env_id,
        fixed_u=1,
        goal_bank_size=goal_bank_size,
    )

    success_u0 = _run_regime(
        model=model,
        eval_env_id=eval_env_id,
        fixed_u=0,
        episodes=episodes_per_regime,
        action_ids=action_ids,
        goal_bank=goal_bank_u0,
    )
    success_u1 = _run_regime(
        model=model,
        eval_env_id=eval_env_id,
        fixed_u=1,
        episodes=episodes_per_regime,
        action_ids=action_ids,
        goal_bank=goal_bank_u1,
    )
    mean_success = 0.5 * (success_u0 + success_u1)
    worst_case_success = min(success_u0, success_u1)
    regime_gap = abs(success_u0 - success_u1)

    return {
        "method": meta["method"],
        "loss_family": meta["loss_family"],
        "checkpoint": str(checkpoint_path),
        "seed": cfg.get("seed"),
        "env_id": cfg.get("env_id"),
        "eval_env_id": eval_env_id,
        "collector_mode": cfg.get("collector_mode"),
        "num_episodes": cfg.get("num_episodes"),
        "num_train_steps": cfg.get("num_train_steps"),
        "w": meta["w"],
        "action_subset": action_subset_label,
        "goal_bank_size": goal_bank_size,
        "episodes_per_regime": episodes_per_regime,
        "success_u0": success_u0,
        "success_u1": success_u1,
        "mean_success": mean_success,
        "worst_case_success": worst_case_success,
        "regime_gap": regime_gap,
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="Forced-U evaluation for HiddenFork checkpoints")
    p.add_argument(
        "--checkpoints",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit checkpoint paths (.npz)",
    )
    p.add_argument(
        "--checkpoint-glob",
        type=str,
        default=None,
        help="Optional glob under checkpoints/, e.g. 'robust_v1_seed0_*.npz'",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_forced_u_eval.csv",
    )
    p.add_argument("--eval-env-id", type=str, default=ENV_CONF)
    p.add_argument("--episodes-per-regime", type=int, default=100)
    p.add_argument(
        "--action-subset",
        type=str,
        default="left,right,forward",
        help="Comma-separated subset from {left,right,forward}",
    )
    p.add_argument("--goal-bank-size", type=int, default=4)
    p.add_argument("--stdout-only", action="store_true")
    args = p.parse_args()

    ckpts: list[Path] = []
    if args.checkpoints:
        ckpts.extend(args.checkpoints)
    if args.checkpoint_glob:
        ckpts.extend(sorted((ROOT / "checkpoints").glob(args.checkpoint_glob)))
    if not ckpts:
        raise SystemExit("No checkpoints provided. Use --checkpoints or --checkpoint-glob.")

    action_names = [part.strip() for part in args.action_subset.split(",") if part.strip()]
    unknown = [name for name in action_names if name not in ACTION_NAME_TO_ID]
    if unknown:
        raise SystemExit(f"Unknown action names: {unknown}")
    action_ids = [ACTION_NAME_TO_ID[name] for name in action_names]
    action_subset_label = ",".join(action_names)

    rows = [
        evaluate_checkpoint(
            checkpoint_path=ckpt,
            eval_env_id=args.eval_env_id,
            episodes_per_regime=args.episodes_per_regime,
            action_ids=action_ids,
            action_subset_label=action_subset_label,
            goal_bank_size=args.goal_bank_size,
        )
        for ckpt in ckpts
    ]

    if not args.stdout_only:
        _write_csv(rows, args.output)

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()
