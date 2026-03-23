"""
Linear-probe diagnostics for HiddenFork robust-v1 models.

Goal:
- Train robust-v1 models for one or more weights
- Build a **pre-divergence paired dataset** from fixed_u=0 / fixed_u=1 rollouts
- Measure how much hidden U can be predicted from:
  1) action only
  2) learned embedding h(s, a)

If embedding accuracy stays high above action-only accuracy on pre-divergence states,
the representation may still encode confounder information.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs  # noqa: F401  # register ids

from configs.training_defaults import (
    ROBUST_V1_COLLECTOR_MODE,
    ROBUST_V1_LOGIT_SCALE,
    ROBUST_V1_M,
    ROBUST_V1_NUM_EPISODES,
    ROBUST_V1_ORACLE_EPSILON,
    ROBUST_V1_P,
    ROBUST_V1_SWEEP_WEIGHTS,
    ROBUST_V1_W,
    TRAIN_NUMPY_LR,
    TRAIN_NUM_STEPS,
)
from experiments.train_contrastive_robust_v1 import train_numpy_robust_v1
from utils.offline_data import build_policy
from utils.collector import rollout_episode
from utils.preprocess import extract_state

ENV_CONF = "CausalContrastive-HiddenFork-15x15-v0"

CSV_FIELDS = [
    "weight",
    "probe_env_id",
    "collector_mode",
    "oracle_epsilon",
    "num_train_steps",
    "num_train_episodes",
    "num_probe_episode_pairs",
    "num_probe_samples",
    "action_only_acc",
    "embedding_acc",
]


def _rollout_fixed_u_pair(
    *,
    seed: int,
    env_id: str,
    collector_mode: str,
    oracle_epsilon: float,
) -> tuple[list[dict], list[dict]]:
    pair = []
    for fixed_u in (0, 1):
        env = gym.make(env_id, render_mode="rgb_array", fixed_u=fixed_u)
        env.action_space.seed(seed)
        policy = build_policy(
            env,
            collector_mode=collector_mode,
            oracle_epsilon=oracle_epsilon,
            seed=seed,
        )
        traj = rollout_episode(env, policy, max_steps=500, seed=seed)
        pair.append(traj)
    return pair[0], pair[1]


def _collect_predivergence_probe_dataset(
    *,
    num_episode_pairs: int,
    env_id: str,
    collector_mode: str,
    oracle_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[int] = []
    labels: list[int] = []

    for seed in range(num_episode_pairs):
        traj_u0, traj_u1 = _rollout_fixed_u_pair(
            seed=seed,
            env_id=env_id,
            collector_mode=collector_mode,
            oracle_epsilon=oracle_epsilon,
        )
        T = min(len(traj_u0), len(traj_u1))
        for t in range(T):
            o0 = traj_u0[t]["obs"]
            o1 = traj_u1[t]["obs"]
            if not np.array_equal(o0["image"], o1["image"]):
                break
            states.append(extract_state(o0))
            actions.append(int(traj_u0[t]["action"]))
            labels.append(0)

            states.append(extract_state(o1))
            actions.append(int(traj_u1[t]["action"]))
            labels.append(1)

    return (
        np.asarray(states, dtype=np.float64),
        np.asarray(actions, dtype=np.int64),
        np.asarray(labels, dtype=np.float64),
    )


def _split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = max(1, int(0.7 * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    if test_idx.size == 0:
        test_idx = train_idx
    return train_idx, test_idx


def _fit_linear_probe(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    train_idx, test_idx = _split_indices(len(X), seed)
    Xtr = X[train_idx]
    ytr = y[train_idx]
    Xte = X[test_idx]
    yte = y[test_idx]

    # Standardize features using training stats.
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    Xtr_b = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], axis=1)
    Xte_b = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], axis=1)

    lam = 1e-3
    A = Xtr_b.T @ Xtr_b + lam * np.eye(Xtr_b.shape[1], dtype=np.float64)
    b = Xtr_b.T @ ytr
    w = np.linalg.solve(A, b)
    pred = (Xte_b @ w >= 0.5).astype(np.float64)
    return float(np.mean(pred == yte))


def _action_features(actions: np.ndarray, n_actions: int) -> np.ndarray:
    X = np.zeros((actions.shape[0], n_actions), dtype=np.float64)
    X[np.arange(actions.shape[0]), actions] = 1.0
    return X


def run_probe(
    *,
    env_id: str,
    weights: list[float],
    collector_mode: str,
    oracle_epsilon: float,
    num_train_episodes: int,
    num_train_steps: int,
    lr: float,
    P: int,
    M: int,
    logit_scale: float,
    num_probe_episode_pairs: int,
) -> list[dict[str, object]]:
    probe_states, probe_actions, probe_labels = _collect_predivergence_probe_dataset(
        num_episode_pairs=num_probe_episode_pairs,
        env_id=env_id,
        collector_mode=collector_mode,
        oracle_epsilon=oracle_epsilon,
    )

    rows: list[dict[str, object]] = []
    for w in weights:
        result = train_numpy_robust_v1(
            seed=0,
            env_id=env_id,
            num_episodes=num_train_episodes,
            num_steps=num_train_steps,
            lr=lr,
            P=P,
            M=M,
            w=w,
            collector_mode=collector_mode,
            oracle_epsilon=oracle_epsilon,
            logit_scale=logit_scale,
            return_model=True,
            verbose=False,
        )
        model = result["model"]

        emb, _ = model._embed(probe_states, probe_actions)
        n_actions = int(model.n_actions)
        X_act = _action_features(probe_actions, n_actions)

        rows.append(
            {
                "weight": w,
                "probe_env_id": env_id,
                "collector_mode": collector_mode,
                "oracle_epsilon": oracle_epsilon,
                "num_train_steps": num_train_steps,
                "num_train_episodes": num_train_episodes,
                "num_probe_episode_pairs": num_probe_episode_pairs,
                "num_probe_samples": int(probe_labels.shape[0]),
                "action_only_acc": _fit_linear_probe(X_act, probe_labels, seed=123),
                "embedding_acc": _fit_linear_probe(emb, probe_labels, seed=123),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser(description="Linear probe for hidden U on robust-v1 embeddings")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "results" / "hidden_fork_probe_robust_v1.csv",
    )
    p.add_argument("--env-id", type=str, default=ENV_CONF)
    p.add_argument("--num-probe-episode-pairs", type=int, default=50)
    p.add_argument("--num-episodes", type=int, default=ROBUST_V1_NUM_EPISODES)
    p.add_argument("--num-steps", type=int, default=TRAIN_NUM_STEPS)
    p.add_argument("--lr", type=float, default=TRAIN_NUMPY_LR)
    p.add_argument("--P", type=int, default=ROBUST_V1_P)
    p.add_argument("--M", type=int, default=ROBUST_V1_M)
    p.add_argument("--logit-scale", type=float, default=ROBUST_V1_LOGIT_SCALE)
    p.add_argument(
        "--collector-mode",
        type=str,
        default=ROBUST_V1_COLLECTOR_MODE,
        choices=["random", "oracle_eps"],
    )
    p.add_argument("--oracle-epsilon", type=float, default=ROBUST_V1_ORACLE_EPSILON)
    p.add_argument("--w", type=float, default=None)
    p.add_argument("--weights", type=float, nargs="+")
    p.add_argument("--stdout-only", action="store_true")
    args = p.parse_args()

    if args.weights is not None:
        weights = list(args.weights)
    elif args.w is not None:
        weights = [args.w]
    else:
        weights = list(ROBUST_V1_SWEEP_WEIGHTS)

    rows = run_probe(
        env_id=args.env_id,
        weights=weights,
        collector_mode=args.collector_mode,
        oracle_epsilon=args.oracle_epsilon,
        num_train_episodes=args.num_episodes,
        num_train_steps=args.num_steps,
        lr=args.lr,
        P=args.P,
        M=args.M,
        logit_scale=args.logit_scale,
        num_probe_episode_pairs=args.num_probe_episode_pairs,
    )

    if not args.stdout_only:
        _write_csv(rows, args.output)

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


if __name__ == "__main__":
    main()

