"""Probe risky-vs-safe route preferences from contrastive critic checkpoints."""
from __future__ import annotations

import argparse
import csv
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
from minigrid.core.actions import Actions

import envs  # noqa: F401

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from utils.preprocess import extract_state

CSV_FIELDS = [
    "checkpoint",
    "checkpoint_label",
    "anchor_name",
    "anchor_pos",
    "anchor_dir",
    "action",
    "score_risky",
    "score_safe",
    "delta_risky_minus_safe",
]

ACTION_NAME = {
    int(Actions.left): "left",
    int(Actions.right): "right",
    int(Actions.forward): "forward",
}

RISKY_ROUTE = [
    (1, 13),
    (2, 13),
    (3, 13),
    (4, 13),
    (5, 13),
    (6, 13),
    (7, 13),
    (8, 13),
    (9, 13),
    (10, 13),
    (11, 13),
    (11, 12),
    (11, 11),
    (11, 10),
    (11, 9),
    (11, 8),
    (11, 7),
    (11, 6),
    (11, 5),
    (12, 5),
    (13, 5),
]

SAFE_ROUTE = [
    (1, 13),
    (2, 13),
    (3, 13),
    (3, 12),
    (3, 11),
    (3, 10),
    (3, 9),
    (4, 9),
    (5, 9),
    (6, 9),
    (7, 9),
    (7, 8),
    (7, 7),
    (7, 6),
    (7, 5),
    (8, 5),
    (9, 5),
    (10, 5),
    (11, 5),
    (12, 5),
    (13, 5),
]

ANCHORS: dict[str, tuple[tuple[int, int], int]] = {
    "top_split": ((3, 13), 0),
    "top_mid": ((5, 13), 0),
    "risk_turn": ((11, 13), 0),
}

ROUTES_BY_ANCHOR: dict[str, dict[str, list[tuple[int, int]]]] = {
    "top_split": {
        "risky": RISKY_ROUTE[RISKY_ROUTE.index((3, 13)) :],
        "safe": SAFE_ROUTE,
    },
    "top_mid": {
        "risky": RISKY_ROUTE[RISKY_ROUTE.index((5, 13)) :],
        "safe": [
            (5, 13),
            (4, 13),
            (3, 13),
            (3, 12),
            (3, 11),
            (3, 10),
            (3, 9),
            (4, 9),
            (5, 9),
            (6, 9),
            (7, 9),
            (7, 8),
            (7, 7),
            (7, 6),
            (7, 5),
            (8, 5),
            (9, 5),
            (10, 5),
            (11, 5),
            (12, 5),
            (13, 5),
        ],
    },
    "risk_turn": {
        "risky": RISKY_ROUTE[RISKY_ROUTE.index((11, 13)) :],
        "safe": [
            (11, 13),
            (10, 13),
            (9, 13),
            (8, 13),
            (7, 13),
            (6, 13),
            (5, 13),
            (4, 13),
            (3, 13),
            (3, 12),
            (3, 11),
            (3, 10),
            (3, 9),
            (4, 9),
            (5, 9),
            (6, 9),
            (7, 9),
            (7, 8),
            (7, 7),
            (7, 6),
            (7, 5),
            (8, 5),
            (9, 5),
            (10, 5),
            (11, 5),
            (12, 5),
            (13, 5),
        ],
    },
}


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


def _desired_direction(src: tuple[int, int], dst: tuple[int, int]) -> int:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx == 1 and dy == 0:
        return 0
    if dx == 0 and dy == 1:
        return 1
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == -1:
        return 3
    raise ValueError(f"Non-adjacent move from {src} to {dst}")


def _obs_from_pose(env, pos: tuple[int, int], direction: int):
    raw = env.unwrapped
    raw.agent_pos = tuple(int(v) for v in pos)
    raw.agent_dir = int(direction)
    return raw._augment_obs(raw.gen_obs())


def _route_bank(env, route: list[tuple[int, int]]) -> np.ndarray:
    if len(route) < 2:
        raise ValueError("Route tail is too short.")

    states: list[np.ndarray] = []
    prev_dir = _desired_direction(route[0], route[1])
    for idx, cell in enumerate(route[1:], start=1):
        if idx < len(route) - 1:
            direction = _desired_direction(cell, route[idx + 1])
            prev_dir = direction
        else:
            direction = prev_dir
        obs = _obs_from_pose(env, cell, direction)
        states.append(extract_state(obs).astype(np.float64, copy=False))
    return np.stack(states, axis=0)


def _score_bank(
    model: ContrastiveCriticNumpy,
    state: np.ndarray,
    action_id: int,
    bank: np.ndarray,
) -> float:
    batch_size = bank.shape[0]
    s_batch = np.repeat(state[None, :], batch_size, axis=0)
    a_batch = np.full(batch_size, action_id, dtype=np.int64)
    h_sa, _ = model._embed(s_batch, a_batch)
    h_goal, _ = model._embed(bank, a_batch)
    logits = np.sum(h_sa * h_goal, axis=1) / model.tau
    return float(np.max(logits))


def _probe_checkpoint(
    *,
    checkpoint: Path,
    checkpoint_label: str,
    env_id: str,
) -> list[dict[str, Any]]:
    model, _ = _load_numpy_checkpoint(checkpoint)
    env = gym.make(env_id, render_mode="rgb_array", wind_strength=0.0)
    rows: list[dict[str, Any]] = []

    try:
        env.reset(seed=0)
        for anchor_name, (anchor_pos, anchor_dir) in ANCHORS.items():
            anchor_obs = _obs_from_pose(env, anchor_pos, anchor_dir)
            anchor_state = extract_state(anchor_obs).astype(np.float64, copy=False)
            risky_bank = _route_bank(env, ROUTES_BY_ANCHOR[anchor_name]["risky"])
            safe_bank = _route_bank(env, ROUTES_BY_ANCHOR[anchor_name]["safe"])

            for action_id in (int(Actions.left), int(Actions.right), int(Actions.forward)):
                score_risky = _score_bank(model, anchor_state, action_id, risky_bank)
                score_safe = _score_bank(model, anchor_state, action_id, safe_bank)
                rows.append(
                    {
                        "checkpoint": str(checkpoint),
                        "checkpoint_label": checkpoint_label,
                        "anchor_name": anchor_name,
                        "anchor_pos": str(anchor_pos),
                        "anchor_dir": int(anchor_dir),
                        "action": ACTION_NAME[action_id],
                        "score_risky": score_risky,
                        "score_safe": score_safe,
                        "delta_risky_minus_safe": score_risky - score_safe,
                    }
                )
    finally:
        env.close()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe route preferences from critic checkpoints.")
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--confounded-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--env-id",
        type=str,
        default="CausalContrastive-WindyCorridor-15x15-Lethal-Clean-v0",
    )
    parser.add_argument("--stdout-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "probe_critic_values.csv",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    rows.extend(
        _probe_checkpoint(
            checkpoint=args.clean_checkpoint,
            checkpoint_label="clean",
            env_id=args.env_id,
        )
    )
    rows.extend(
        _probe_checkpoint(
            checkpoint=args.confounded_checkpoint,
            checkpoint_label="confounded",
            env_id=args.env_id,
        )
    )

    if not args.stdout_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
