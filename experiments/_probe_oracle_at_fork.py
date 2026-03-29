"""Probe oracle model's preferences at fork points under U=0 and U=1."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import gymnasium as gym
from minigrid.core.actions import Actions

import envs  # noqa: F401
from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from utils.preprocess import extract_state_oracle
import json

ACTION_IDS = {
    "left": int(Actions.left),
    "right": int(Actions.right),
    "forward": int(Actions.forward),
}

RISKY_ROUTE = [
    (1, 13), (2, 13), (3, 13), (4, 13), (5, 13), (6, 13), (7, 13),
    (8, 13), (9, 13), (10, 13), (11, 13), (11, 12), (11, 11), (11, 10),
    (11, 9), (11, 8), (11, 7), (11, 6), (11, 5), (12, 5), (13, 5),
]

SAFE_ROUTE = [
    (1, 13), (2, 13), (3, 13), (3, 12), (3, 11), (3, 10), (3, 9),
    (4, 9), (5, 9), (6, 9), (7, 9), (7, 8), (7, 7), (7, 6), (7, 5),
    (8, 5), (9, 5), (10, 5), (11, 5), (12, 5), (13, 5),
]

FORK_POINTS = {
    "fork_3_13_right": {"pos": (3, 13), "dir": 0},
    "fork_3_13_up": {"pos": (3, 13), "dir": 3},
    "fork_11_13_right": {"pos": (11, 13), "dir": 0},
    "fork_11_13_down": {"pos": (11, 13), "dir": 1},
}


def _load(path):
    ckpt = np.load(path, allow_pickle=True)
    model = ContrastiveCriticNumpy(
        state_dim=int(ckpt["state_dim"]),
        n_actions=int(ckpt["n_actions"]),
        hidden=int(ckpt["W1"].shape[1]),
        emb_dim=int(ckpt["W2"].shape[1]),
        tau=float(ckpt["tau"]),
        seed=0,
    )
    model.W1[...] = ckpt["W1"]
    model.b1[...] = ckpt["b1"]
    model.W2[...] = ckpt["W2"]
    model.b2[...] = ckpt["b2"]
    return model


def _obs_at(env, pos, direction):
    raw = env.unwrapped
    raw.agent_pos = tuple(int(v) for v in pos)
    raw.agent_dir = int(direction)
    return raw._augment_obs(raw.gen_obs())


def _desired_direction(src, dst):
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    if dx == 1 and dy == 0: return 0
    if dx == 0 and dy == 1: return 1
    if dx == -1 and dy == 0: return 2
    if dx == 0 and dy == -1: return 3
    raise ValueError(f"Non-adjacent: {src} -> {dst}")


def _route_bank(env, route, u_value):
    """Build goal bank from route with correct U value."""
    states = []
    prev_dir = _desired_direction(route[0], route[1])
    for idx, cell in enumerate(route[1:], start=1):
        if idx < len(route) - 1:
            direction = _desired_direction(cell, route[idx + 1])
            prev_dir = direction
        else:
            direction = prev_dir
        obs = _obs_at(env, cell, direction)
        info = {"confounder": u_value}
        states.append(extract_state_oracle(obs, info).astype(np.float64))
    return np.stack(states, axis=0)


def _score(model, state, action_id, bank):
    s = np.repeat(state[None, :], bank.shape[0], axis=0)
    a = np.full(bank.shape[0], action_id, dtype=np.int64)
    h_sa, _ = model._embed(s, a)
    h_g, _ = model._embed(bank, a)
    return float(np.mean(np.sum(h_sa * h_g, axis=1) / model.tau))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()

    model = _load(args.checkpoint)
    env = gym.make(
        "CausalContrastive-WindyCorridor-15x15-Lethal-v0",
        render_mode="rgb_array", wind_strength=0.0,
    )
    env.reset(seed=0)

    for u_val in (0, 1):
        print(f"\n{'='*70}")
        print(f"Oracle Probe with U={u_val}")
        print(f"  Expected: U=0 → prefer risky (shortcut), U=1 → prefer safe")
        print(f"{'='*70}")

        risky_bank = _route_bank(env, RISKY_ROUTE, u_val)
        safe_bank = _route_bank(env, SAFE_ROUTE, u_val)

        for name, info in FORK_POINTS.items():
            pos, direction = info["pos"], info["dir"]
            obs = _obs_at(env, pos, direction)
            state = extract_state_oracle(obs, {"confounder": u_val}).astype(np.float64)

            print(f"\n  --- {name}: pos={pos} dir={direction} ---")
            print(f"    state[-1] (U) = {state[-1]}")
            print(f"    {'action':<10} {'→risky':>10} {'→safe':>10} {'delta(r-s)':>12}")

            for aname, aid in ACTION_IDS.items():
                sr = _score(model, state, aid, risky_bank)
                ss = _score(model, state, aid, safe_bank)
                print(f"    {aname:<10} {sr:>10.4f} {ss:>10.4f} {sr - ss:>12.4f}")

            best_risky = max(ACTION_IDS.items(), key=lambda x: _score(model, state, x[1], risky_bank))
            best_safe = max(ACTION_IDS.items(), key=lambda x: _score(model, state, x[1], safe_bank))
            print(f"    best toward risky: {best_risky[0]}")
            print(f"    best toward safe:  {best_safe[0]}")

    env.close()


if __name__ == "__main__":
    main()
