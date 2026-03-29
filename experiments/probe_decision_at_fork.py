"""
At the critical fork point (3,13), compare how clean vs confounded critics
score each action when the goal is simply the final state at (13,5).

This removes all evaluator/planner/waypoint influence — it's a pure
measurement of what the critic "wants" to do at the decision point.

We test two goal-bank configurations:
  A) goal = final state at (13,5) — "which direction does the critic pull?"
  B) goal = next 3 cells on risky route vs safe route — "local preference"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions

import envs  # noqa: F401
from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from utils.preprocess import extract_state

ACTION_IDS = {
    "left": int(Actions.left),
    "right": int(Actions.right),
    "forward": int(Actions.forward),
}

FORK_POINTS = {
    "fork_3_13": {"pos": (3, 13), "dir": 0, "note": "first divergence: right=risky, up=safe"},
    "fork_7_13": {"pos": (7, 13), "dir": 0, "note": "mid corridor: still heading right toward risky"},
    "fork_11_13": {"pos": (11, 13), "dir": 0, "note": "at risky turn: down=risky x=11 passage"},
    "fork_3_13_up": {"pos": (3, 13), "dir": 3, "note": "facing up at fork: forward=safe route"},
    "fork_11_13_down": {"pos": (11, 13), "dir": 1, "note": "facing down at x=11: forward=risky descent"},
}


def _load(path: Path) -> ContrastiveCriticNumpy:
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


def _score(model, state, action_id, goal_states):
    s = np.repeat(state[None, :], goal_states.shape[0], axis=0)
    a = np.full(goal_states.shape[0], action_id, dtype=np.int64)
    h_sa, _ = model._embed(s, a)
    h_g, _ = model._embed(goal_states, a)
    return float(np.mean(np.sum(h_sa * h_g, axis=1) / model.tau))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--clean-checkpoint", type=Path, required=True)
    p.add_argument("--confounded-checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str,
                   default="CausalContrastive-WindyCorridor-15x15-Lethal-v0")
    args = p.parse_args()

    clean_model = _load(args.clean_checkpoint)
    conf_model = _load(args.confounded_checkpoint)

    env = gym.make(args.env_id, render_mode="rgb_array", wind_strength=0.0)
    env.reset(seed=0)

    goal_obs = _obs_at(env, (13, 5), 0)
    goal_state = extract_state(goal_obs).astype(np.float64)[None, :]

    print("=" * 80)
    print("DECISION-LEVEL PROBE: clean vs confounded at fork points")
    print("Goal bank: single final state at (13,5)")
    print("=" * 80)

    for name, info in FORK_POINTS.items():
        pos, direction = info["pos"], info["dir"]
        obs = _obs_at(env, pos, direction)
        state = extract_state(obs).astype(np.float64)

        print(f"\n--- {name}: pos={pos} dir={direction} ({info['note']}) ---")
        print(f"  {'action':<10} {'clean_score':>12} {'conf_score':>12} {'delta(conf-clean)':>18}")

        for aname, aid in ACTION_IDS.items():
            cs = _score(clean_model, state, aid, goal_state)
            fs = _score(conf_model, state, aid, goal_state)
            delta = fs - cs
            print(f"  {aname:<10} {cs:>12.4f} {fs:>12.4f} {delta:>18.4f}")

        clean_best = max(ACTION_IDS.items(), key=lambda x: _score(clean_model, state, x[1], goal_state))
        conf_best = max(ACTION_IDS.items(), key=lambda x: _score(conf_model, state, x[1], goal_state))
        print(f"  >> clean picks: {clean_best[0]}")
        print(f"  >> confounded picks: {conf_best[0]}")
        if clean_best[0] != conf_best[0]:
            print(f"  ** DIVERGENCE: different action choices! **")

    # Also probe with local waypoint banks (next 3 cells on each route)
    risky_next_from_3_13 = [(4, 13), (5, 13), (6, 13)]
    safe_next_from_3_13 = [(3, 12), (3, 11), (3, 10)]

    print("\n" + "=" * 80)
    print("LOCAL WAYPOINT PROBE at (3,13) facing right")
    print("  risky_bank = next 3 cells on risky route (4,13)(5,13)(6,13)")
    print("  safe_bank  = next 3 cells on safe route  (3,12)(3,11)(3,10)")
    print("=" * 80)

    obs_fork = _obs_at(env, (3, 13), 0)
    state_fork = extract_state(obs_fork).astype(np.float64)

    risky_bank = np.stack([
        extract_state(_obs_at(env, c, 0)).astype(np.float64) for c in risky_next_from_3_13
    ])
    safe_bank = np.stack([
        extract_state(_obs_at(env, c, 3)).astype(np.float64) for c in safe_next_from_3_13
    ])

    print(f"\n  {'action':<10} {'clean→risky':>12} {'clean→safe':>12} {'conf→risky':>12} {'conf→safe':>12}")
    for aname, aid in ACTION_IDS.items():
        cr = _score(clean_model, state_fork, aid, risky_bank)
        cs = _score(clean_model, state_fork, aid, safe_bank)
        fr = _score(conf_model, state_fork, aid, risky_bank)
        fs = _score(conf_model, state_fork, aid, safe_bank)
        print(f"  {aname:<10} {cr:>12.4f} {cs:>12.4f} {fr:>12.4f} {fs:>12.4f}")

    env.close()


if __name__ == "__main__":
    main()
