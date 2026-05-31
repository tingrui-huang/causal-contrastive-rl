"""
Critic-greedy eval: bypass the BC-anchored actor entirely and pick actions by
``argmax_a C(s, a, goal_state)`` at each step.

This isolates the causal-pessimistic training's effect on the *critic* from the
actor-extraction step's BC regularisation. Useful as a diagnostic for whether
the worst-case mechanism correctly shifted the critic's preference.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from minigrid.core.actions import Actions

from agents.contrastive_critic_numpy import ContrastiveCriticNumpy
from envs import make_windy_corridor_scm
from envs.windy_corridor import GOAL_POS

VALID_ACTIONS = [int(Actions.left), int(Actions.right), int(Actions.forward), int(Actions.done)]

FORCED_WIND_PRESETS = {
    "south":    (0.0, 1.0, 0.0, 0.0, 0.0),
    "still":    (0.0, 0.0, 0.0, 0.0, 1.0),
    "notebook": (0.1, 0.1, 0.1, 0.1, 0.6),
}


def _load_critic(path: Path):
    ckpt = np.load(path, allow_pickle=True)
    cfg = json.loads(str(ckpt["config_json"]))
    critic = ContrastiveCriticNumpy(
        state_dim=cfg["state_dim"], n_actions=cfg["n_actions"],
        hidden=cfg["hidden"], emb_dim=cfg["emb_dim"], tau=cfg["tau"], seed=cfg["seed"],
    )
    critic.sa_W1[...] = ckpt["sa_W1"]; critic.sa_b1[...] = ckpt["sa_b1"]
    critic.sa_W2[...] = ckpt["sa_W2"]; critic.sa_b2[...] = ckpt["sa_b2"]
    critic.g_W1[...]  = ckpt["g_W1"];  critic.g_b1[...]  = ckpt["g_b1"]
    critic.g_W2[...]  = ckpt["g_W2"];  critic.g_b2[...]  = ckpt["g_b2"]
    goal_state = np.asarray(ckpt["goal_state"], dtype=np.float64)
    return critic, goal_state


def _critic_greedy_action(critic, state: np.ndarray, goal: np.ndarray) -> int:
    scores = critic.score_actions(state[None], goal[None])[0]
    best = VALID_ACTIONS[0]
    best_val = scores[best]
    for a in VALID_ACTIONS[1:]:
        if scores[a] > best_val:
            best_val = scores[a]; best = a
    return int(best)


def _state(obs, env) -> np.ndarray:
    return np.array([int(obs[0]), int(obs[1]), int(env.agent_dir)], dtype=np.float64)


def _classify_route(visited_ys: set[int]) -> str:
    return "far" if 13 in visited_ys else "near"


def _run_regime(critic, goal_state, num_episodes, max_steps, seed, forced_wind, verbose):
    wind_dist = FORCED_WIND_PRESETS[forced_wind] if forced_wind else None
    outcomes: Counter[str] = Counter()
    route_outcome = {"near": Counter(), "far": Counter()}
    steps_list = []

    for ep in range(num_episodes):
        env = make_windy_corridor_scm(wind_dist=wind_dist) if wind_dist else make_windy_corridor_scm()
        obs, info = env.reset(seed=seed + ep)
        visited_ys: set[int] = set()
        terminated = False
        truncated = False
        n_steps = 0

        for _ in range(max_steps):
            state = _state(obs, env)
            visited_ys.add(int(state[1]))
            action = _critic_greedy_action(critic, state, goal_state)
            obs, _, terminated, truncated, _ = env.step(action)
            n_steps += 1
            if terminated or truncated:
                break

        final_pos = (int(obs[0]), int(obs[1]))
        if final_pos == GOAL_POS:
            outcome = "goal"
        elif terminated:
            outcome = "lava"
        elif truncated:
            outcome = "timeout"
        else:
            outcome = "incomplete"
        outcomes[outcome] += 1
        route_outcome[_classify_route(visited_ys)][outcome] += 1
        steps_list.append(n_steps)

    if verbose:
        far_count = sum(route_outcome["far"].values())
        print(f"  success={outcomes['goal']/num_episodes:.2%}  "
              f"lava={outcomes['lava']/num_episodes:.2%}  "
              f"timeout={outcomes.get('timeout',0)/num_episodes:.2%}  "
              f"incomplete={outcomes.get('incomplete',0)/num_episodes:.2%}")
        print(f"  FAR rate: {far_count}/{num_episodes} = {far_count/num_episodes:.0%}  "
              f"mean_steps={np.mean(steps_list):.1f}")
        print(f"  near: {dict(route_outcome['near'])}")
        print(f"  far : {dict(route_outcome['far'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    critic, goal_state = _load_critic(args.checkpoint)
    print(f"[Critic-greedy eval] {args.checkpoint.name}  N={args.num_episodes}\n")
    for regime in ["notebook", "south", "still"]:
        print(f"[regime={regime}]")
        _run_regime(
            critic, goal_state,
            num_episodes=args.num_episodes, max_steps=args.max_steps,
            seed=args.seed,
            forced_wind=regime if regime != "notebook" else None,
            verbose=True,
        )
        print()


if __name__ == "__main__":
    main()
