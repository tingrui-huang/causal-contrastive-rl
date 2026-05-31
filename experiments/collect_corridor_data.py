"""
Collect offline trajectories from the WindyCorridor environment using
``CorridorExpertPolicy`` (80% NEAR / 20% FAR, with wait-for-wind on lethal cells).

Output format (`.npz` with object arrays)
------------------------------------------
Per-episode arrays (object arrays of length ``n_episodes``):
  * ``episodes_states``   : (T_i + 1, 3) float32 — [x, y, agent_dir], one row per
                            time step plus the terminal state
  * ``episodes_actions``  : (T_i,) int64 — MiniGrid Actions enum
  * ``episodes_rewards``  : (T_i,) float32
  * ``episodes_winds``    : (T_i,) int64 — latent wind direction at each step
                            (the confounder U, hidden from the trained agent)
  * ``episodes_routes``   : str "near" / "far"
  * ``episodes_outcomes`` : str "goal" / "lava" / "timeout"

Scalars / fixed:
  * ``goal_state`` : (3,) float32 — [13, 1, 0], canonical goal vector for the
                     fixed-goal contrastive critic
  * ``state_dim``  : 3
  * ``n_actions``  : 7 (full MiniGrid action space; expert only uses left, right,
                     forward, done — others are unused but kept for one-hot
                     compatibility with ContrastiveCriticNumpy)
  * ``n_episodes`` : N
  * ``p_near``     : behavior policy parameter used at collection time

Usage
-----
::
    python experiments/collect_corridor_data.py \\
        --num-episodes 1000 --seed 0 \\
        --output data/corridor_expert_n1000.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from agents.expert_policy import CorridorExpertPolicy
from configs import corridor_defaults as C
from envs import make_windy_corridor_scm
from envs.windy_corridor import GOAL_POS


N_ACTIONS = 7
STATE_DIM = 3
GOAL_STATE_DIR = 0  # canonical agent_dir at goal (east); episode terminates on arrival


def _rollout_episode(env, expert, *, max_steps: int, env_seed: int, expert_seed: int) -> dict:
    """One episode → per-step arrays. ``states`` has length ``T+1`` (includes final)."""
    obs, info = env.reset(seed=env_seed)
    expert.reset(seed=expert_seed, agent_dir=int(env.agent_dir))

    states: list[list[int]] = []
    actions: list[int] = []
    rewards: list[float] = []
    winds: list[int] = []

    terminated = False
    truncated = False
    for _ in range(max_steps):
        pos = tuple(int(v) for v in obs)
        agent_dir = int(expert.agent_dir)
        wind = int(info["wind"])

        states.append([pos[0], pos[1], agent_dir])
        winds.append(wind)

        action = expert(pos, wind)
        obs, reward, terminated, truncated, info = env.step(action)

        actions.append(int(action))
        rewards.append(float(reward))

        if terminated or truncated:
            break

    # Append the final (post-terminal or post-truncation) state.
    # We read agent_dir directly from the env to capture any mid-sequence wind
    # drift that ended in lava — expert's tracked dir would be stale here.
    final_pos = tuple(int(v) for v in obs)
    final_dir = int(env.agent_dir)
    states.append([final_pos[0], final_pos[1], final_dir])

    if tuple(states[-1][:2]) == GOAL_POS:
        outcome = "goal"
    elif terminated:
        outcome = "lava"
    elif truncated:
        outcome = "timeout"
    else:
        outcome = "incomplete"

    return {
        "states": np.array(states, dtype=np.float32),
        "actions": np.array(actions, dtype=np.int64),
        "rewards": np.array(rewards, dtype=np.float32),
        "winds": np.array(winds, dtype=np.int64),
        "route": expert.route or "unknown",
        "outcome": outcome,
    }


def collect(
    *,
    num_episodes: int,
    max_steps: int,
    seed: int,
    p_near: float,
    verbose: bool = True,
) -> dict:
    env = make_windy_corridor_scm()
    expert = CorridorExpertPolicy(p_near=p_near)

    states_list = []
    actions_list = []
    rewards_list = []
    winds_list = []
    routes: list[str] = []
    outcomes: list[str] = []

    outcome_counter: Counter[str] = Counter()
    route_counter: Counter[str] = Counter()
    route_outcome: dict[str, Counter[str]] = {"near": Counter(), "far": Counter()}

    for ep in range(num_episodes):
        traj = _rollout_episode(
            env,
            expert,
            max_steps=max_steps,
            env_seed=seed + ep,
            expert_seed=seed + ep + 1_000_000,
        )
        states_list.append(traj["states"])
        actions_list.append(traj["actions"])
        rewards_list.append(traj["rewards"])
        winds_list.append(traj["winds"])
        routes.append(traj["route"])
        outcomes.append(traj["outcome"])

        outcome_counter[traj["outcome"]] += 1
        route_counter[traj["route"]] += 1
        if traj["route"] in route_outcome:
            route_outcome[traj["route"]][traj["outcome"]] += 1

    total_transitions = sum(len(a) for a in actions_list)

    if verbose:
        print(f"Collected {num_episodes} episodes  ({total_transitions} transitions)")
        print(f"  routes:   {dict(route_counter)}")
        print(f"  outcomes: {dict(outcome_counter)}")
        for r, c in route_outcome.items():
            print(f"    {r:<4}: {dict(c)}")

    return {
        "states_list": states_list,
        "actions_list": actions_list,
        "rewards_list": rewards_list,
        "winds_list": winds_list,
        "routes": routes,
        "outcomes": outcomes,
        "total_transitions": total_transitions,
    }


def save_npz(out: dict, output_path: Path, *, p_near: float, num_episodes: int) -> None:
    goal_state = np.array([GOAL_POS[0], GOAL_POS[1], GOAL_STATE_DIR], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        episodes_states=np.array(out["states_list"], dtype=object),
        episodes_actions=np.array(out["actions_list"], dtype=object),
        episodes_rewards=np.array(out["rewards_list"], dtype=object),
        episodes_winds=np.array(out["winds_list"], dtype=object),
        episodes_routes=np.array(out["routes"], dtype=object),
        episodes_outcomes=np.array(out["outcomes"], dtype=object),
        goal_state=goal_state,
        state_dim=np.array(STATE_DIM),
        n_actions=np.array(N_ACTIONS),
        n_episodes=np.array(num_episodes),
        p_near=np.array(p_near, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect WindyCorridor offline data.")
    parser.add_argument("--num-episodes", type=int, default=C.NUM_EPISODES)
    parser.add_argument("--max-steps", type=int, default=C.MAX_STEPS)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--p-near", type=float, default=C.P_NEAR)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "corridor_expert_n1000.npz",
    )
    args = parser.parse_args()

    out = collect(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        p_near=args.p_near,
    )
    save_npz(out, args.output, p_near=args.p_near, num_episodes=args.num_episodes)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
