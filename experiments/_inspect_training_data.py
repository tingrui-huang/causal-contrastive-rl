"""Inspect the training data distribution: what routes does each regime take?"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np

import envs  # noqa: F401
from utils.offline_data import build_oracle_eps_policy
from utils.collector import rollout_episode

ENV_ID = "CausalContrastive-WindyCorridor-15x15-Lethal-v0"
N_EPISODES = 50
MAX_STEPS = 500
SEED = 0
EPSILON = 0.2


def classify_route(trajectory, env):
    """Check if trajectory goes through x=11 vertical passage (risky) or x=3 (safe)."""
    positions = []
    raw = env.unwrapped
    # Reconstruct positions from obs
    for tr in trajectory:
        obs = tr["obs"]
        if isinstance(obs, dict) and "agent_pos" in obs:
            positions.append((int(obs["agent_pos"][0]), int(obs["agent_pos"][1])))

    went_through_x11 = any(x == 11 and 6 <= y <= 12 for x, y in positions)
    went_through_x3_up = any(x == 3 and 9 <= y <= 12 for x, y in positions)
    success = any(tr["reward"] > 0 for tr in trajectory)

    return {
        "risky": went_through_x11,
        "safe": went_through_x3_up,
        "success": success,
        "length": len(trajectory),
    }


u0_stats = {"total": 0, "success": 0, "risky": 0, "safe": 0, "lengths": []}
u1_stats = {"total": 0, "success": 0, "risky": 0, "safe": 0, "lengths": []}

for ep in range(N_EPISODES):
    ep_seed = SEED + ep
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env.action_space.seed(ep_seed)
    policy = build_oracle_eps_policy(env, epsilon=EPSILON, seed=ep_seed)
    traj = rollout_episode(env, policy, max_steps=MAX_STEPS, seed=ep_seed)

    u = int(env.unwrapped.hidden_u)
    info = classify_route(traj, env)

    stats = u0_stats if u == 0 else u1_stats
    stats["total"] += 1
    if info["success"]:
        stats["success"] += 1
    if info["risky"]:
        stats["risky"] += 1
    if info["safe"]:
        stats["safe"] += 1
    stats["lengths"].append(info["length"])

    env.close()

for label, stats in [("U=0", u0_stats), ("U=1", u1_stats)]:
    n = stats["total"]
    if n == 0:
        print(f"{label}: no episodes")
        continue
    print(f"{label}: {n} episodes")
    print(f"  success: {stats['success']}/{n} = {stats['success']/n:.0%}")
    print(f"  risky route: {stats['risky']}/{n} = {stats['risky']/n:.0%}")
    print(f"  safe route:  {stats['safe']}/{n} = {stats['safe']/n:.0%}")
    print(f"  mean length: {np.mean(stats['lengths']):.1f}")
    print()

# Check anchor distribution: how many transitions are from each regime?
print(f"Total transitions: U=0={sum(u0_stats['lengths'])}, U=1={sum(u1_stats['lengths'])}")
print(f"Ratio: U=0 {sum(u0_stats['lengths'])/max(1, sum(u0_stats['lengths'])+sum(u1_stats['lengths'])):.0%}")
