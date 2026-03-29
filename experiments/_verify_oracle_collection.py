"""Verify oracle success rate under both regimes in Lethal environment."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import envs  # noqa: F401
from utils.offline_data import build_oracle_eps_policy
from utils.collector import rollout_episode

ENV_IDS = [
    "CausalContrastive-WindyCorridor-15x15-Lethal-v0",
    "CausalContrastive-WindyCorridor-15x15-Lethal-Clean-v0",
]
N_EPISODES = 50
MAX_STEPS = 200


def test_env(env_id: str):
    print(f"\n{'='*60}")
    print(f"Testing: {env_id}")
    print(f"{'='*60}")

    u0_success = 0
    u1_success = 0
    u0_total = 0
    u1_total = 0

    for ep in range(N_EPISODES):
        seed = 42 + ep
        env = gym.make(env_id, render_mode="rgb_array")
        env.action_space.seed(seed)
        policy = build_oracle_eps_policy(env, epsilon=0.05, seed=seed)
        traj = rollout_episode(env, policy, max_steps=MAX_STEPS, seed=seed)

        u = int(env.unwrapped.hidden_u)
        success = any(t["reward"] > 0 for t in traj)

        if u == 0:
            u0_total += 1
            if success:
                u0_success += 1
        else:
            u1_total += 1
            if success:
                u1_success += 1

        env.close()

    u0_rate = u0_success / max(u0_total, 1)
    u1_rate = u1_success / max(u1_total, 1)
    print(f"  u=0: {u0_success}/{u0_total} = {u0_rate:.2%}")
    print(f"  u=1: {u1_success}/{u1_total} = {u1_rate:.2%}")
    print(f"  total: {u0_success + u1_success}/{N_EPISODES} = {(u0_success + u1_success)/N_EPISODES:.2%}")
    return u0_rate, u1_rate


for eid in ENV_IDS:
    test_env(eid)
