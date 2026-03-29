"""Verify oracle success rate with epsilon=0 to isolate wind effects."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import envs  # noqa: F401
from utils.offline_data import build_oracle_eps_policy
from utils.collector import rollout_episode

N_EPISODES = 100
MAX_STEPS = 200


def test_regime(env_id, fixed_u, epsilon):
    successes = 0
    lava_deaths = 0
    timeouts = 0
    for ep in range(N_EPISODES):
        seed = 42 + ep
        env = gym.make(env_id, render_mode="rgb_array", fixed_u=fixed_u)
        env.action_space.seed(seed)
        policy = build_oracle_eps_policy(env, epsilon=epsilon, seed=seed)
        traj = rollout_episode(env, policy, max_steps=MAX_STEPS, seed=seed)

        success = any(t["reward"] > 0 for t in traj)
        terminated_no_reward = traj[-1]["done"] and not success and len(traj) < MAX_STEPS
        if success:
            successes += 1
        elif terminated_no_reward:
            lava_deaths += 1
        else:
            timeouts += 1
        env.close()

    print(f"  fixed_u={fixed_u}, eps={epsilon}: "
          f"success={successes}/{N_EPISODES} ({successes/N_EPISODES:.0%}), "
          f"lava={lava_deaths}, timeout={timeouts}")


env_id = "CausalContrastive-WindyCorridor-15x15-Lethal-v0"
print(f"Testing: {env_id}")
print("--- epsilon=0 (pure oracle) ---")
test_regime(env_id, 0, 0.0)
test_regime(env_id, 1, 0.0)
print("--- epsilon=0.05 ---")
test_regime(env_id, 0, 0.05)
test_regime(env_id, 1, 0.05)
print("--- epsilon=0.10 ---")
test_regime(env_id, 0, 0.10)
test_regime(env_id, 1, 0.10)
