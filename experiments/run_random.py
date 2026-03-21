import gymnasium as gym
import minigrid

def random_policy(env):
    return lambda obs: env.action_space.sample()

def main():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")

    obs, info = env.reset(seed=42)
    print("Initial observation keys:", obs.keys() if isinstance(obs, dict) else type(obs))

    done = False
    step = 0
    total_reward = 0

    while not done and step < 20:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        print(f"step={step}, action={action}, reward={reward}, done={done}")

    print("Episode finished.")
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()