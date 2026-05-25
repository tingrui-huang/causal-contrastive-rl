import gymnasium as gym
from causal_gym.envs import WindyMiniGridPCH

env = WindyMiniGridPCH(gym.make('MiniGrid-LavaGapS6-v0', render_mode='rgb_array'))
obs, info = env.reset(seed=42)
print("OK:", obs, info)
