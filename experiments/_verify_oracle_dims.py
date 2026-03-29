"""Verify that oracle state extraction produces correct U values in ref trajectories."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import gymnasium as gym
import envs  # noqa: F401
from utils.preprocess import extract_state, extract_state_oracle

for fixed_u in (0, 1):
    env = gym.make(
        "CausalContrastive-WindyCorridor-15x15-Lethal-v0",
        render_mode="rgb_array",
        fixed_u=fixed_u,
        wind_strength=0.0,
    )
    obs, info = env.reset(seed=0)

    base = extract_state(obs, info)
    oracle = extract_state_oracle(obs, info)

    print(f"fixed_u={fixed_u}:")
    print(f"  info['confounder'] = {info.get('confounder')}")
    print(f"  env.unwrapped.hidden_u = {env.unwrapped.hidden_u}")
    print(f"  base dim = {base.shape[0]}")
    print(f"  oracle dim = {oracle.shape[0]}")
    print(f"  oracle[-1] (should be {fixed_u}.0) = {oracle[-1]}")
    print(f"  oracle[-4:-1] (x, y, dir) = {oracle[-4:-1]}")

    # Step a few times and check
    for step in range(3):
        obs, _, _, _, info = env.step(2)  # forward
        oracle_s = extract_state_oracle(obs, info)
        print(f"  step {step}: info['confounder']={info.get('confounder')} oracle[-1]={oracle_s[-1]}")

    env.close()
    print()

# Now check what the evaluator's reference trajectory looks like
from experiments.evaluate_windy_corridor_forced_u import _reference_trajectory_from_fixed_u

for fixed_u in (0, 1):
    ref_states, ref_positions = _reference_trajectory_from_fixed_u(
        env_id="CausalContrastive-WindyCorridor-15x15-Lethal-v0",
        fixed_u=fixed_u,
        seed=0,
        state_fn=extract_state_oracle,
    )
    print(f"Reference trajectory fixed_u={fixed_u}:")
    print(f"  ref_states shape = {ref_states.shape}")
    print(f"  ref_states dim = {ref_states.shape[1]}")
    u_values = ref_states[:, -1]
    print(f"  U values in ref: unique={np.unique(u_values)}, all={fixed_u}? {np.all(u_values == fixed_u)}")
    print(f"  first 5 U values: {u_values[:5]}")
    print(f"  last 5 U values: {u_values[-5:]}")
    print()
