def rollout_episode(env, policy_fn, max_steps=100, seed=None):
    """Roll out one episode. If ``seed`` is set, it is passed to ``env.reset``."""
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    trajectory = []

    for t in range(max_steps):
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": terminated or truncated,
        })

        obs = next_obs

        if terminated or truncated:
            break

    return trajectory