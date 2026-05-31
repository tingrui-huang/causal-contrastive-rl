def rollout_episode(env, policy_fn, max_steps=100, seed=None):
    """Roll out one episode. If ``seed`` is set, it is passed to ``env.reset``."""
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    trajectory = []

    for t in range(max_steps):
        action = policy_fn(obs)
        prev_info = info
        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": terminated or truncated,
            "info": prev_info,
            "next_info": info,
        })

        obs = next_obs

        if terminated or truncated:
            break

    return trajectory


def rollout_scm_episode(scm_env, expert, max_steps=200, seed=None):
    """
    Roll out one episode against a causal_gym WindyMiniGridSCM.

    ``expert`` is a ``CorridorExpertPolicy``-style object: must support
    ``expert.reset(seed=...)`` and ``expert(pos, wind_dir) -> int``. Each
    transition records position, agent direction (tracked by the expert),
    chosen action, reward, wind direction observed when the action was chosen,
    and the route the expert committed to at episode start.
    """
    obs, info = scm_env.reset(seed=seed)
    route = expert.reset(seed=seed, agent_dir=int(scm_env.agent_dir))

    trajectory = []
    for t in range(max_steps):
        pos = tuple(int(v) for v in obs)
        wind = int(info["wind"])
        agent_dir = int(expert.agent_dir)
        action = expert(pos, wind)

        next_obs, reward, terminated, truncated, next_info = scm_env.step(action)

        trajectory.append({
            "pos": pos,
            "agent_dir": agent_dir,
            "wind": wind,
            "action": int(action),
            "reward": float(reward),
            "next_pos": tuple(int(v) for v in next_obs),
            "next_wind": int(next_info["wind"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "route": route,
        })

        obs = next_obs
        info = next_info
        if terminated or truncated:
            break

    return trajectory