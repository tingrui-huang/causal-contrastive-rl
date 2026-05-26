"""
Smoke test for the new WindyCorridor environment + expert policy.

Runs:
1. Random policy against the SCM-wrapped env (~1000 steps total).
2. Expert (80/20 NEAR/FAR) against the SCM-wrapped env (~1000 steps total).

Reports:
* route split for the expert
* terminal outcome counts (goal / lava / timeout) per route
* wind direction histogram per cell-type (NEAR row vs FAR/other)
* expert NEAR success rate and number of wait-steps used
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from minigrid.core.actions import Actions

from agents.expert_policy import CorridorExpertPolicy
from envs import make_windy_corridor_scm
from envs.windy_corridor import LETHAL_X
from utils.collector import rollout_scm_episode


def _outcome(traj: list[dict]) -> str:
    if not traj:
        return "empty"
    last = traj[-1]
    if last["reward"] > 0 or (last["terminated"] and last["reward"] >= 0):
        # goal sets reward to 0 (after -0.1 step penalty)... use position check instead
        pass
    # Position-based outcome — goal at (13, 1)
    if last["next_pos"] == (13, 1):
        return "goal"
    if last["terminated"]:
        return "lava"
    if last["truncated"]:
        return "timeout"
    return "other"


def _run_random(n_episodes: int, max_steps: int, seed: int) -> dict:
    env = make_windy_corridor_scm()
    rng = np.random.default_rng(seed)
    outcomes: Counter[str] = Counter()
    steps_total = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        last_pos = tuple(int(v) for v in obs)
        last_terminated = False
        last_truncated = False
        for t in range(max_steps):
            action = int(rng.integers(0, 7))
            obs, reward, terminated, truncated, info = env.step(action)
            last_pos = tuple(int(v) for v in obs)
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            steps_total += 1
            if terminated or truncated:
                break
        if last_pos == (13, 1):
            outcomes["goal"] += 1
        elif last_terminated:
            outcomes["lava"] += 1
        elif last_truncated:
            outcomes["timeout"] += 1
        else:
            outcomes["other"] += 1

    return {"episodes": n_episodes, "steps_total": steps_total, "outcomes": outcomes}


def _run_expert(n_episodes: int, max_steps: int, seed: int) -> dict:
    env = make_windy_corridor_scm()
    expert = CorridorExpertPolicy(p_near=0.8, rng=np.random.default_rng(seed))

    per_route_outcomes: dict[str, Counter[str]] = {
        "near": Counter(),
        "far": Counter(),
    }
    per_route_steps: dict[str, list[int]] = {"near": [], "far": []}
    wait_steps_near = 0
    far_lava_deaths = 0  # MUST be 0
    wind_counts_y1: Counter[int] = Counter()
    wind_counts_far: Counter[int] = Counter()

    for ep in range(n_episodes):
        traj = rollout_scm_episode(env, expert, max_steps=max_steps, seed=seed + ep)
        if not traj:
            continue
        route = traj[0]["route"]
        outcome = _outcome(traj)
        per_route_outcomes[route][outcome] += 1
        per_route_steps[route].append(len(traj))

        for step in traj:
            x, y = step["pos"]
            if y == 1:
                wind_counts_y1[step["wind"]] += 1
            else:
                wind_counts_far[step["wind"]] += 1
            if route == "near" and step["action"] == int(Actions.done):
                wait_steps_near += 1
        if route == "far" and outcome == "lava":
            far_lava_deaths += 1

    return {
        "episodes": n_episodes,
        "per_route_outcomes": per_route_outcomes,
        "per_route_mean_steps": {
            k: (sum(v) / len(v) if v else float("nan"))
            for k, v in per_route_steps.items()
        },
        "wait_steps_near": wait_steps_near,
        "far_lava_deaths": far_lava_deaths,
        "wind_counts_y1": wind_counts_y1,
        "wind_counts_far": wind_counts_far,
    }


def main() -> None:
    print("== Random policy ==")
    rand = _run_random(n_episodes=20, max_steps=200, seed=0)
    print(f"  episodes={rand['episodes']}  steps_total={rand['steps_total']}")
    print(f"  outcomes={dict(rand['outcomes'])}")

    print()
    print("== Expert policy (p_near=0.8) ==")
    exp = _run_expert(n_episodes=100, max_steps=400, seed=0)
    near_n = sum(exp["per_route_outcomes"]["near"].values())
    far_n = sum(exp["per_route_outcomes"]["far"].values())
    print(f"  episodes={exp['episodes']}  route_split  near={near_n}  far={far_n}")
    print(f"  NEAR outcomes: {dict(exp['per_route_outcomes']['near'])}")
    print(f"  FAR  outcomes: {dict(exp['per_route_outcomes']['far'])}")
    print(f"  mean steps  NEAR={exp['per_route_mean_steps']['near']:.1f}"
          f"  FAR={exp['per_route_mean_steps']['far']:.1f}")
    print(f"  wait-steps on NEAR (action=done): {exp['wait_steps_near']}")
    print(f"  FAR lava deaths (MUST be 0): {exp['far_lava_deaths']}")
    print()
    print(f"  wind dist on y=1 cells (NEAR): {dict(exp['wind_counts_y1'])}")
    print(f"  wind dist elsewhere (FAR / connector): {dict(exp['wind_counts_far'])}")

    assert exp["far_lava_deaths"] == 0, "FAR route killed the expert — map bug!"
    print()
    print("OK: FAR route has zero deaths as designed.")
    print(f"   LETHAL_X on NEAR: {sorted(LETHAL_X)}")


if __name__ == "__main__":
    main()
