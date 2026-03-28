"""Phase W1/W2 wind smoke tests."""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from minigrid.core.actions import Actions

from envs.windy_corridor import WindyCorridorEnv


def _neighbors(cell: tuple[int, int], walkable: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = cell
    out: list[tuple[int, int]] = []
    for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if nxt in walkable:
            out.append(nxt)
    return out


def _shortest_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    walkable: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    q: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        cell = q.popleft()
        if cell == goal:
            break
        for nxt in _neighbors(cell, walkable):
            if nxt not in parent:
                parent[nxt] = cell
                q.append(nxt)

    if goal not in parent:
        raise RuntimeError("goal is unreachable")

    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _desired_direction(src: tuple[int, int], dst: tuple[int, int]) -> int:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if dx == 1 and dy == 0:
        return 0
    if dx == 0 and dy == 1:
        return 1
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == -1:
        return 3
    raise ValueError(f"Non-adjacent move from {src} to {dst}")


def _scripted_action(env: WindyCorridorEnv) -> Actions:
    pos = tuple(int(v) for v in env.agent_pos)
    goal = env.goal_pos()
    if pos == goal:
        return Actions.done

    path = _shortest_path(pos, goal, env.walkable_cells())
    if len(path) < 2:
        return Actions.done

    desired_dir = _desired_direction(path[0], path[1])
    if env.agent_dir == desired_dir:
        return Actions.forward
    if (desired_dir - env.agent_dir) % 4 == 1:
        return Actions.right
    return Actions.left


def run_w1() -> None:
    actions = [
        Actions.right,
        Actions.left,
        Actions.right,
        Actions.left,
        Actions.right,
        Actions.left,
    ]

    trajectories: dict[float, list[tuple[int, int]]] = {}

    for wind_strength in (0.0, 1.0):
        env = WindyCorridorEnv(
            render_mode="rgb_array",
            confound=True,
            fixed_u=1,
            wind_strength=wind_strength,
            wind_per="step",
        )
        _, info = env.reset(seed=0)
        print(f"[EnvWind] reset hidden_u = {int(info['confounder'])}")
        print(f"[EnvWind] reset wind_strength = {float(info['wind_strength']):.1f}")

        positions: list[tuple[int, int]] = [tuple(int(v) for v in env.agent_pos)]
        for step_idx, action in enumerate(actions, start=1):
            _, _, terminated, truncated, info = env.step(action)
            pos = tuple(int(v) for v in env.agent_pos)
            print(
                f"[EnvWind] step = {step_idx}, wind_direction = {int(info['wind_direction'])}, "
                f"agent_pos = {pos}"
            )
            positions.append(pos)
            if terminated or truncated:
                break

        trajectories[wind_strength] = positions
        env.close()

    if trajectories[0.0] != trajectories[1.0]:
        raise RuntimeError("Phase W1 failed: non-forward actions should not change positions.")


def run_w2() -> None:
    def _single_step_case(
        *,
        pos: tuple[int, int],
        agent_dir: int,
        wind_dist: dict[int, tuple[float, ...]],
        expected_pos: tuple[int, int],
        expected_dir: int,
        label: str,
    ) -> tuple[int, int]:
        env = WindyCorridorEnv(
            render_mode="rgb_array",
            confound=True,
            fixed_u=1,
            wind_strength=1.0,
            wind_per="episode",
            wind_dist=wind_dist,
        )
        env.reset(seed=0)
        env.agent_pos = pos
        env.agent_dir = agent_dir
        _, _, _, _, info = env.step(Actions.forward)
        next_pos = tuple(int(v) for v in env.agent_pos)
        next_dir = int(env.agent_dir)
        print(
            f"[Dynamics] {label} wind_dir={int(info['wind_direction'])} "
            f"next_pos = {next_pos} next_dir = {next_dir}"
        )
        env.close()
        if next_pos != expected_pos or next_dir != expected_dir:
            raise RuntimeError(
                f"Phase W2 failed for {label}: expected pos={expected_pos}, dir={expected_dir}; "
                f"got pos={next_pos}, dir={next_dir}."
            )
        return next_pos

    calm_pos = _single_step_case(
        pos=(8, 13),
        agent_dir=0,
        wind_dist={
            0: (0.0, 0.0, 0.0, 0.0, 1.0),
            1: (0.0, 0.0, 0.0, 0.0, 1.0),
        },
        expected_pos=(9, 13),
        expected_dir=0,
        label="calm",
    )
    tailwind_pos = _single_step_case(
        pos=(8, 13),
        agent_dir=0,
        wind_dist={
            0: (0.0, 0.0, 0.0, 0.0, 1.0),
            1: (1.0, 0.0, 0.0, 0.0, 0.0),
        },
        expected_pos=(10, 13),
        expected_dir=0,
        label="tailwind",
    )
    headwind_pos = _single_step_case(
        pos=(8, 13),
        agent_dir=0,
        wind_dist={
            0: (0.0, 0.0, 0.0, 0.0, 1.0),
            1: (0.0, 0.0, 1.0, 0.0, 0.0),
        },
        expected_pos=(8, 13),
        expected_dir=0,
        label="headwind",
    )
    lateral_pos = _single_step_case(
        pos=(10, 9),
        agent_dir=0,
        wind_dist={
            0: (0.0, 0.0, 0.0, 0.0, 1.0),
            1: (0.0, 1.0, 0.0, 0.0, 0.0),
        },
        expected_pos=(11, 10),
        expected_dir=0,
        label="lateral_drift",
    )

    if len({calm_pos, tailwind_pos, headwind_pos, lateral_pos}) < 4:
        raise RuntimeError("Phase W2 failed: wind modes did not produce distinct transitions.")

    # Additional required check: wind_strength=0.0 recovers W1 behavior.
    no_wind_env = WindyCorridorEnv(
        render_mode="rgb_array",
        confound=True,
        fixed_u=1,
        wind_strength=0.0,
        wind_per="episode",
        wind_dist={
            0: (0.0, 0.0, 0.0, 0.0, 1.0),
            1: (1.0, 0.0, 0.0, 0.0, 0.0),
        },
    )
    no_wind_env.reset(seed=0)
    no_wind_env.agent_pos = (8, 13)
    no_wind_env.agent_dir = 0
    _, _, _, _, info = no_wind_env.step(Actions.forward)
    no_wind_pos = tuple(int(v) for v in no_wind_env.agent_pos)
    no_wind_env.close()
    if no_wind_pos != (9, 13):
        raise RuntimeError(
            "Phase W2 failed: wind_strength=0.0 did not recover normal one-step forward behavior."
        )
    print(f"[Dynamics] wind_strength=0.0 wind_dir={int(info['wind_direction'])} next_pos = {no_wind_pos}")


def run_w3() -> None:
    strengths = (0.0, 0.3, 0.6, 1.0)
    baseline_mean: float | None = None
    measured_means: list[float] = []

    for wind_strength in strengths:
        lengths: list[int] = []
        successes = 0
        for seed in range(8):
            env = WindyCorridorEnv(
                render_mode="rgb_array",
                confound=True,
                wind_strength=wind_strength,
                wind_per="step",
            )
            env.reset(seed=seed)
            terminated = False
            truncated = False
            steps = 0
            while not (terminated or truncated):
                action = _scripted_action(env)
                _, reward, terminated, truncated, _ = env.step(action)
                steps += 1
                if reward > 0:
                    successes += 1
                if steps >= env.max_steps:
                    truncated = True
            lengths.append(steps)
            env.close()

        mean_len = float(sum(lengths)) / float(len(lengths))
        success_rate = float(successes) / float(len(lengths))
        measured_means.append(mean_len)
        print(
            f"[WindSweep] wind_strength = {wind_strength:.1f}, "
            f"mean_traj_len = {mean_len:.2f}, success_rate = {success_rate:.2f}"
        )

        if wind_strength == 0.0:
            baseline_mean = mean_len

    if baseline_mean is None:
        raise RuntimeError("Phase W3 failed: baseline mean length was not computed.")
    if abs(measured_means[0] - baseline_mean) > 1e-9:
        raise RuntimeError("Phase W3 failed: baseline bookkeeping mismatch.")
    if len({round(v, 2) for v in measured_means}) == 1:
        raise RuntimeError("Phase W3 failed: trajectory statistics did not change across strengths.")


def run_wpos() -> None:
    env = WindyCorridorEnv(
        render_mode="rgb_array",
        confound=True,
        fixed_u=1,
        wind_strength=1.0,
        wind_per="step",
    )
    env.reset(seed=0)

    sheltered = (3, 13)
    exposed = (10, 13)
    sheltered_probs = env._effective_wind_probs(sheltered)
    exposed_probs = env._effective_wind_probs(exposed)

    sheltered_counts = np.zeros(5, dtype=np.int64)
    exposed_counts = np.zeros(5, dtype=np.int64)

    for _ in range(400):
        env.agent_pos = sheltered
        sheltered_counts[env._sample_wind_direction()] += 1
        env.agent_pos = exposed
        exposed_counts[env._sample_wind_direction()] += 1

    env.close()

    print(f"[WindPos] sheltered_probs = {tuple(round(v, 2) for v in sheltered_probs)}")
    print(f"[WindPos] exposed_probs = {tuple(round(v, 2) for v in exposed_probs)}")
    print(f"[WindPos] sheltered_counts = {tuple(int(v) for v in sheltered_counts)}")
    print(f"[WindPos] exposed_counts = {tuple(int(v) for v in exposed_counts)}")

    if np.allclose(sheltered_probs, exposed_probs):
        raise RuntimeError("Position-dependent wind failed: sheltered/exposed probabilities are identical.")
    if exposed_probs[0] <= sheltered_probs[0]:
        raise RuntimeError("Position-dependent wind failed: exposed east-wind probability did not increase.")
    if exposed_counts[0] <= sheltered_counts[0]:
        raise RuntimeError("Position-dependent wind failed: exposed samples did not show more east wind.")


def run_whazard() -> None:
    env = WindyCorridorEnv(
        render_mode="rgb_array",
        confound=True,
        fixed_u=1,
        wind_strength=1.0,
        wind_per="episode",
        lethal_boundaries=True,
        wind_dist={
            0: (0.0, 0.0, 0.0, 0.0, 1.0),
            1: (1.0, 0.0, 0.0, 0.0, 0.0),
        },
    )
    _, info = env.reset(seed=0)
    env.agent_pos = (11, 9)
    env.agent_dir = 1  # face down so right wind becomes lateral drift into lava
    _, reward, terminated, truncated, info = env.step(Actions.forward)
    next_pos = tuple(int(v) for v in env.agent_pos)
    env.close()

    print(
        f"[WindHazard] wind_dir={int(info['wind_direction'])} next_pos = {next_pos} "
        f"reward = {reward:.2f} terminated = {terminated} truncated = {truncated}"
    )

    if not info["lethal_boundaries"]:
        raise RuntimeError("Hazard smoke failed: lethal_boundaries flag missing from info.")
    if next_pos != (12, 10):
        raise RuntimeError(f"Hazard smoke failed: expected drift into lava at (12, 10), got {next_pos}.")
    if not terminated or truncated:
        raise RuntimeError("Hazard smoke failed: drift into lava did not terminate the episode immediately.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wind smoke script.")
    parser.add_argument("--phase", required=True, choices=["w1", "w2", "w3", "wpos", "whazard"])
    args = parser.parse_args()

    if args.phase == "w1":
        run_w1()
    elif args.phase == "w2":
        run_w2()
    elif args.phase == "w3":
        run_w3()
    elif args.phase == "wpos":
        run_wpos()
    elif args.phase == "whazard":
        run_whazard()


if __name__ == "__main__":
    main()
