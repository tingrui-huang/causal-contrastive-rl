"""Phase W1/W2 wind smoke tests."""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    test_wind_dist = {
        0: (0.0, 0.0, 0.0, 0.0, 1.0),  # U=0: no wind
        1: (1.0, 0.0, 0.0, 0.0, 0.0),  # U=1: always wind to the right
    }
    next_positions: dict[int, tuple[int, int]] = {}

    for u in (0, 1):
        env = WindyCorridorEnv(
            render_mode="rgb_array",
            confound=True,
            fixed_u=u,
            wind_strength=1.0,
            wind_per="step",
            wind_dist=test_wind_dist,
        )
        env.reset(seed=0)
        env.agent_pos = (8, 13)
        env.agent_dir = 0  # face right
        _, _, _, _, info = env.step(Actions.forward)
        next_pos = tuple(int(v) for v in env.agent_pos)
        next_positions[u] = next_pos
        print(f"[Dynamics] U={u} wind_dir={int(info['wind_direction'])} next_pos = {next_pos}")
        env.close()

    if next_positions[0] == next_positions[1]:
        raise RuntimeError("Phase W2 failed: expected different next_pos across U.")

    # Additional required check: wind_strength=0.0 recovers W1 behavior.
    no_wind_env = WindyCorridorEnv(
        render_mode="rgb_array",
        confound=True,
        fixed_u=1,
        wind_strength=0.0,
        wind_per="step",
        wind_dist=test_wind_dist,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Wind smoke script.")
    parser.add_argument("--phase", required=True, choices=["w1", "w2", "w3"])
    args = parser.parse_args()

    if args.phase == "w1":
        run_w1()
    elif args.phase == "w2":
        run_w2()
    elif args.phase == "w3":
        run_w3()


if __name__ == "__main__":
    main()
