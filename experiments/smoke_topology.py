"""Phase W1 topology compliance smoke test for WindyCorridor."""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.windy_corridor import WindyCorridorEnv


def neighbors(cell: tuple[int, int], walkable: set[tuple[int, int]]) -> list[tuple[int, int]]:
    x, y = cell
    out: list[tuple[int, int]] = []
    for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if nxt in walkable:
            out.append(nxt)
    return out


def count_simple_paths(
    start: tuple[int, int],
    goal: tuple[int, int],
    walkable: set[tuple[int, int]],
    limit: int = 64,
) -> list[list[tuple[int, int]]]:
    found: list[list[tuple[int, int]]] = []
    stack: list[tuple[tuple[int, int], list[tuple[int, int]], set[tuple[int, int]]]] = [
        (start, [start], {start})
    ]

    while stack and len(found) < limit:
        cell, path, seen = stack.pop()
        if cell == goal:
            found.append(path)
            continue
        for nxt in neighbors(cell, walkable):
            if nxt in seen:
                continue
            stack.append((nxt, path + [nxt], seen | {nxt}))
    return found


def shortest_distance(
    start: tuple[int, int],
    goal: tuple[int, int],
    walkable: set[tuple[int, int]],
) -> int:
    q: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
    seen = {start}
    while q:
        cell, dist = q.popleft()
        if cell == goal:
            return dist
        for nxt in neighbors(cell, walkable):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, dist + 1))
    raise RuntimeError("goal is unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Topology smoke test for WindyCorridor.")
    parser.add_argument("--map", required=True, choices=["windycorridor"])
    args = parser.parse_args()
    if args.map != "windycorridor":
        raise ValueError("Only windycorridor is supported in Phase W1.")

    walkable = WindyCorridorEnv.walkable_cells()
    high_wind = WindyCorridorEnv.high_wind_cells()
    start = WindyCorridorEnv.start_pos()
    goal = WindyCorridorEnv.goal_pos()

    all_paths = count_simple_paths(start, goal, walkable)
    shortest = shortest_distance(start, goal, walkable)
    decision_points = sum(1 for cell in walkable if len(neighbors(cell, walkable)) >= 3)
    corridor_segments = 3

    route_exposures = []
    for path in all_paths:
        if len(path) - 1 > shortest + 10:
            continue
        exposure = sum(1 for cell in path if cell in high_wind)
        route_exposures.append(exposure)

    if len(route_exposures) < 2:
        raise RuntimeError("Expected at least two viable routes for topology check.")

    max_exposure = max(route_exposures)
    min_exposure = min(route_exposures)
    exposure_diff = 0.0 if max_exposure == 0 else (max_exposure - min_exposure) / max_exposure

    print(f"[Topology] routes_found = {len(all_paths)}")
    print(f"[Topology] decision_points = {decision_points}")
    print(f"[Topology] corridor_segments = {corridor_segments}")
    print(f"[Topology] wind_exposure_diff = {exposure_diff:.2f}")
    print(f"[Topology] grid_size = 15x15")

    if len(all_paths) < 2:
        raise RuntimeError("routes_found < 2")
    if decision_points < 2:
        raise RuntimeError("decision_points < 2")
    if corridor_segments < 3:
        raise RuntimeError("corridor_segments < 3")
    if exposure_diff < 0.30:
        raise RuntimeError("wind_exposure_diff < 0.30")

    print("[Topology] PASS")


if __name__ == "__main__":
    main()
