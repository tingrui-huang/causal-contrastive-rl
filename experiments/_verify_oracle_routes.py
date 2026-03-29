"""Quick verification that the wind-aware oracle produces two distinct routes."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.offline_data import _safe_walkable_for_regime, _shortest_path_next_cell
from envs.windy_corridor import WindyCorridorEnv

walkable = WindyCorridorEnv.walkable_cells()
start = (1, 13)
goal = (13, 5)


def trace_route(walkable_set):
    path = [start]
    cur = start
    for _ in range(50):
        nxt = _shortest_path_next_cell(cur, goal, walkable_set)
        if nxt == cur:
            break
        path.append(nxt)
        cur = nxt
    return path


w0 = _safe_walkable_for_regime(walkable, 0, None)
path0 = trace_route(w0)
print(f"u=0 route length: {len(path0)}")
print(f"u=0 route: {path0}")
print()

w1 = _safe_walkable_for_regime(walkable, 1, None)
path1 = trace_route(w1)
print(f"u=1 route length: {len(path1)}")
print(f"u=1 route: {path1}")
print()

shared = 0
for i in range(min(len(path0), len(path1))):
    if path0[i] == path1[i]:
        shared += 1
    else:
        break
print(f"Routes share first {shared} cells, then diverge")
print(f"u=0 goes through x=11 passage: {any(c[0]==11 and c[1]>5 for c in path0)}")
print(f"u=1 avoids x=11 passage: {not any(c[0]==11 and c[1]>5 for c in path1)}")
print(f"u=0 reaches goal: {path0[-1] == goal}")
print(f"u=1 reaches goal: {path1[-1] == goal}")
