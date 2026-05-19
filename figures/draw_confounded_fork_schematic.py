"""Pure schematic of ConfoundedFork (15×15): walkable / lava / S / G / two routes.

Run from repo root:
  python figures/draw_confounded_fork_schematic.py
  python figures/draw_confounded_fork_schematic.py -o figures/cf_schematic.png
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from envs.confounded_fork import ConfoundedForkEnv

SIZE = 15
START = ConfoundedForkEnv.start_pos()
GOAL = ConfoundedForkEnv.goal_pos()


def _layer(walk, lava):
    arr = np.zeros((SIZE, SIZE), dtype=np.int8)
    for y in range(SIZE):
        for x in range(SIZE):
            if x == 0 or y == 0 or x == SIZE - 1 or y == SIZE - 1:
                arr[y, x] = 0
            elif (x, y) in lava:
                arr[y, x] = 2
            elif (x, y) in walk:
                arr[y, x] = 1
            else:
                arr[y, x] = 0
    return arr


def _bfs(walkable, start, goal):
    q = deque([start])
    parent = {start: None}
    while q:
        c = q.popleft()
        if c == goal:
            break
        x, y = c
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in walkable and nxt not in parent:
                parent[nxt] = c
                q.append(nxt)
    if goal not in parent:
        return []
    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    return list(reversed(path))


def draw(out_path: Path, dpi: int = 150) -> None:
    walk = ConfoundedForkEnv.walkable_cells()
    lava = ConfoundedForkEnv.hazard_cells()
    left_path = _bfs(ConfoundedForkEnv.walkable_for_regime(0), START, GOAL)
    right_path = _bfs(ConfoundedForkEnv.walkable_for_regime(1), START, GOAL)

    arr = _layer(walk, lava)
    colors = np.array(
        [
            [0.16, 0.16, 0.18],
            [0.92, 0.96, 0.94],
            [0.90, 0.22, 0.14],
        ],
        dtype=np.float32,
    )
    rgb = colors[arr]

    fig, ax = plt.subplots(figsize=(8.0, 8.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(
        rgb,
        origin="upper",
        extent=(-0.5, SIZE - 0.5, SIZE - 0.5, -0.5),
        interpolation="nearest",
        zorder=1,
    )
    for g in range(SIZE + 1):
        ax.axhline(g - 0.5, color="#c0c0c0", lw=0.45, zorder=2)
        ax.axvline(g - 0.5, color="#c0c0c0", lw=0.45, zorder=2)

    # Left (short risky) path
    if left_path:
        xs = [c[0] for c in left_path]
        ys = [c[1] for c in left_path]
        ax.plot(xs, ys, color="#b71c1c", lw=2.6, solid_capstyle="round",
                zorder=4, label=f"left short ({len(left_path) - 1} steps, double-sided lava)")

    # Right (long safe) path
    if right_path:
        xs = [c[0] for c in right_path]
        ys = [c[1] for c in right_path]
        ax.plot(xs, ys, color="#0d47a1", lw=2.6, solid_capstyle="round",
                zorder=4, label=f"right long ({len(right_path) - 1} steps, no lava)")

    sx, sy = START
    gx, gy = GOAL
    ax.add_patch(mpatches.Rectangle((sx - 0.5, sy - 0.5), 1, 1,
                                    facecolor="#90caf9", edgecolor="#0d47a1",
                                    lw=1.2, zorder=5))
    ax.add_patch(mpatches.Rectangle((gx - 0.5, gy - 0.5), 1, 1,
                                    facecolor="#a5d6a7", edgecolor="#1b5e20",
                                    lw=1.2, zorder=5))
    for xx, yy, s in [(sx, sy, "S"), (gx, gy, "G")]:
        t = ax.text(xx, yy, s, ha="center", va="center", fontsize=14,
                    fontweight="bold", color="#0d0d0d", zorder=7)
        t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    # Fork marker
    fx, fy = (2, 12)
    ax.add_patch(plt.Circle((fx, fy), radius=0.35, fill=False,
                            edgecolor="#ffb300", lw=2.5, zorder=6))
    t = ax.text(fx + 0.6, fy + 0.0, "fork (1 step up from S)",
                fontsize=9, fontweight="bold", color="#bf360c", zorder=7)
    t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(SIZE))
    ax.set_yticks(range(SIZE))
    for sp in ax.spines.values():
        sp.set_edgecolor("#888")
        sp.set_linewidth(0.6)

    ax.set_title(
        "ConfoundedFork (15×15): U=0 picks left, U=1 picks right\n"
        "wind dist — U=0: none | U=1: each direction 20% (80% wind total)",
        fontsize=11, pad=10,
    )

    terrain_handles = [
        mpatches.Patch(facecolor=tuple(colors[0]), edgecolor="#000", label="wall / blocked"),
        mpatches.Patch(facecolor=tuple(colors[1]), edgecolor="#888", label="walkable corridor"),
        mpatches.Patch(facecolor=tuple(colors[2]), edgecolor="#000", label="lava (lethal)"),
    ]
    leg1 = ax.legend(handles=terrain_handles, loc="upper left", fontsize=9,
                     framealpha=0.95, title="Terrain", title_fontsize=10)
    ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
              title="Oracle U-aware routes", title_fontsize=10)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path.resolve()}")
    print(f"  left path length: {len(left_path) - 1} steps")
    print(f"  right path length: {len(right_path) - 1} steps")
    print(f"  walkable cells: {len(walk)}")
    print(f"  lava cells: {len(lava)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path,
                   default=_REPO / "figures" / "confounded_fork_schematic.png")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    draw(args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
