"""
Single-frame schematic of WindyCorridor (15x15) for slides: topology, lava, S/G, shortcut vs detour.

Topology must match ``envs/windy_corridor.WindyCorridorEnv`` (walkable / hazards / start / goal).

Run from repo root:
  python figures/draw_windy_corridor_schematic.py
  python figures/draw_windy_corridor_schematic.py -o results/windy_corridor_schematic.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _walkable_cells() -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    cells.update((x, 13) for x in range(1, 14))
    cells.update((x, 9) for x in range(3, 12))
    cells.update((x, 5) for x in range(7, 14))
    cells.update((3, y) for y in range(9, 14))
    cells.update((7, y) for y in range(5, 10))
    cells.update((11, y) for y in range(5, 14))
    return cells


def _hazard_cells() -> set[tuple[int, int]]:
    return {(10, 10), (10, 11), (12, 10), (12, 11), (12, 12)}


START_POS = (1, 13)
GOAL_POS = (13, 5)


def _build_layer() -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """layer[y, x] in image coords (origin bottom-left for imshow with origin=lower)."""
    size = 15
    walk = _walkable_cells()
    lava = _hazard_cells()
    sx, sy = START_POS
    gx, gy = GOAL_POS

    # 0 wall, 1 corridor, 2 lava, 3 goal (drawn separately)
    layer = np.zeros((size, size), dtype=np.int8)
    for y in range(size):
        for x in range(size):
            if x == 0 or y == 0 or x == size - 1 or y == size - 1:
                layer[y, x] = 0
            elif (x, y) in lava:
                layer[y, x] = 2
            elif (x, y) in walk:
                layer[y, x] = 1
            else:
                layer[y, x] = 0
    layer[gy, gx] = 1  # goal on walkable
    meta = {"start": (sx, sy), "goal": (gx, gy)}
    return layer, meta


# Hand-validated routes on walkable_cells (shortcut: bottom band then right spine; detour: early turn-up then middle band)
SHORTCUT: list[tuple[int, int]] = [
    *[(x, 13) for x in range(1, 12)],  # (1,13)..(11,13)
    *[(11, y) for y in range(12, 4, -1)],  # (11,12)..(11,5)
    (12, 5),
    (13, 5),
]

DETOUR: list[tuple[int, int]] = [
    (1, 13),
    (2, 13),
    (3, 13),
    (3, 12),
    (3, 11),
    (3, 10),
    (3, 9),
    (4, 9),
    (5, 9),
    (6, 9),
    (7, 9),
    (7, 8),
    (7, 7),
    (7, 6),
    (7, 5),
    (8, 5),
    (9, 5),
    (10, 5),
    (11, 5),
    (12, 5),
    (13, 5),
]

# Another valid route (hybrid): right spine to mid band, then cut across y=9 to x=7, then up-band to goal.
# The env does not single out only two paths — this illustrates that many walks exist; we still contrast shortcut vs detour for the story.
HYBRID_EXAMPLE: list[tuple[int, int]] = [
    *[(x, 13) for x in range(1, 12)],
    *[(11, y) for y in range(12, 8, -1)],  # (11,12)..(11,9)
    (10, 9),
    (9, 9),
    (8, 9),
    (7, 9),
    (7, 8),
    (7, 7),
    (7, 6),
    (7, 5),
    (8, 5),
    (9, 5),
    (10, 5),
    (11, 5),
    (12, 5),
    (13, 5),
]


def _verify_paths() -> None:
    w = _walkable_cells()
    goal = GOAL_POS
    for name, path in ("shortcut", SHORTCUT), ("detour", DETOUR), ("hybrid", HYBRID_EXAMPLE):
        for x, y in path:
            if (x, y) not in w and (x, y) != goal:
                raise ValueError(f"{name} uses non-walkable cell {(x, y)}")
        for a, b in zip(path, path[1:]):
            dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
            if dx + dy != 1:
                raise ValueError(f"{name} non-adjacent step {a}->{b}")


def draw_schematic(out_path: Path, dpi: int = 150) -> None:
    _verify_paths()

    layer, meta = _build_layer()
    sx, sy = meta["start"]
    gx, gy = meta["goal"]

    # High-contrast: dark = blocked wall/rock, off-white = walkable corridor, bright = lava
    colors = np.array(
        [
            [0.12, 0.12, 0.16],  # wall / blocked (non-walkable interior + border)
            [0.98, 0.98, 0.99],  # corridor floor
            [0.90, 0.22, 0.14],  # lava (lethal when lethal_boundaries=True)
        ],
        dtype=np.float32,
    )
    rgb = colors[layer]

    fig, ax = plt.subplots(figsize=(6.2, 6.4), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(rgb, origin="lower", extent=(-0.5, 14.5, -0.5, 14.5), interpolation="nearest", zorder=1)

    # Light grid so cells are readable on projector
    for g in range(16):
        ax.axhline(g - 0.5, color="#c5c5c5", lw=0.45, zorder=2)
        ax.axvline(g - 0.5, color="#c5c5c5", lw=0.45, zorder=2)

    # Routes (x, y)
    sx_pts = [p[0] for p in SHORTCUT]
    sy_pts = [p[1] for p in SHORTCUT]
    dx_pts = [p[0] for p in DETOUR]
    dy_pts = [p[1] for p in DETOUR]
    hx_pts = [p[0] for p in HYBRID_EXAMPLE]
    hy_pts = [p[1] for p in HYBRID_EXAMPLE]

    ax.plot(sx_pts, sy_pts, color="#b71c1c", lw=2.8, solid_capstyle="round", zorder=4, label="shortcut (risky)")
    ax.plot(dx_pts, dy_pts, color="#0d47a1", lw=2.8, solid_capstyle="round", zorder=4, label="detour (safe)")
    ax.plot(
        hx_pts,
        hy_pts,
        color="#5d4037",
        lw=1.8,
        ls=(0, (4, 3)),
        solid_capstyle="round",
        zorder=3,
        label="other feasible route (example)",
    )

    ax.scatter([sx], [sy], s=130, c="#ffc107", edgecolors="#1a1a1a", linewidths=1.4, zorder=6)
    ax.scatter([gx], [gy], s=130, c="#2e7d32", edgecolors="#1a1a1a", linewidths=1.4, zorder=6)

    def _lbl(xx: float, yy: float, s: str) -> None:
        t = ax.text(xx, yy + 0.55, s, ha="center", va="bottom", fontsize=12, fontweight="bold", color="#0d0d0d")
        t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    _lbl(sx, sy, "S")
    _lbl(gx, gy, "G")

    # Wind hint (lateral push — illustrative)
    ax.annotate(
        "",
        xy=(11.35, 10.5),
        xytext=(9.0, 10.5),
        arrowprops=dict(arrowstyle="->", color="#212121", lw=2.0),
        zorder=5,
    )
    wt = ax.text(
        10.0,
        11.12,
        "strong wind\n(lateral drift)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#000000",
    )
    wt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 14.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Windy Corridor (15×15)", fontsize=13, pad=10, color="#111111")

    terrain_handles = [
        mpatches.Patch(facecolor=tuple(colors[0]), edgecolor="#000000", linewidth=0.6, label="blocked / wall"),
        mpatches.Patch(facecolor=tuple(colors[1]), edgecolor="#888888", linewidth=0.5, label="corridor (walkable)"),
        mpatches.Patch(facecolor=tuple(colors[2]), edgecolor="#000000", linewidth=0.6, label="lava (failure)"),
    ]
    leg1 = ax.legend(handles=terrain_handles, loc="upper left", fontsize=8, framealpha=0.98, title="Terrain", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.98, title="Example routes", title_fontsize=9)

    fig.text(
        0.5,
        0.02,
        "Same topology as envs/windy_corridor.py: non-walkable cells render as walls/lava. Many paths exist; "
        "experiments contrast shortcut vs detour under hidden wind U — not “only two routes”.",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#222222",
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_REPO / "figures" / "windy_corridor_schematic.png",
        help="Output PNG path",
    )
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()
    draw_schematic(args.output, dpi=args.dpi)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
