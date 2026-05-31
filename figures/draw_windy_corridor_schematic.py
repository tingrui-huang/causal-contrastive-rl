"""
Single-frame schematic of WindyCorridor (15x15) for slides: topology, lava, S/G, NEAR vs FAR.

Topology mirrors ``envs/windy_corridor.WindyCorridorEnv`` (walkable / hazards / start / goal).

Run from repo root:
  python figures/draw_windy_corridor_schematic.py
  python figures/draw_windy_corridor_schematic.py -o figures/windy_corridor_schematic.png
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]


def _load_env_module():
    """Load envs/windy_corridor.py without triggering envs/__init__.py (which needs causal_gym)."""
    spec = importlib.util.spec_from_file_location(
        "_windy_corridor_static", _REPO / "envs" / "windy_corridor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_env = _load_env_module()
GOAL_POS = _env.GOAL_POS
LETHAL_X = _env.LETHAL_X
SAFE_WAIT_X = _env.SAFE_WAIT_X
SIZE = _env.SIZE
START_POS = _env.START_POS
WindyCorridorEnv = _env.WindyCorridorEnv


def _y2_lava_cells() -> set[tuple[int, int]]:
    pattern = WindyCorridorEnv._y2_pattern()
    return {(x, 2) for x, cell in pattern.items() if cell == "L"}


def _build_layer() -> np.ndarray:
    """layer[y, x]: 0 wall, 1 corridor, 2 lava. origin=lower for imshow."""
    walk = WindyCorridorEnv.walkable_cells()
    lava = _y2_lava_cells()

    layer = np.zeros((SIZE, SIZE), dtype=np.int8)
    for y in range(SIZE):
        for x in range(SIZE):
            if (x, y) in lava:
                layer[y, x] = 2
            elif (x, y) in walk:
                layer[y, x] = 1
            else:
                layer[y, x] = 0
    gx, gy = GOAL_POS
    layer[gy, gx] = 1
    return layer


# NEAR (shortcut, risky): straight along y=1 from start to goal.
NEAR_ROUTE: list[tuple[int, int]] = [(x, 1) for x in range(1, 14)]

# FAR (detour, safe): down the left connector, east along y=13, up the right connector.
FAR_ROUTE: list[tuple[int, int]] = (
    [(1, y) for y in range(1, 14)]
    + [(x, 13) for x in range(2, 14)]
    + [(13, y) for y in range(12, 0, -1)]
)


def _verify_paths() -> None:
    walk = WindyCorridorEnv.walkable_cells()
    for name, path in ("NEAR", NEAR_ROUTE), ("FAR", FAR_ROUTE):
        for cell in path:
            if cell not in walk:
                raise ValueError(f"{name} uses non-walkable cell {cell}")
        for a, b in zip(path, path[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
                raise ValueError(f"{name} non-adjacent step {a}->{b}")
        if path[0] != START_POS or path[-1] != GOAL_POS:
            raise ValueError(f"{name} endpoints {path[0]}->{path[-1]} mismatch S/G")


def draw_schematic(out_path: Path, dpi: int = 150) -> None:
    _verify_paths()

    layer = _build_layer()
    sx, sy = START_POS
    gx, gy = GOAL_POS

    colors = np.array(
        [
            [0.12, 0.12, 0.16],  # wall / blocked
            [0.98, 0.98, 0.99],  # corridor floor
            [0.90, 0.22, 0.14],  # lava
        ],
        dtype=np.float32,
    )
    rgb = colors[layer]

    fig, ax = plt.subplots(figsize=(6.4, 6.6), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(
        rgb,
        origin="lower",
        extent=(-0.5, SIZE - 0.5, -0.5, SIZE - 0.5),
        interpolation="nearest",
        zorder=1,
    )

    for g in range(SIZE + 1):
        ax.axhline(g - 0.5, color="#c5c5c5", lw=0.45, zorder=2)
        ax.axvline(g - 0.5, color="#c5c5c5", lw=0.45, zorder=2)

    # Highlight lethal-under-south-wind cells on the NEAR row (faint red ring)
    for x in sorted(LETHAL_X):
        ax.add_patch(
            mpatches.Rectangle(
                (x - 0.5, 1 - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#b71c1c",
                linewidth=1.2,
                linestyle=(0, (2, 1.5)),
                zorder=3,
            )
        )
    # Faint green ring on safe-wait cells
    for x in sorted(SAFE_WAIT_X):
        ax.add_patch(
            mpatches.Rectangle(
                (x - 0.5, 1 - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#1b5e20",
                linewidth=1.2,
                linestyle=(0, (2, 1.5)),
                zorder=3,
            )
        )

    nx_pts = [p[0] for p in NEAR_ROUTE]
    ny_pts = [p[1] for p in NEAR_ROUTE]
    fx_pts = [p[0] for p in FAR_ROUTE]
    fy_pts = [p[1] for p in FAR_ROUTE]

    ax.plot(
        nx_pts,
        ny_pts,
        color="#b71c1c",
        lw=2.8,
        solid_capstyle="round",
        zorder=5,
        label=f"NEAR (~{len(NEAR_ROUTE) - 1} steps, risky)",
    )
    ax.plot(
        fx_pts,
        fy_pts,
        color="#0d47a1",
        lw=2.8,
        solid_capstyle="round",
        zorder=4,
        label=f"FAR (~{len(FAR_ROUTE) - 1} steps, safe)",
    )

    ax.scatter([sx], [sy], s=140, c="#ffc107", edgecolors="#1a1a1a", linewidths=1.4, zorder=7)
    ax.scatter([gx], [gy], s=140, c="#2e7d32", edgecolors="#1a1a1a", linewidths=1.4, zorder=7)

    def _lbl(xx: float, yy: float, s: str, dx: float, dy: float) -> None:
        t = ax.text(
            xx + dx,
            yy + dy,
            s,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#0d0d0d",
        )
        t.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    _lbl(sx, sy, "S", dx=0.0, dy=0.85)
    _lbl(gx, gy, "G", dx=0.0, dy=0.85)

    # South-wind hint: arrow points down from the wall block onto a lava cell
    ax.annotate(
        "",
        xy=(4.0, 1.6),
        xytext=(4.0, 6.0),
        arrowprops=dict(arrowstyle="->", color="#fafafa", lw=2.2),
        zorder=6,
    )
    wt = ax.text(
        4.0,
        6.6,
        "south wind U\n(pushes NEAR onto lava)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#fafafa",
    )
    wt.set_path_effects([pe.withStroke(linewidth=2.5, foreground="#1a1a1a")])

    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(-0.5, SIZE - 0.5)
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
        mpatches.Patch(facecolor="none", edgecolor="#b71c1c", linewidth=1.2, linestyle="--", label="NEAR lethal under south wind"),
        mpatches.Patch(facecolor="none", edgecolor="#1b5e20", linewidth=1.2, linestyle="--", label="NEAR safe-wait cell"),
    ]
    leg1 = ax.legend(
        handles=terrain_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.04),
        fontsize=7.5,
        framealpha=0.98,
        title="Terrain",
        title_fontsize=8.5,
        ncol=1,
    )
    ax.add_artist(leg1)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, -0.04),
        fontsize=8,
        framealpha=0.98,
        title="Routes",
        title_fontsize=9,
    )

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
