"""Death-map visualization for ConfoundedFork (11x11).

Runs forced-U evaluation of a trained checkpoint, records where each episode
terminates, and overlays death counts (per lava cell) plus per-cell traversal
stats onto the env schematic.

The reference orange line shows the U-aware oracle's preferred path under the
chosen regime — i.e. where the agent SHOULD go but didn't.

Usage:
  python figures/draw_confounded_fork_deathmap.py \\
      --checkpoint checkpoints/causal_thm2_..._cf3.npz \\
      --fixed-u 1 \\
      --label "Thm 2" \\
      --output figures/cf_deathmap_thm2_u1.png
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gymnasium as gym
import minigrid  # noqa: F401

import envs  # noqa: F401

from envs.confounded_fork import ConfoundedForkEnv
from experiments.evaluate_actor_forced_u import (
    VALID_ACTIONS,
    _get_goal_state,
    _load_checkpoint,
)
from utils.preprocess import extract_state, extract_state_oracle


SIZE = 15
START_POS = ConfoundedForkEnv.start_pos()
GOAL_POS = ConfoundedForkEnv.goal_pos()


def _cells_to_arr(walkable, lava, size=SIZE):
    """Encode each cell as 0=wall, 1=walkable, 2=lava."""
    layer = np.zeros((size, size), dtype=np.int8)
    for y in range(size):
        for x in range(size):
            if x == 0 or y == 0 or x == size - 1 or y == size - 1:
                layer[y, x] = 0
            elif (x, y) in lava:
                layer[y, x] = 2
            elif (x, y) in walkable:
                layer[y, x] = 1
            else:
                layer[y, x] = 0
    return layer


def _bfs_path(walkable, start, goal):
    """Return one BFS shortest path start→goal (list of cells)."""
    from collections import deque

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


def _oracle_reference_path(fixed_u: int):
    """Path the U-aware oracle would take under fixed_u (used as the orange reference)."""
    walkable_restricted = ConfoundedForkEnv.walkable_for_regime(fixed_u)
    return _bfs_path(walkable_restricted, START_POS, GOAL_POS)


def _run_with_trajectories(
    *, actor, eval_env_id, fixed_u, episodes, goal_state, state_fn,
    max_steps=200, temperature=1.0,
):
    """Roll out episodes and record cells visited + death location."""
    per_cell_visits: dict[tuple[int, int], int] = defaultdict(int)
    per_cell_deaths: dict[tuple[int, int], int] = defaultdict(int)
    successes = 0
    lava_deaths = 0
    timeouts = 0
    death_examples: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []

    for ep in range(episodes):
        ep_seed = 100 + fixed_u * 1000 + ep
        rng = np.random.default_rng(ep_seed)
        env = gym.make(eval_env_id, render_mode="rgb_array", fixed_u=fixed_u)
        obs, info = env.reset(seed=ep_seed)
        if "confounder" not in info:
            info["confounder"] = int(getattr(env.unwrapped, "hidden_u", 0))

        cells: list[tuple[int, int]] = []
        success = False
        died = False
        died_at = None
        for step in range(max_steps):
            raw = env.unwrapped
            cur_cell = (int(raw.agent_pos[0]), int(raw.agent_pos[1]))
            cells.append(cur_cell)
            per_cell_visits[cur_cell] += 1

            state = state_fn(obs, info).astype(np.float64)
            action = actor.sample_action(
                state, goal_state,
                valid_actions=VALID_ACTIONS,
                rng=rng,
                temperature=temperature,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            if "confounder" not in info:
                info["confounder"] = int(getattr(env.unwrapped, "hidden_u", 0))

            if reward > 0:
                success = True
                break
            if terminated:
                died = True
                next_cell = (int(env.unwrapped.agent_pos[0]), int(env.unwrapped.agent_pos[1]))
                died_at = next_cell
                break
            if truncated:
                break

        if success:
            successes += 1
        elif died:
            lava_deaths += 1
            if died_at is not None:
                per_cell_deaths[died_at] += 1
                death_examples.append((died_at, cells))
        else:
            timeouts += 1

        env.close()

    return {
        "successes": successes,
        "lava_deaths": lava_deaths,
        "timeouts": timeouts,
        "per_cell_visits": dict(per_cell_visits),
        "per_cell_deaths": dict(per_cell_deaths),
    }


def draw(
    checkpoint: Path,
    *,
    fixed_u: int,
    episodes: int,
    label: str,
    output: Path,
    eval_env_id: str = "CausalContrastive-ConfoundedFork-11x11-Lethal-v0",
    temperature: float = 1.0,
    dpi: int = 150,
) -> None:
    actor, cfg = _load_checkpoint(checkpoint)
    state_fn = extract_state_oracle if cfg.get("oracle_state") else extract_state

    goal_state = _get_goal_state(
        env_id=eval_env_id, fixed_u=fixed_u, seed=0, state_fn=state_fn,
    )

    rollout = _run_with_trajectories(
        actor=actor, eval_env_id=eval_env_id, fixed_u=fixed_u,
        episodes=episodes, goal_state=goal_state, state_fn=state_fn,
        max_steps=200, temperature=temperature,
    )

    walkable = ConfoundedForkEnv.walkable_cells()
    lava = ConfoundedForkEnv.hazard_cells()
    layer = _cells_to_arr(walkable, lava)

    colors = np.array(
        [
            [0.16, 0.16, 0.18],
            [0.92, 0.96, 0.94],
            [0.90, 0.22, 0.14],
        ],
        dtype=np.float32,
    )
    rgb = colors[layer]

    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(rgb, origin="upper", extent=(-0.5, SIZE - 0.5, SIZE - 0.5, -0.5),
              interpolation="nearest", zorder=1)

    for g in range(SIZE + 1):
        ax.axhline(g - 0.5, color="#bdbdbd", lw=0.5, zorder=2)
        ax.axvline(g - 0.5, color="#bdbdbd", lw=0.5, zorder=2)

    # Highlight the oracle reference path (semi-transparent yellow)
    ref_path = _oracle_reference_path(fixed_u)
    ref_label = f"U={fixed_u} oracle path (reference)"
    if ref_path:
        for cx, cy in ref_path:
            ax.add_patch(mpatches.Rectangle(
                (cx - 0.5, cy - 0.5), 1, 1,
                facecolor="#ffd54f", alpha=0.55, edgecolor="none", zorder=2.5,
            ))
        rx = [c[0] for c in ref_path]
        ry = [c[1] for c in ref_path]
        ax.plot(rx, ry, color="#ef6c00", lw=1.8, alpha=0.9, zorder=4, label=ref_label)

    # Per-cell visit annotations: only on a small sample of representative cells
    # on the reference path (corners + every Nth cell), placed in margins to
    # avoid overlap.
    deaths = rollout["per_cell_deaths"]
    visits = rollout["per_cell_visits"]
    n_eps = episodes

    def _annotate_cell(cx, cy, side="right", color="#1a237e"):
        v = visits.get((cx, cy), 0)
        if v == 0:
            return
        if side == "right":
            tx = SIZE - 0.2
            ha = "left"
        elif side == "left":
            tx = -0.2
            ha = "right"
        else:
            tx = cx
            ha = "center"
        t = ax.annotate(
            f"({cx},{cy}): {v} visits",
            xy=(cx, cy), xytext=(tx, cy),
            fontsize=7.5, color=color, ha=ha, va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.4, alpha=0.6),
            zorder=5,
        )

    # Annotate ONLY a few key landmarks on the reference path so labels stay
    # legible: start, fork (next cell after start), and 3-4 evenly spaced
    # waypoints. Stagger left vs right based on cell x-position.
    if ref_path and len(ref_path) >= 2:
        n = len(ref_path)
        sample_idx = [0, 1]
        for frac in (0.33, 0.66, 0.9):
            sample_idx.append(int((n - 1) * frac))
        seen: set[tuple[int, int]] = set()
        for i in sample_idx:
            cell = ref_path[i]
            if cell in seen or cell in lava:
                continue
            seen.add(cell)
            side = "right" if cell[0] <= SIZE // 2 else "left"
            _annotate_cell(*cell, side=side)

    # Death circles on lava cells
    if deaths:
        max_deaths = max(deaths.values())
        for (cx, cy), n_died in deaths.items():
            radius = 0.18 + 0.32 * (n_died / max_deaths)
            ax.add_patch(plt.Circle(
                (cx, cy), radius=radius, color="#b71c1c", alpha=0.85, zorder=6,
            ))
            t = ax.text(
                cx + 0.65, cy - 0.25,
                f"{n_died} deaths\n({cx},{cy})",
                fontsize=8.5, fontweight="bold", color="#b71c1c", zorder=7,
            )
            t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    # Start and goal markers
    sx, sy = START_POS
    gx, gy = GOAL_POS
    ax.add_patch(mpatches.Rectangle(
        (sx - 0.5, sy - 0.5), 1, 1,
        facecolor="#90caf9", edgecolor="#0d47a1", lw=1.0, zorder=5,
    ))
    ax.add_patch(mpatches.Rectangle(
        (gx - 0.5, gy - 0.5), 1, 1,
        facecolor="#a5d6a7", edgecolor="#1b5e20", lw=1.0, zorder=5,
    ))
    for xx, yy, s in [(sx, sy, "S"), (gx, gy, "G")]:
        t = ax.text(xx, yy, s, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="#0d0d0d", zorder=7)
        t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

    # Stats header — placed in an unused part of the wall area top-right
    head = (f"success: {rollout['successes']}/{n_eps}\n"
            f"lava: {rollout['lava_deaths']}/{n_eps}\n"
            f"timeout: {rollout['timeouts']}")
    ht = ax.text(SIZE - 1.5, 0.0, head, fontsize=10, fontweight="bold",
                 color="#b71c1c", ha="right", va="top", zorder=7)
    ht.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(SIZE))
    ax.set_yticks(range(SIZE))
    for sp in ax.spines.values():
        sp.set_edgecolor("#888888")
        sp.set_linewidth(0.6)

    ax.set_title(
        f"Where U={fixed_u} episodes die ({n_eps} eval eps, {label})",
        fontsize=12, pad=10,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output.resolve()}")
    print(f"  success_rate={rollout['successes']}/{n_eps}={rollout['successes']/n_eps:.2f}")
    print(f"  lava_deaths={rollout['lava_deaths']}  timeouts={rollout['timeouts']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--fixed-u", type=int, choices=[0, 1], required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--label", type=str, default="actor")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eval-env-id", type=str,
                   default="CausalContrastive-ConfoundedFork-11x11-Lethal-v0")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    draw(
        args.checkpoint,
        fixed_u=args.fixed_u,
        episodes=args.episodes,
        label=args.label,
        output=args.output,
        eval_env_id=args.eval_env_id,
        temperature=args.temperature,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
