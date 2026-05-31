"""
Pure-function analytical replication of causal_gym.WindyMiniGridSCM dynamics.

Why we re-implement instead of calling env.step:
* Worst-case neighbor enumeration (Theorem 2's (iii) branch) needs to evaluate
  the result of one (state, action, wind) tuple, in isolation, without
  mutating any env's internal state or sampling the next wind. The SCM's step
  also advances steps_cnt and resamples wind — both are inappropriate here.

Coverage:
* MiniGrid actions: left (0), right (1), forward (2), done (6). pickup/drop/
  toggle are no-ops in this env and we don't enumerate them.
* Wind values: 0=east, 1=south, 2=west, 3=north, 4=still — same encoding as
  causal_gym.envs.windy_minigrid.
* Agent dir: same convention. dir_vec(0)=(+1,0), dir_vec(1)=(0,+1), etc.

The forward+wind action-sequence logic mirrors WindyMiniGridSCM._wind_to_actions
verbatim, including the early termination on lava during the sequence (which is
what makes lethal_x at y=1 deadly under south wind).
"""
from __future__ import annotations

from minigrid.core.actions import Actions

from envs.windy_corridor import GOAL_POS, LETHAL_X, WindyCorridorEnv

# Derive lava positions from the env's actual y=2 pattern (single source of truth).
# Previously this was wrongly defined as `(x, 2) for x in LETHAL_X` — but LETHAL_X is
# the set of y=1 cells from which forward+south-wind lands in lava ONE COLUMN OVER
# (the drift sequence ends at (x+1, 2)). Using LETHAL_X directly off-by-one'd the
# lava set to {(2,2),(3,2),(4,2),(7,2),(8,2),(9,2)} (which includes walls (2,2),(7,2)
# and misses real lava (5,2),(10,2)). Reading from the pattern keeps this in sync
# with what `_gen_grid` actually places via `Lava()` objects.
LAVA_CELLS: frozenset[tuple[int, int]] = frozenset(
    (x, 2) for x, cell in WindyCorridorEnv._y2_pattern().items() if cell == "L"
)
WALKABLE_CELLS: frozenset[tuple[int, int]] = frozenset(WindyCorridorEnv.walkable_cells())
_GOAL_CELL: tuple[int, int] = GOAL_POS

WIND_STILL = 4

_DIR_VEC = {
    0: (1, 0),    # east
    1: (0, 1),    # south
    2: (-1, 0),   # west
    3: (0, -1),   # north
    WIND_STILL: (0, 0),
}


def _primitive_step(
    pos: tuple[int, int], dir_: int, action: int
) -> tuple[tuple[int, int], int, bool]:
    """One MiniGrid primitive (no wind). Returns (new_pos, new_dir, terminated)."""
    if action == int(Actions.left):
        return pos, (dir_ - 1) % 4, False
    if action == int(Actions.right):
        return pos, (dir_ + 1) % 4, False
    if action == int(Actions.forward):
        dx, dy = _DIR_VEC[dir_]
        target = (pos[0] + dx, pos[1] + dy)
        if target in LAVA_CELLS:
            return target, dir_, True
        if target == _GOAL_CELL:
            return target, dir_, True  # MiniGrid terminates on goal arrival
        if target in WALKABLE_CELLS:
            return target, dir_, False
        return pos, dir_, False  # wall / out of bounds → no move
    # done / pickup / drop / toggle → no movement, no termination
    return pos, dir_, False


def _wind_action_sequence(dir_: int, wind: int) -> list[int]:
    """Mirrors WindyMiniGridSCM._wind_to_actions exactly."""
    if wind == WIND_STILL:
        return [int(Actions.forward)]
    if dir_ == wind:
        return [int(Actions.forward), int(Actions.forward)]
    if (dir_ - 2) % 4 == wind:
        return [int(Actions.done)]
    # sideways wind: forward, turn-into-wind, forward (drift), turn-back
    diff = dir_ - wind
    if diff == 1 or diff == -3:
        first_turn = int(Actions.left)
        second_turn = int(Actions.right)
    else:
        first_turn = int(Actions.right)
        second_turn = int(Actions.left)
    return [int(Actions.forward), first_turn, int(Actions.forward), second_turn]


def analytical_step(
    pos: tuple[int, int], dir_: int, action: int, wind: int
) -> tuple[tuple[int, int], int, bool]:
    """Apply (action, wind) at (pos, dir). Returns (new_pos, new_dir, terminated).

    Replicates WindyMiniGridSCM.step (without reward, step counting, or next-wind
    resampling). Wind only affects ``forward``; turns/done ignore wind.
    """
    if action != int(Actions.forward):
        return _primitive_step(pos, dir_, action)

    seq = _wind_action_sequence(dir_, wind)
    cur_pos, cur_dir = pos, dir_
    for act in seq:
        cur_pos, cur_dir, terminated = _primitive_step(cur_pos, cur_dir, act)
        if terminated:
            return cur_pos, cur_dir, True
    return cur_pos, cur_dir, False


def enumerate_neighbors(
    pos: tuple[int, int], dir_: int, action: int
) -> list[tuple[tuple[int, int], int, bool]]:
    """N(s, x): list of (new_pos, new_dir, terminated) over all 5 wind values.

    For non-forward actions all 5 winds collapse to the same neighbor — caller
    can still safely take argmin over the duplicated list.
    """
    return [analytical_step(pos, dir_, action, w) for w in range(5)]


def is_goal(pos: tuple[int, int]) -> bool:
    return pos == GOAL_POS


def is_lava(pos: tuple[int, int]) -> bool:
    return pos in LAVA_CELLS
