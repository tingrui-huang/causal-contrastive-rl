r"""
Empirical observational transition $P_{obs}(s' \mid s, x)$ from offline data.

Theorem 2's (i) branch should replay the *observational* transition conditioned
on the action actually taken -- i.e. resample a next state from the empirical
distribution of (s, a) -> s' seen in the data. This is NOT the same as
re-rolling a fresh wind from WIND_DIST and stepping analytically: under
confounding the behavior policy's action is correlated with the latent wind, so
conditioning on the action implicitly conditions on the winds that co-occurred
with it. (E.g. the corridor expert only goes `forward` on a lethal cell when the
wind is NOT south -> P_obs(lava | lethal, forward) = 0 in the data, whereas a
fresh WIND_DIST roll would inject ~10% lava.)

State is canonicalised to integer (x, y, dir), matching propensity.py.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from utils.wind_dynamics import is_goal, is_lava

State = tuple[int, int, int]


def estimate_obs_transition(
    episodes_states: list[np.ndarray],
    episodes_actions: list[np.ndarray],
) -> dict[tuple[State, int], tuple[list[State], np.ndarray]]:
    """Return mapping (s, a) -> (next_states, probs) over observed transitions.

    For each transition t in each episode, count s=states[t], a=actions[t],
    s'=states[t+1]. ``probs`` is the empirical P_obs(s' | s, a).
    """
    counts: dict[tuple[State, int], Counter] = defaultdict(Counter)
    for states, actions in zip(episodes_states, episodes_actions):
        T = len(actions)
        for t in range(T):
            s: State = (int(states[t][0]), int(states[t][1]), int(states[t][2]))
            a = int(actions[t])
            sp: State = (int(states[t + 1][0]), int(states[t + 1][1]), int(states[t + 1][2]))
            counts[(s, a)][sp] += 1

    table: dict[tuple[State, int], tuple[list[State], np.ndarray]] = {}
    for key, ctr in counts.items():
        nexts = list(ctr.keys())
        c = np.array([ctr[n] for n in nexts], dtype=np.float64)
        table[key] = (nexts, c / c.sum())
    return table


def sample_obs_transition(
    table: dict[tuple[State, int], tuple[list[State], np.ndarray]],
    state,
    action: int,
    rng: np.random.Generator,
) -> tuple[State, bool] | None:
    """Sample s' ~ P_obs(. | s, a). Returns (next_state, terminated), or None if
    (s, a) was never observed in the data (caller decides the fallback).

    ``terminated`` is derived from the sampled cell (lava or goal), matching
    ``analytical_step`` termination semantics.
    """
    s: State = (int(state[0]), int(state[1]), int(state[2]))
    entry = table.get((s, int(action)))
    if entry is None:
        return None
    nexts, probs = entry
    idx = int(rng.choice(len(nexts), p=probs))
    sp = nexts[idx]
    terminated = is_goal((sp[0], sp[1])) or is_lava((sp[0], sp[1]))
    return sp, terminated
