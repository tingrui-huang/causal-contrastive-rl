"""
Empirical behavior-policy propensity P_b(x | s) from observational data.

Theorem 2's (ii) branch fires w.p. P(x | s) — the behavior policy's probability
of selecting action x at state s in the data-generating distribution. For our
collected expert data, the behavior policy is deterministic given (s, wind),
but marginalising over wind makes it stochastic in s alone.

We estimate P_b(x | s) by counting (s, a) co-occurrences in the trajectories.
State is canonicalised to integer (x, y, dir). Laplace smoothing handles
unseen (s, a) pairs.

For states never visited in the data, ``lookup`` returns ``default_p`` — caller
should choose this conservatively (low default = more (iii) firings = more
worst-case influence).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def estimate_propensity(
    episodes_states: list[np.ndarray],
    episodes_actions: list[np.ndarray],
    *,
    n_actions: int,
    laplace_alpha: float = 1.0,
) -> dict[tuple[int, int, int], np.ndarray]:
    """Return mapping (x, y, dir) → np.array of length n_actions giving P_b(a | s).

    Only canonical anchor states (excluding each trajectory's terminal state)
    are counted, since the terminal state has no associated action.
    """
    sa_counts: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
    s_counts: Counter[tuple[int, int, int]] = Counter()

    for ep_states, ep_actions in zip(episodes_states, episodes_actions):
        T = len(ep_actions)
        for t in range(T):
            s = (int(ep_states[t][0]), int(ep_states[t][1]), int(ep_states[t][2]))
            a = int(ep_actions[t])
            sa_counts[s][a] += 1
            s_counts[s] += 1

    table: dict[tuple[int, int, int], np.ndarray] = {}
    for s, counter in sa_counts.items():
        probs = np.full(n_actions, laplace_alpha, dtype=np.float64)
        for a, c in counter.items():
            probs[a] += c
        probs /= probs.sum()
        table[s] = probs
    return table


def lookup(
    table: dict[tuple[int, int, int], np.ndarray],
    state,
    action: int,
    *,
    default_p: float = 0.0,
) -> float:
    """P_b(action | state) with fallback for unseen states."""
    s = (int(state[0]), int(state[1]), int(state[2]))
    probs = table.get(s)
    if probs is None:
        return float(default_p)
    return float(probs[int(action)])
