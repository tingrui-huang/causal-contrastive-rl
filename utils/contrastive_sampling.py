"""
Contrastive tuple construction: (s, a, s+, s-).

Design (Phase 5 baseline):
- Fixed ``k`` (default ``K=4``): positive is the processed state at ``t+k`` in the
  same trajectory as anchor ``t``.
- Negative: ``state`` field from a uniformly random transition in the replay buffer.
- Anchor indices ``t`` are sampled uniformly from valid set ``{0, ..., T-1-k}``
  with replacement (standard minibatch).
"""
from __future__ import annotations

import numpy as np

from utils.preprocess import extract_state

# Default horizon offset for positive pairs (fixed per project decision).
# Align with HiddenFork: U0 vs U1 observations first diverge after k=3 forward steps, so k=4
# positives (s_{t+4}) can lie past that boundary for early anchors.
DEFAULT_K = 4


def _trajectory_states(trajectory: list[dict]) -> np.ndarray:
    """Stack processed states ``s_t = extract_state(obs_t)`` for each step ``t``."""
    return np.stack([extract_state(tr["obs"]) for tr in trajectory], axis=0)


def build_contrastive_batch(
    trajectory: list[dict],
    buffer,
    *,
    k: int = DEFAULT_K,
    batch_size: int = 32,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """
    Build a batch of contrastive tuples.

    Valid anchor indices: ``t`` such that ``0 <= t <= T - 1 - k`` (i.e. ``T - k`` anchors).

    Parameters
    ----------
    trajectory
        Raw rollout from ``rollout_episode`` (ordered list of transitions).
    buffer
        ``ReplayBuffer`` with processed items (must include ``"state"`` keys).
    k
        Positive offset; positive state is ``s_{t+k}`` from the same trajectory.
    batch_size
        Number of anchors to draw (with replacement from valid anchors).
    rng
        Optional NumPy Generator; default ``numpy.random.default_rng()``.

    Returns
    -------
    dict
        ``s``, ``a``, ``s_pos``, ``s_neg`` as batched arrays
        (``s`` / ``s_pos`` / ``s_neg``: float32, shape ``(B, dim)``;
        ``a``: int64, shape ``(B,)``).
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(trajectory)
    if T < k + 1:
        raise ValueError(
            f"Trajectory length T={T} must be at least k+1={k+1} for k={k}."
        )

    num_valid = T - k
    if num_valid < 1:
        raise ValueError("No valid anchor indices (tail shorter than k).")

    states = _trajectory_states(trajectory)
    actions = np.array([tr["action"] for tr in trajectory], dtype=np.int64)

    anchor_t = rng.integers(low=0, high=num_valid, size=batch_size, endpoint=False)

    s_batch = states[anchor_t]
    a_batch = actions[anchor_t]
    s_pos_batch = states[anchor_t + k]

    s_neg_batch = np.empty_like(s_pos_batch)
    for i in range(batch_size):
        neg_item = buffer.sample(1)[0]
        s_neg_batch[i] = np.asarray(neg_item["state"], dtype=np.float32)

    return {
        "s": s_batch.astype(np.float32, copy=False),
        "a": a_batch,
        "s_pos": s_pos_batch.astype(np.float32, copy=False),
        "s_neg": s_neg_batch.astype(np.float32, copy=False),
    }
