"""
Single source of truth for WindyCorridor experiment defaults.

WHY THIS EXISTS
---------------
These constants used to be duplicated as literal defaults across several scripts
(`collect_corridor_data.py`, `collect_corridor_worst_case.py`, `expert_policy.py`,
`train_corridor_onehot.py`, ...). That caused a real, silent bug: the observational
data was collected with ``p_near=0.95`` but the worst-case collector defaulted to
``p_near=0.8``, so the two pipelines disagreed on the route mix without any error.

Import these constants instead of re-typing literals. Change a value HERE and every
script that reads it changes together. Argparse flags still override per-run, but
their ``default=`` should point at these names, never a bare literal.

GROUPS
------
* Behavior policy:    P_NEAR — the ONE knob for the near/far demo mix.
* Propensity (P_b):   LAPLACE_ALPHA, PROPENSITY_DEFAULT.
* MDP / geometry:     GAMMA, GOAL_CELL, VALID_ACTIONS.
* Critic net:         TAU, HIDDEN, EMB_DIM, CRITIC_LR, CRITIC_STEPS, BATCH_SIZE.
* Actor:              ACTOR_LR, ACTOR_STEPS, AWR_BETA, ACTOR_LAM, BALANCED_SAMPLING.
* Collection / eval:  NUM_EPISODES, MAX_STEPS, EVAL_EPISODES, SEED.
"""
from __future__ import annotations

# --- Behavior policy (the single p_near knob; was duplicated & drifted) -------
P_NEAR: float = 0.95         # P(expert picks NEAR route) at episode reset. 0.95 => 5% FAR demos.

# --- Propensity P_b(a|s) estimation -------------------------------------------
LAPLACE_ALPHA: float = 1.0   # Laplace smoothing in estimate_propensity.
PROPENSITY_DEFAULT: float = 0.0  # P_b fallback for unseen states (low => more adversarial firings).

# --- MDP / geometry -----------------------------------------------------------
GAMMA: float = 0.99
GOAL_CELL: tuple[int, int] = (13, 1)
VALID_ACTIONS: list[int] = [0, 1, 2, 6]  # left, right, forward, done

# --- Contrastive critic -------------------------------------------------------
TAU: float = 0.07
HIDDEN: int = 128
EMB_DIM: int = 64
CRITIC_LR: float = 2e-3
CRITIC_STEPS: int = 30000
BATCH_SIZE: int = 64

# --- Actor (AWR / paper) ------------------------------------------------------
ACTOR_LR: float = 2e-3
ACTOR_STEPS: int = 15000
AWR_BETA: float = 0.5
ACTOR_LAM: float = 0.5        # BC coefficient for the paper-style (1-λ)adv+λBC actor.
BALANCED_SAMPLING: bool = True  # balance over distinct (s,a) pairs (SUMMARY #5).

# --- Collection / evaluation --------------------------------------------------
NUM_EPISODES: int = 1000
MAX_STEPS: int = 400
EVAL_EPISODES: int = 500
SEED: int = 0
