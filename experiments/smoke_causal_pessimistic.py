from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.causal_pessimistic_actor import CausalPessimisticActorNumpy
from agents.contrastive_critic_numpy import ContrastiveCriticNumpy


def main() -> None:
    rng = np.random.default_rng(0)
    state_dim = 4
    n_actions = 3
    emb_dim = 8
    batch_size = 2
    neg_n = 5

    critic = ContrastiveCriticNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=16,
        emb_dim=emb_dim,
        seed=0,
    )

    actor_min = CausalPessimisticActorNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=16,
        seed=1,
        pessimism_mode="min_neg",
        constant_M=2.0,
    )
    actor_const = CausalPessimisticActorNumpy(
        state_dim=state_dim,
        n_actions=n_actions,
        hidden=16,
        seed=1,
        pessimism_mode="constant",
        constant_M=2.0,
    )

    s = rng.normal(size=(batch_size, state_dim))
    goals = rng.normal(size=(batch_size, state_dim))
    a_obs = np.array([0, 2], dtype=np.int64)
    neg_goals = np.concatenate(
        [goals, rng.normal(size=(neg_n, state_dim))],
        axis=0,
    )

    critic_scores = critic.score_actions(s, goals)
    pessimistic_q, diag = actor_min.build_pessimistic_scores(
        state=s,
        a_obs=a_obs,
        critic_scores=critic_scores,
        neg_goals=neg_goals,
        phi_fn=lambda ss, aa: critic._embed_sa(ss, aa)[0],
        psi_fn=lambda gg: critic._embed_g(gg)[0],
        score_scale=1.0 / critic.tau,
    )
    constant_q, _ = actor_const.build_pessimistic_scores(
        state=s,
        a_obs=a_obs,
        critic_scores=critic_scores,
        neg_goals=None,
        phi_fn=None,
        psi_fn=None,
    )

    observed_mask = np.zeros_like(critic_scores, dtype=bool)
    observed_mask[np.arange(batch_size), a_obs] = True
    counterfactual_mask = ~observed_mask
    expected_const = np.min(critic_scores, axis=1, keepdims=True) - actor_const.constant_M

    observed_action_match = bool(
        np.allclose(
            pessimistic_q[observed_mask],
            critic_scores[observed_mask],
            atol=1e-10,
            rtol=0.0,
        )
    )
    counterfactual_pessimism = bool(
        np.all(pessimistic_q[counterfactual_mask] <= critic_scores[counterfactual_mask] + 1e-10)
    )
    constant_mode_exact = bool(
        np.allclose(
            constant_q[counterfactual_mask],
            np.broadcast_to(expected_const, constant_q.shape)[counterfactual_mask],
            atol=1e-10,
            rtol=0.0,
        )
    )

    print(f"[Phase9.2] critic_scores shape = {critic_scores.shape}")
    print(f"[Phase9.2] pessimistic_Q shape = {pessimistic_q.shape}")
    print(f"[Phase9.2] observed_action_match = {observed_action_match}")
    print(f"[Phase9.2] counterfactual_pessimism = {counterfactual_pessimism}")
    print(f"[Phase9.2] constant_mode_exact = {constant_mode_exact}")
    print(f"[Phase9.2] pessimism_gap = {diag['pessimism_gap']:.6f}")
    if not (observed_action_match and counterfactual_pessimism and constant_mode_exact):
        raise SystemExit("[Phase9.2] FAIL")
    print("[Phase9.2] PASS")


if __name__ == "__main__":
    main()
