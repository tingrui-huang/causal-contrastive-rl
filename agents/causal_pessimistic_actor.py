"""
Causal-pessimistic goal-conditioned actor (NumPy-only, discrete actions).

This actor keeps the exact same policy parameterization and BC-regularized
objective structure as `GoalConditionedActorNumpy`, but replaces the raw critic
scores with a pessimistic lower bound for counterfactual actions.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


class CausalPessimisticActorNumpy:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        seed: int = 0,
        pessimism_mode: str = "neighbor",
        constant_M: float = 2.0,
    ) -> None:
        if pessimism_mode not in {"min_neg", "constant", "neighbor"}:
            raise ValueError(
                f"Unknown pessimism_mode={pessimism_mode!r}; "
                f"expected 'min_neg', 'constant', or 'neighbor'."
            )
        rng = np.random.default_rng(seed)
        in_dim = state_dim * 2
        s1 = np.sqrt(2.0 / in_dim)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0.0, s1, (in_dim, hidden)).astype(np.float64)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.normal(0.0, s2, (hidden, n_actions)).astype(np.float64)
        self.b2 = np.zeros(n_actions, dtype=np.float64)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden = hidden
        self.pessimism_mode = pessimism_mode
        self.constant_M = float(constant_M)

    def _forward(
        self, state: np.ndarray, goal: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        x = np.concatenate(
            [np.asarray(state, dtype=np.float64), np.asarray(goal, dtype=np.float64)],
            axis=1,
        )
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2 + self.b2
        return z2, (x, z1, h1)

    def action_probs(self, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        z2, _ = self._forward(state, goal)
        z2 = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def greedy_action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        valid_actions: list[int] | None = None,
    ) -> int:
        z2, _ = self._forward(state[None], goal[None])
        logits = z2[0]
        if valid_actions is not None:
            masked = np.full(self.n_actions, -np.inf, dtype=np.float64)
            for a in valid_actions:
                masked[a] = logits[a]
            return int(np.argmax(masked))
        return int(np.argmax(logits))

    def sample_action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        valid_actions: list[int] | None = None,
        rng: np.random.Generator | None = None,
        temperature: float = 1.0,
    ) -> int:
        if rng is None:
            rng = np.random.default_rng()
        z2, _ = self._forward(state[None], goal[None])
        logits = z2[0] / temperature
        if valid_actions is not None:
            filtered = np.array([logits[a] for a in valid_actions], dtype=np.float64)
            filtered = filtered - np.max(filtered)
            exp_f = np.exp(filtered)
            probs = exp_f / np.sum(exp_f)
            idx = rng.choice(len(valid_actions), p=probs)
            return int(valid_actions[idx])
        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / np.sum(exp_l)
        return int(rng.choice(self.n_actions, p=probs))

    def build_pessimistic_scores(
        self,
        state: np.ndarray,
        a_obs: np.ndarray,
        critic_scores: np.ndarray,
        neg_goals: np.ndarray | None,
        phi_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
        psi_fn: Callable[[np.ndarray], np.ndarray] | None,
        score_scale: float = 1.0,
        neighbor_states: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        state = np.asarray(state, dtype=np.float64)
        a_obs = np.asarray(a_obs, dtype=np.int64)
        critic_scores = np.asarray(critic_scores, dtype=np.float64)

        if critic_scores.shape != (state.shape[0], self.n_actions):
            raise ValueError(
                f"critic_scores has shape {critic_scores.shape}, expected "
                f"({state.shape[0]}, {self.n_actions})."
            )

        B = state.shape[0]
        pessimistic = np.array(critic_scores, copy=True)
        counterfactual_mask = np.ones((B, self.n_actions), dtype=bool)
        counterfactual_mask[np.arange(B), a_obs] = False

        if self.pessimism_mode == "neighbor":
            if phi_fn is None or psi_fn is None:
                raise ValueError("neighbor mode requires phi_fn and psi_fn.")
            if neighbor_states is None:
                raise ValueError("neighbor mode requires neighbor_states.")

            neighbor_embs = [psi_fn(ns) for ns in neighbor_states]

            for action_idx in range(self.n_actions):
                action_batch = np.full(B, action_idx, dtype=np.int64)
                h_sa = phi_fn(state, action_batch)
                for i in range(B):
                    if action_idx == a_obs[i]:
                        continue
                    scores_i = score_scale * (h_sa[i : i + 1] @ neighbor_embs[i].T)
                    pessimistic[i, action_idx] = min(
                        critic_scores[i, action_idx],
                        float(np.min(scores_i)),
                    )

        elif self.pessimism_mode == "min_neg":
            if neg_goals is None or phi_fn is None or psi_fn is None:
                raise ValueError("min_neg mode requires neg_goals, phi_fn, and psi_fn.")
            neg_goals = np.asarray(neg_goals, dtype=np.float64)
            h_neg = psi_fn(neg_goals)
            if h_neg.ndim != 2:
                raise ValueError(f"psi_fn must return a 2D array, got shape {h_neg.shape}.")

            for action_idx in range(self.n_actions):
                action_batch = np.full(B, action_idx, dtype=np.int64)
                h_sa = phi_fn(state, action_batch)
                if h_sa.ndim != 2:
                    raise ValueError(
                        f"phi_fn must return a 2D array, got shape {h_sa.shape}."
                    )
                neg_scores = score_scale * (h_sa @ h_neg.T)
                pessimistic[:, action_idx] = np.minimum(
                    critic_scores[:, action_idx],
                    np.min(neg_scores, axis=1),
                )

        elif self.pessimism_mode == "constant":
            score_min = np.min(critic_scores, axis=1, keepdims=True)
            pessimistic[:, :] = score_min - self.constant_M
        else:
            raise RuntimeError(f"Unhandled pessimism_mode={self.pessimism_mode!r}")

        pessimistic[np.arange(B), a_obs] = critic_scores[np.arange(B), a_obs]

        observed_match = bool(
            np.allclose(
                pessimistic[np.arange(B), a_obs],
                critic_scores[np.arange(B), a_obs],
                atol=1e-10,
                rtol=0.0,
            )
        )
        if np.any(counterfactual_mask):
            pessimism_gap = float(
                np.mean(critic_scores[counterfactual_mask])
                - np.mean(pessimistic[counterfactual_mask])
            )
            counterfactual_pessimism = bool(
                np.all(pessimistic[counterfactual_mask] <= critic_scores[counterfactual_mask] + 1e-10)
            )
        else:
            pessimism_gap = 0.0
            counterfactual_pessimism = True

        diagnostics = {
            "critic_score_min": float(np.min(critic_scores)),
            "critic_score_mean": float(np.mean(critic_scores)),
            "critic_score_max": float(np.max(critic_scores)),
            "pessimism_gap": pessimism_gap,
            "observed_action_match": float(observed_match),
            "counterfactual_pessimism": float(counterfactual_pessimism),
            "num_counterfactual_penalties": float(np.sum(counterfactual_mask)),
        }
        if self.pessimism_mode == "constant":
            row_score_min = np.min(critic_scores, axis=1)
            diagnostics["effective_constant_penalty_mean"] = float(
                np.mean(row_score_min - self.constant_M)
            )
        return pessimistic, diagnostics

    def loss_and_grads(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        a_obs: np.ndarray,
        critic_scores: np.ndarray,
        neg_goals: np.ndarray | None,
        phi_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
        psi_fn: Callable[[np.ndarray], np.ndarray] | None,
        lam: float = 0.5,
        score_scale: float = 1.0,
        neighbor_states: list[np.ndarray] | None = None,
    ) -> tuple[float, dict[str, np.ndarray], dict[str, float]]:
        """
        Same objective structure as the baseline actor, but evaluated against a
        pessimistic score table instead of the raw critic score table.
        """
        B = state.shape[0]
        pessimistic_scores, diagnostics = self.build_pessimistic_scores(
            state=state,
            a_obs=a_obs,
            critic_scores=critic_scores,
            neg_goals=neg_goals,
            phi_fn=phi_fn,
            psi_fn=psi_fn,
            score_scale=score_scale,
            neighbor_states=neighbor_states,
        )

        z2, (x, z1, h1) = self._forward(state, goal)
        z2_stable = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2_stable)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        V = np.sum(probs * pessimistic_scores, axis=1)
        adv_loss = -(1.0 - lam) * np.mean(V)

        log_sum = np.log(np.sum(exp_z, axis=1))
        log_pi_a = z2_stable[np.arange(B), a_obs.astype(np.int64)] - log_sum
        bc_loss = -lam * np.mean(log_pi_a)
        total_loss = float(adv_loss + bc_loss)

        one_hot_a = np.zeros_like(probs)
        one_hot_a[np.arange(B), a_obs.astype(np.int64)] = 1.0
        dz2 = (
            -(1.0 - lam) * probs * (pessimistic_scores - V[:, None])
            + lam * (probs - one_hot_a)
        ) / B

        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0).astype(np.float64)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)
        diagnostics["mean_policy_entropy"] = float(
            -np.mean(np.sum(probs * np.log(probs + 1e-12), axis=1))
        )
        return total_loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, diagnostics

    def apply_sgd(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor_W1": self.W1.copy(),
            "actor_b1": self.b1.copy(),
            "actor_W2": self.W2.copy(),
            "actor_b2": self.b2.copy(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "actor_hidden": self.hidden,
            "pessimism_mode": self.pessimism_mode,
            "constant_M": self.constant_M,
        }
