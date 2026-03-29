"""
Goal-conditioned actor (NumPy-only, discrete actions).

Paper reference: Section 5.5, Eq. 7-8 of "Contrastive Learning as Goal-Conditioned RL".
Offline RL loss: max_π  E[(1-λ)·f(s,a,sg) + λ·log π(a_orig|s,sg)]

For discrete actions we can enumerate all actions exactly — no REINFORCE needed.
"""
from __future__ import annotations

import numpy as np


class GoalConditionedActorNumpy:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        seed: int = 0,
    ) -> None:
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

    def _forward(
        self, state: np.ndarray, goal: np.ndarray
    ) -> tuple[np.ndarray, tuple]:
        x = np.concatenate(
            [np.asarray(state, dtype=np.float64),
             np.asarray(goal, dtype=np.float64)], axis=1
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
        """Single state/goal → best action (restricted to valid_actions if given)."""
        z2, _ = self._forward(state[None], goal[None])
        logits = z2[0]
        if valid_actions is not None:
            mask = np.full(self.n_actions, -np.inf)
            for a in valid_actions:
                mask[a] = logits[a]
            return int(np.argmax(mask))
        return int(np.argmax(logits))

    def sample_action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        valid_actions: list[int] | None = None,
        rng: np.random.Generator | None = None,
        temperature: float = 1.0,
    ) -> int:
        """Sample from softmax distribution (restricted to valid_actions)."""
        if rng is None:
            rng = np.random.default_rng()
        z2, _ = self._forward(state[None], goal[None])
        logits = z2[0] / temperature
        if valid_actions is not None:
            filtered = np.array([logits[a] for a in valid_actions])
            filtered = filtered - np.max(filtered)
            exp_f = np.exp(filtered)
            probs = exp_f / np.sum(exp_f)
            idx = rng.choice(len(valid_actions), p=probs)
            return valid_actions[idx]
        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        probs = exp_l / np.sum(exp_l)
        return int(rng.choice(self.n_actions, p=probs))

    def loss_and_grads(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        a_orig: np.ndarray,
        critic_scores: np.ndarray,
        lam: float = 0.5,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """
        Parameters
        ----------
        state : (B, state_dim)
        goal  : (B, state_dim)
        a_orig : (B,) int — actions from the dataset
        critic_scores : (B, n_actions) — f(s, a_i, sg) for all actions (detached)
        lam : BC coefficient (1 = pure BC, 0 = pure advantage)
        """
        B = state.shape[0]
        z2, (x, z1, h1) = self._forward(state, goal)

        z2_stable = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z = np.exp(z2_stable)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        V = np.sum(probs * critic_scores, axis=1)
        adv_loss = -(1.0 - lam) * np.mean(V)

        log_sum = np.log(np.sum(exp_z, axis=1))
        log_pi_a = z2_stable[np.arange(B), a_orig.astype(np.int64)] - log_sum
        bc_loss = -lam * np.mean(log_pi_a)

        total_loss = float(adv_loss + bc_loss)

        one_hot_a = np.zeros_like(probs)
        one_hot_a[np.arange(B), a_orig.astype(np.int64)] = 1.0

        dz2 = (
            -(1.0 - lam) * probs * (critic_scores - V[:, None])
            + lam * (probs - one_hot_a)
        ) / B

        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0).astype(np.float64)
        dW1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return total_loss, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def apply_sgd(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def state_dict(self) -> dict:
        return {
            "actor_W1": self.W1.copy(),
            "actor_b1": self.b1.copy(),
            "actor_W2": self.W2.copy(),
            "actor_b2": self.b2.copy(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "actor_hidden": self.hidden,
        }
