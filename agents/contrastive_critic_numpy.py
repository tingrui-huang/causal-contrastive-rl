"""
Dual-encoder contrastive critic (NumPy-only) aligned with Algorithm 1 of
'Contrastive Learning as Goal-Conditioned RL' (Eysenbach et al., 2022).

    f(s, a, s_g) = phi(s, a)^T  psi(s_g) / tau

phi = sa_encoder  — takes (state, action_one_hot), outputs L2-normalized embedding
psi = g_encoder   — takes goal state ONLY (no action), outputs L2-normalized embedding

Critic loss uses **in-batch negatives**: a B x B logit matrix where diagonal
entries are positive pairs and off-diagonal entries are negatives, trained with
sigmoid binary cross-entropy (NCE-binary / InfoMAX objective).
"""
from __future__ import annotations

import numpy as np


def _one_hot(indices: np.ndarray, n_classes: int) -> np.ndarray:
    b = indices.shape[0]
    out = np.zeros((b, n_classes), dtype=np.float64)
    out[np.arange(b), indices.astype(np.int64)] = 1.0
    return out


def _embed_forward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> tuple[np.ndarray, tuple]:
    """2-layer MLP -> L2-normalized embedding.  Returns (h, cache)."""
    z1 = x @ W1 + b1
    h1 = np.maximum(z1, 0.0)
    z2 = h1 @ W2 + b2
    norm = np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8
    h = z2 / norm
    cache = (x, z1, h1, z2, norm, h)
    return h, cache


def _embed_backward(
    dh: np.ndarray,
    cache: tuple,
    W1: np.ndarray,
    W2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backprop through _embed_forward; returns dW1, db1, dW2, db2, dx."""
    x, z1, h1, z2, norm, h = cache
    dz2 = (dh - h * np.sum(dh * h, axis=1, keepdims=True)) / norm
    dW2 = h1.T @ dz2
    db2 = np.sum(dz2, axis=0)
    dh1 = dz2 @ W2.T
    dz1 = dh1 * (z1 > 0).astype(np.float64)
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)
    dx = dz1 @ W1.T
    return dW1, db1, dW2, db2, dx


class ContrastiveCriticNumpy:
    """Dual-encoder contrastive critic with in-batch negatives (Algorithm 1)."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        emb_dim: int = 64,
        tau: float = 0.07,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.tau = tau
        self.emb_dim = emb_dim

        # --- sa_encoder: phi(s, a) ----------------------------------------
        sa_in = state_dim + n_actions
        s1 = np.sqrt(2.0 / sa_in)
        s2 = np.sqrt(2.0 / hidden)
        self.sa_W1 = rng.normal(0.0, s1, (sa_in, hidden)).astype(np.float64)
        self.sa_b1 = np.zeros(hidden, dtype=np.float64)
        self.sa_W2 = rng.normal(0.0, s2, (hidden, emb_dim)).astype(np.float64)
        self.sa_b2 = np.zeros(emb_dim, dtype=np.float64)

        # --- g_encoder: psi(s_g) — NO action input ------------------------
        g_in = state_dim
        g_s1 = np.sqrt(2.0 / g_in)
        g_s2 = np.sqrt(2.0 / hidden)
        self.g_W1 = rng.normal(0.0, g_s1, (g_in, hidden)).astype(np.float64)
        self.g_b1 = np.zeros(hidden, dtype=np.float64)
        self.g_W2 = rng.normal(0.0, g_s2, (hidden, emb_dim)).astype(np.float64)
        self.g_b2 = np.zeros(emb_dim, dtype=np.float64)

    # ----- forward helpers ------------------------------------------------

    def _embed_sa(
        self, s: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, tuple]:
        """phi(s, a): (B, state_dim), (B,) int -> (B, emb_dim)."""
        s = np.asarray(s, dtype=np.float64)
        oh = _one_hot(np.asarray(a, dtype=np.int64), self.n_actions)
        x = np.concatenate([s, oh], axis=1)
        return _embed_forward(x, self.sa_W1, self.sa_b1, self.sa_W2, self.sa_b2)

    def _embed_g(
        self, g: np.ndarray
    ) -> tuple[np.ndarray, tuple]:
        """psi(s_g): (B, state_dim) -> (B, emb_dim)."""
        g = np.asarray(g, dtype=np.float64)
        return _embed_forward(g, self.g_W1, self.g_b1, self.g_W2, self.g_b2)

    # ----- scoring --------------------------------------------------------

    def score_actions(
        self, s: np.ndarray, goals: np.ndarray,
    ) -> np.ndarray:
        """f(s, a_i, s_g) for every discrete action -> (B, n_actions).

        The goal embedding psi(s_g) is computed once and reused across actions.
        """
        B = s.shape[0]
        h_g, _ = self._embed_g(goals)
        scores = np.zeros((B, self.n_actions), dtype=np.float64)
        for a_i in range(self.n_actions):
            a_batch = np.full(B, a_i, dtype=np.int64)
            h_sa, _ = self._embed_sa(s, a_batch)
            scores[:, a_i] = np.sum(h_sa * h_g, axis=1) / self.tau
        return scores

    # ----- contrastive loss (in-batch negatives) --------------------------

    def loss_and_grads(
        self,
        s: np.ndarray,
        a: np.ndarray,
        s_future: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Algorithm 1 critic loss with in-batch negatives.

        logits[i,j] = phi(s_i, a_i)^T psi(s_future_j) / tau   (B x B)
        labels      = eye(B)    — diagonal = positive pairs
        loss        = mean sigmoid_BCE(logits, labels)
        """
        B = s.shape[0]
        h_sa, c_sa = self._embed_sa(s, a)       # (B, E)
        h_g, c_g = self._embed_g(s_future)       # (B, E)

        logits = (h_sa @ h_g.T) / self.tau       # (B, B)
        labels = np.eye(B, dtype=np.float64)

        loss = float(np.mean(
            labels * np.logaddexp(0.0, -logits)
            + (1.0 - labels) * np.logaddexp(0.0, logits)
        ))

        sig = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        dl = (sig - labels) / (B * B)            # (B, B)

        d_h_sa = (dl @ h_g) / self.tau            # (B, E)
        d_h_g = (dl.T @ h_sa) / self.tau          # (B, E)

        sa_dW1, sa_db1, sa_dW2, sa_db2, _ = _embed_backward(
            d_h_sa, c_sa, self.sa_W1, self.sa_W2,
        )
        g_dW1, g_db1, g_dW2, g_db2, _ = _embed_backward(
            d_h_g, c_g, self.g_W1, self.g_W2,
        )

        grads = {
            "sa_W1": sa_dW1, "sa_b1": sa_db1,
            "sa_W2": sa_dW2, "sa_b2": sa_db2,
            "g_W1": g_dW1, "g_b1": g_db1,
            "g_W2": g_dW2, "g_b2": g_db2,
        }
        return loss, grads

    # ----- monitoring -----------------------------------------------------

    def margin(
        self,
        s: np.ndarray,
        a: np.ndarray,
        s_future: np.ndarray,
    ) -> float:
        """Mean positive logit minus mean negative logit (B x B matrix)."""
        h_sa, _ = self._embed_sa(s, a)
        h_g, _ = self._embed_g(s_future)
        logits = (h_sa @ h_g.T) / self.tau
        B = logits.shape[0]
        pos_mean = float(np.trace(logits) / B)
        neg_mean = float((np.sum(logits) - np.trace(logits)) / max(B * B - B, 1))
        return pos_mean - neg_mean

    # ----- optimiser ------------------------------------------------------

    def apply_sgd(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.sa_W1 -= lr * grads["sa_W1"]
        self.sa_b1 -= lr * grads["sa_b1"]
        self.sa_W2 -= lr * grads["sa_W2"]
        self.sa_b2 -= lr * grads["sa_b2"]
        self.g_W1 -= lr * grads["g_W1"]
        self.g_b1 -= lr * grads["g_b1"]
        self.g_W2 -= lr * grads["g_W2"]
        self.g_b2 -= lr * grads["g_b2"]

    # ----- serialisation --------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "sa_W1": self.sa_W1.copy(), "sa_b1": self.sa_b1.copy(),
            "sa_W2": self.sa_W2.copy(), "sa_b2": self.sa_b2.copy(),
            "g_W1": self.g_W1.copy(), "g_b1": self.g_b1.copy(),
            "g_W2": self.g_W2.copy(), "g_b2": self.g_b2.copy(),
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "tau": self.tau,
            "emb_dim": self.emb_dim,
        }
