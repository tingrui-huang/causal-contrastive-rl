"""
NumPy-only contrastive critic (same intent as ``contrastive_critic.py``).

Used when PyTorch cannot be imported (e.g. broken CUDA DLL on Windows).
SGD on a 2-layer MLP + row L2-normalized embeddings; 2-way softmax contrastive loss.
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
    """Returns normalized embedding h and cache for backward."""
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
    """Backprop through embed; returns dW1, db1, dW2, db2, dx."""
    x, z1, h1, z2, norm, h = cache
    # h = z2 / norm
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
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        emb_dim: int = 64,
        tau: float = 0.07,
        seed: int = 0,
        loss_family: str = "sigmoid_bce_weight1",
    ) -> None:
        rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.tau = tau
        self.loss_family = loss_family
        in_dim = state_dim + n_actions
        # He-ish init
        s1 = np.sqrt(2.0 / in_dim)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0.0, s1, (in_dim, hidden)).astype(np.float64)
        self.b1 = np.zeros(hidden, dtype=np.float64)
        self.W2 = rng.normal(0.0, s2, (hidden, emb_dim)).astype(np.float64)
        self.b2 = np.zeros(emb_dim, dtype=np.float64)

    def _embed(self, s: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, tuple]:
        """s, a: batch (B, dim) and (B,) int."""
        s = np.asarray(s, dtype=np.float64)
        oh = _one_hot(a, self.n_actions)
        x = np.concatenate([s, oh], axis=1)
        return _embed_forward(x, self.W1, self.b1, self.W2, self.b2)

    def logits(
        self,
        s: np.ndarray,
        a: np.ndarray,
        s_pos: np.ndarray,
        s_neg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch pos/neg logits (same definition as in loss)."""
        ha, _ = self._embed(s, a)
        hpos, _ = self._embed(s_pos, a)
        hneg, _ = self._embed(s_neg, a)
        pos_logit = np.sum(ha * hpos, axis=1) / self.tau
        neg_logit = np.sum(ha * hneg, axis=1) / self.tau
        return pos_logit, neg_logit

    def loss_and_grads(
        self,
        s: np.ndarray,
        a: np.ndarray,
        s_pos: np.ndarray,
        s_neg: np.ndarray,
    ) -> tuple[float, dict[str, np.ndarray]]:
        """Mean contrastive loss and param grads."""
        b = s.shape[0]
        ha, ca = self._embed(s, a)
        hpos, cpos = self._embed(s_pos, a)
        hneg, cneg = self._embed(s_neg, a)

        pos_logit = np.sum(ha * hpos, axis=1) / self.tau
        neg_logit = np.sum(ha * hneg, axis=1) / self.tau

        if self.loss_family == "softmax_ce":
            m = np.maximum(np.maximum(pos_logit, neg_logit), 0.0)
            e0 = np.exp(pos_logit - m)
            e1 = np.exp(neg_logit - m)
            denom = e0 + e1
            p0 = e0 / denom
            p1 = e1 / denom
            loss = -np.mean(np.log(p0 + 1e-12))
            d_lp = (p0 - 1.0) / b
            d_ln = p1 / b
        elif self.loss_family == "sigmoid_bce_weight1":
            loss = float(
                np.mean(np.logaddexp(0.0, -pos_logit) + np.logaddexp(0.0, neg_logit))
            )
            sig_pos = 1.0 / (1.0 + np.exp(-pos_logit))
            sig_neg = 1.0 / (1.0 + np.exp(-neg_logit))
            d_lp = (sig_pos - 1.0) / b
            d_ln = sig_neg / b
        else:
            raise ValueError(f"Unknown loss_family: {self.loss_family!r}")

        d_ha = (d_lp[:, None] * hpos + d_ln[:, None] * hneg) / self.tau
        d_hpos = (d_lp[:, None] * ha) / self.tau
        d_hneg = (d_ln[:, None] * ha) / self.tau

        gW1 = np.zeros_like(self.W1)
        gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2)
        gb2 = np.zeros_like(self.b2)

        for dh, cache in [(d_ha, ca), (d_hpos, cpos), (d_hneg, cneg)]:
            dW1, db1, dW2, db2, _ = _embed_backward(dh, cache, self.W1, self.W2)
            gW1 += dW1
            gb1 += db1
            gW2 += dW2
            gb2 += db2

        return float(loss), {
            "W1": gW1,
            "b1": gb1,
            "W2": gW2,
            "b2": gb2,
        }

    def apply_sgd(self, grads: dict[str, np.ndarray], lr: float) -> None:
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def state_dict(self) -> dict:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "tau": self.tau,
            "loss_family": self.loss_family,
        }
