"""
Minimal contrastive critic (Phase 6 baseline, non-causal).

Architecture: MLP on ``concat(state, one_hot(action))`` → L2-normalized embedding;
scores are dot products / temperature-scaled 2-way softmax (positive vs one negative).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveCritic(nn.Module):
    """
    Binary contrastive loss per anchor: classify (s_pos vs s_neg) given (s, a).

    Uses the **same** action embedding for anchor, positive, and negative states
    (minimal baseline; causal variants can change this later).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 128,
        emb_dim: int = 64,
        tau: float = 0.07,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.tau = tau
        in_dim = state_dim + n_actions
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim),
        )

    def embed(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``s`` (B, state_dim), ``a`` (B,) long."""
        oh = F.one_hot(a.long(), num_classes=self.n_actions).float()
        x = torch.cat([s, oh], dim=-1)
        h = self.encoder(x)
        return F.normalize(h, dim=-1)

    def forward(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        s_pos: torch.Tensor,
        s_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns scalar loss (mean over batch).
        """
        h = self.embed(s, a)
        h_pos = self.embed(s_pos, a)
        h_neg = self.embed(s_neg, a)
        pos_logit = (h * h_pos).sum(dim=-1) / self.tau
        neg_logit = (h * h_neg).sum(dim=-1) / self.tau
        logits = torch.stack([pos_logit, neg_logit], dim=1)
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)
