"""BERT encoder + either a linear head or a Mixture-of-Experts (MoE) head.

MoE head
--------
A shared BERT encoder produces one vector ``h`` per comment. ``E`` expert MLPs each
map ``h`` to logits for **all six labels**; a router (a linear layer + softmax)
decides how much each expert contributes::

    g = softmax(W_g h)                      # (E,) router probabilities
    g = renormalise(top_k(g))               # optional sparse routing (k of E experts)
    logits = sum_i g_i * Expert_i(h)        # (6,)

Because every expert predicts every label, the router can learn to send, e.g.,
identity-based hate and sexual obscenity to different specialists. A Switch-style
load-balancing loss (Fedus et al., 2021) keeps the router from collapsing onto a
single expert:  ``aux = E * sum_i f_i * P_i`` where ``f_i`` is the fraction of
comments whose top-1 expert is ``i`` and ``P_i`` the mean router probability.

This replaces the course-project design, which gave each *label* its own expert and
multiplied each label's logit by a softmax weight computed *across labels* -- that
forced the six independent labels to compete for one unit of probability mass and
shrank every logit by ~6x, so it was not a mixture of experts in the usual sense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelOutput:
    logits: torch.Tensor
    gates: Optional[torch.Tensor] = None  # (batch, experts) routing weights actually used
    aux_loss: Optional[torch.Tensor] = None


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    raise ValueError(f"Unknown pooling {mode!r}")


class LinearHead(nn.Module):
    def __init__(self, hidden: int, num_labels: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, num_labels)

    def forward(self, h: torch.Tensor) -> ModelOutput:
        return ModelOutput(logits=self.out(self.dropout(h)))


class Expert(nn.Module):
    def __init__(self, hidden: int, expert_hidden: int, num_labels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, expert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_hidden, num_labels),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def load_balancing_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """Switch Transformer auxiliary loss; equals 1.0 for a perfectly balanced router."""
    num_experts = router_probs.shape[-1]
    top1 = router_probs.argmax(dim=-1)
    fraction = F.one_hot(top1, num_experts).float().mean(dim=0)  # f_i (no gradient)
    mean_prob = router_probs.float().mean(dim=0)  # P_i (differentiable)
    return num_experts * torch.sum(fraction * mean_prob)


class MoEHead(nn.Module):
    def __init__(
        self,
        hidden: int,
        num_labels: int,
        num_experts: int = 6,
        expert_hidden: int = 256,
        dropout: float = 0.1,
        top_k: int = 2,
    ):
        super().__init__()
        if not 0 <= top_k <= num_experts:
            raise ValueError("top_k must be in [0, num_experts]")
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.router = nn.Linear(hidden, num_experts)
        self.experts = nn.ModuleList([Expert(hidden, expert_hidden, num_labels, dropout) for _ in range(num_experts)])

    def route(self, h: torch.Tensor):
        probs = F.softmax(self.router(h).float(), dim=-1)  # router in fp32 for stability
        if 0 < self.top_k < self.num_experts:
            top_vals, top_idx = probs.topk(self.top_k, dim=-1)
            gates = torch.zeros_like(probs).scatter(-1, top_idx, top_vals)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        else:
            gates = probs
        return probs, gates

    def forward(self, h: torch.Tensor) -> ModelOutput:
        h = self.dropout(h)
        probs, gates = self.route(h)
        # (batch, experts, labels). With only 6 small experts on one pooled vector per
        # comment, running every expert densely is cheaper than gather/scatter dispatch;
        # unselected experts get zero weight (and therefore zero gradient).
        expert_logits = torch.stack([expert(h) for expert in self.experts], dim=1).float()
        logits = (gates.unsqueeze(-1) * expert_logits).sum(dim=1)  # fp32 even under autocast
        return ModelOutput(logits=logits, gates=gates, aux_loss=load_balancing_loss(probs))


class ToxicityClassifier(nn.Module):
    """Pretrained transformer encoder + linear or MoE head."""

    def __init__(
        self,
        encoder: nn.Module,
        num_labels: int = 6,
        arch: str = "bert",
        pooling: str = "cls",
        dropout: float = 0.1,
        num_experts: int = 6,
        expert_hidden: int = 256,
        top_k: int = 2,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.arch = arch
        hidden = encoder.config.hidden_size
        if arch == "bert":
            self.head = LinearHead(hidden, num_labels, dropout)
        elif arch == "moe":
            self.head = MoEHead(hidden, num_labels, num_experts, expert_hidden, dropout, top_k)
        else:
            raise ValueError(f"Unknown arch {arch!r}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(pool(hidden, attention_mask, self.pooling))

    def param_groups(self):
        """(encoder params, head params) for separate learning rates."""
        return list(self.encoder.named_parameters()), list(self.head.named_parameters())


def build_model(cfg, num_labels: int = 6, encoder: Optional[nn.Module] = None) -> ToxicityClassifier:
    if encoder is None:
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(cfg.encoder)
    return ToxicityClassifier(
        encoder,
        num_labels=num_labels,
        arch=cfg.arch,
        pooling=cfg.pooling,
        dropout=cfg.dropout,
        num_experts=cfg.num_experts,
        expert_hidden=cfg.expert_hidden,
        top_k=cfg.top_k,
    )
