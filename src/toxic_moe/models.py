"""Pretrained transformer encoder + one of five classification heads.

Heads (``arch``)
----------------
* ``bert``: a linear layer on the pooled vector. The baseline.
* ``mlp``: one hidden layer. With the default 1536 units it has about as many parameters as
  the MoE head, so it separates "more capacity" from "routing".
* ``moe``: a mixture of experts with **one router per comment** (top-k + load balancing).
* ``mmoe``: a multi-gate mixture of experts with **one gate per label** (Ma et al., 2018).
* ``label_attn``: label-wise attention. **Each label pools the token states with its own
  attention** instead of all labels reading the single [CLS] vector (Mullenbach et al., 2018;
  Vu et al., 2020).

MoE head
--------
A shared encoder produces one vector ``h`` per comment. ``E`` expert MLPs each map ``h`` to
logits for **all six labels**; a router (a linear layer + softmax) decides how much each
expert contributes::

    g = softmax(W_g h)                      # (E,) router probabilities
    g = renormalise(top_k(g))               # optional sparse routing (k of E experts)
    logits = sum_i g_i * Expert_i(h)        # (6,)

A Switch-style load-balancing loss (Fedus et al., 2021) keeps the router from collapsing
onto a single expert:  ``aux = E * sum_i f_i * P_i`` where ``f_i`` is the fraction of
comments whose top-1 expert is ``i`` and ``P_i`` the mean router probability.

This replaces the course-project design, which gave each *label* its own expert and
multiplied each label's logit by a softmax weight computed *across labels* -- that
forced the six independent labels to compete for one unit of probability mass and
shrank every logit by ~6x, so it was not a mixture of experts in the usual sense.

The 24-run study found that this router splits toxic from clean comments rather than
specialising by label (see docs/diagnosis.md). ``mmoe`` and ``label_attn`` are the two
heads built to address that: the first lets each label choose its own experts, the
second lets each label read different tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ("bert", "mlp", "moe", "mmoe", "label_attn")


@dataclass
class ModelOutput:
    logits: torch.Tensor
    # routing weights actually used: (batch, experts) for moe, (batch, labels, experts) for mmoe
    gates: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    attention: Optional[torch.Tensor] = None  # label_attn: (batch, labels, tokens)


def pool(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "cls":
        return hidden[:, 0]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    raise ValueError(f"Unknown pooling {mode!r}")


class Head(nn.Module):
    """Base class. ``token_level`` heads receive all token states, the others one pooled vector."""

    token_level = False


class PerLabelLinear(nn.Module):
    """``L`` independent linear outputs: ``logit_l = w_l . x_l + b_l`` for ``x`` of shape (B, L, d).

    Written as an element-wise product + sum, so it stays in fp32 under autocast.
    """

    def __init__(self, num_labels: int, dim: int):
        super().__init__()
        bound = 1.0 / math.sqrt(dim)  # same initialisation as nn.Linear
        self.weight = nn.Parameter(torch.empty(num_labels, dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(num_labels).uniform_(-bound, bound))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.weight).sum(dim=-1) + self.bias


class LinearHead(Head):
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


class MLPHead(Head):
    """A single MLP: exactly one MoE expert, made as wide as all the MoE experts together.

    A control for the MoE head: about the same number of parameters, no routing.
    """

    def __init__(self, hidden: int, num_labels: int, mlp_hidden: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.net = Expert(hidden, mlp_hidden, num_labels, dropout)

    def forward(self, h: torch.Tensor) -> ModelOutput:
        return ModelOutput(logits=self.net(self.dropout(h)))


def load_balancing_loss(router_probs: torch.Tensor) -> torch.Tensor:
    """Switch Transformer auxiliary loss; equals 1.0 for a perfectly balanced router."""
    num_experts = router_probs.shape[-1]
    top1 = router_probs.argmax(dim=-1)
    fraction = F.one_hot(top1, num_experts).float().mean(dim=0)  # f_i (no gradient)
    mean_prob = router_probs.float().mean(dim=0)  # P_i (differentiable)
    return num_experts * torch.sum(fraction * mean_prob)


class MoEHead(Head):
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


class MMoEHead(Head):
    """Multi-gate mixture of experts (Ma et al., 2018): shared experts, one gate per label.

    The experts turn ``h`` into shared representations; every *label* has its own softmax
    gate over the experts and its own output layer::

        z_e = Expert_e(h)                        # (E, d), shared by all labels
        g_l = softmax(W_l h)                     # (E,), one gate per label
        logit_l = w_l . sum_e g_{l,e} z_e + b_l

    The MoE head's router chooses one mixture per comment, so every label of that comment
    goes through the same experts. Here the label chooses, so ``threat`` and ``obscene`` can
    use different experts for the same comment. The gates are dense softmaxes (no top-k), so
    there is nothing to collapse and no load-balancing loss.
    """

    def __init__(
        self, hidden: int, num_labels: int, num_experts: int = 6, expert_hidden: int = 256, dropout: float = 0.1
    ):
        super().__init__()
        self.num_labels, self.num_experts = num_labels, num_experts
        self.dropout = nn.Dropout(dropout)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden, expert_hidden), nn.GELU(), nn.Dropout(dropout))
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(hidden, num_labels * num_experts)
        self.towers = PerLabelLinear(num_labels, expert_hidden)

    def forward(self, h: torch.Tensor) -> ModelOutput:
        h = self.dropout(h)
        z = torch.stack([expert(h) for expert in self.experts], dim=1).float()  # (B, E, d)
        gate_logits = self.gate(h).float().view(-1, self.num_labels, self.num_experts)
        gates = F.softmax(gate_logits, dim=-1)  # (B, L, E)
        mixed = (gates.unsqueeze(-1) * z.unsqueeze(1)).sum(dim=2)  # (B, L, d), fp32
        return ModelOutput(logits=self.towers(mixed), gates=gates)


class LabelAttentionHead(Head):
    """Label-wise attention, as in LAAT (Vu et al., 2020) and CAML (Mullenbach et al., 2018).

    Each label attends over the comment's tokens with its own query and pools the token
    states with that attention, instead of all labels sharing the [CLS] vector::

        a_l = softmax_t( u_l . tanh(W H_t) )     # (T,) label l's attention over tokens
        r_l = sum_t a_{l,t} H_t                  # (d,) label-specific comment vector
        logit_l = w_l . r_l + b_l

    A threat buried in a long comment can then drive ``threat`` without having to dominate
    the whole-comment summary. The attention weights also show which words drove each label.
    """

    token_level = True

    def __init__(self, hidden: int, num_labels: int, attn_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, attn_hidden, bias=False)  # W
        self.queries = nn.Linear(attn_hidden, num_labels, bias=False)  # one query u_l per label
        self.out = PerLabelLinear(num_labels, hidden)

    def forward(self, states: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        states = self.dropout(states)  # (B, T, d)
        scores = self.queries(torch.tanh(self.proj(states))).float()  # (B, T, L)
        scores = scores.masked_fill(attention_mask.unsqueeze(-1) == 0, float("-inf"))  # ignore padding
        attn = F.softmax(scores, dim=1).transpose(1, 2)  # (B, L, T); every row has >= 1 real token
        with torch.autocast(device_type=states.device.type, enabled=False):  # pool in fp32 under autocast too
            pooled = torch.bmm(attn, states.float())  # (B, L, d)
        return ModelOutput(logits=self.out(pooled), attention=attn)


def build_head(
    arch: str,
    hidden: int,
    num_labels: int,
    dropout: float = 0.1,
    num_experts: int = 6,
    expert_hidden: int = 256,
    top_k: int = 2,
    mlp_hidden: int = 1536,
    attn_hidden: int = 256,
) -> Head:
    if arch == "bert":
        return LinearHead(hidden, num_labels, dropout)
    if arch == "mlp":
        return MLPHead(hidden, num_labels, mlp_hidden, dropout)
    if arch == "moe":
        return MoEHead(hidden, num_labels, num_experts, expert_hidden, dropout, top_k)
    if arch == "mmoe":
        return MMoEHead(hidden, num_labels, num_experts, expert_hidden, dropout)
    if arch == "label_attn":
        return LabelAttentionHead(hidden, num_labels, attn_hidden, dropout)
    raise ValueError(f"Unknown arch {arch!r}; expected one of {HEADS}")


class ToxicityClassifier(nn.Module):
    """Pretrained transformer encoder + one of the heads above."""

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
        mlp_hidden: int = 1536,
        attn_hidden: int = 256,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.arch = arch
        self.freeze_encoder = freeze_encoder
        hidden = encoder.config.hidden_size
        self.head = build_head(
            arch, hidden, num_labels, dropout, num_experts, expert_hidden, top_k, mlp_hidden, attn_hidden
        )
        if freeze_encoder:
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True) -> ToxicityClassifier:
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # a frozen encoder is a fixed feature extractor: no dropout
        return self

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ModelOutput:
        # a frozen encoder needs no activation graph, which saves memory and backward time
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.head.token_level:
            return self.head(states, attention_mask)
        return self.head(pool(states, attention_mask, self.pooling))

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
        mlp_hidden=cfg.mlp_hidden,
        attn_hidden=cfg.attn_hidden,
        freeze_encoder=cfg.freeze_encoder,
    )
