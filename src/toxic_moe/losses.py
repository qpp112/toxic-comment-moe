r"""Loss functions for imbalanced multi-label classification.

All losses take raw logits ``z`` and 0/1 targets ``y`` of shape ``(batch, labels)``
and average over every (sample, label) entry.

* ``bce``          - plain binary cross-entropy.
* ``weighted_bce`` - BCE with ``pos_weight_c = #neg_c / #pos_c`` (inverse frequency).
  For ``threat`` that weight is ~330 on the full data, which over-predicts positives.
* ``focal``        - focal loss (Lin et al., 2017): scales each term by
  :math:`(1-p_t)^\gamma`, so the flood of easy negatives stops dominating.
* ``cb_focal``     - class-balanced focal loss (Cui et al., 2019), adapted to
  multi-label sigmoid outputs. Instead of raw counts it uses the *effective number*
  of samples :math:`E(n) = (1-\beta^n)/(1-\beta)`, which saturates as ``n`` grows
  (extra near-duplicate examples add little new information). Each label's positive
  term is weighted by :math:`E(n^-_c)/E(n^+_c)` (negative term weight 1) and then the
  focal modulation is applied:

  .. math::
      \ell_c = -\,w_c\,y\,(1-p)^\gamma \log p \;-\; (1-y)\,p^\gamma \log(1-p),
      \qquad w_c = \frac{E(n^-_c)}{E(n^+_c)}.

  On the full training split with :math:`\beta = 0.9999` this gives ``threat`` a
  weight of ~24 instead of ~333, while ``toxic`` (~10% positive) gets ~1.3 instead of ~9.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number(n: np.ndarray, beta: float) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    if beta <= 0:
        return np.ones_like(n)
    if beta >= 1:
        return n
    return (1.0 - np.power(beta, n)) / (1.0 - beta)


def inverse_frequency_pos_weight(pos: np.ndarray, total: int) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    return (total - pos) / np.maximum(pos, 1.0)


def class_balanced_pos_weight(pos: np.ndarray, total: int, beta: float) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    neg = total - pos
    return effective_number(neg, beta) / np.maximum(effective_number(pos, beta), 1e-12)


def focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 0.0,
    pos_weight: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Element-wise (weighted) focal binary cross-entropy. ``gamma=0`` gives (weighted) BCE."""
    logits = logits.float()
    targets = targets.float()
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma > 0:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = loss * (1 - p_t).clamp_min(0).pow(gamma)
    if alpha is not None:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))
    if pos_weight is not None:
        loss = loss * (pos_weight * targets + (1 - targets))
    return loss


class MultiLabelLoss(nn.Module):
    """One module covering every loss variant, so the trainer has a single code path."""

    def __init__(self, gamma: float = 0.0, pos_weight: Optional[np.ndarray] = None, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.as_tensor(pos_weight, dtype=torch.float32))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return focal_bce(logits, targets, self.gamma, self.pos_weight, self.alpha).mean()

    def extra_repr(self) -> str:
        pw = None if self.pos_weight is None else [round(float(x), 2) for x in self.pos_weight]
        return f"gamma={self.gamma}, alpha={self.alpha}, pos_weight={pw}"


def build_loss(
    name: str,
    pos_counts: np.ndarray,
    total: int,
    gamma: float = 2.0,
    alpha: Optional[float] = None,
    beta: float = 0.9999,
) -> MultiLabelLoss:
    """Build a loss from the *training split's* label counts."""
    if name == "bce":
        return MultiLabelLoss()
    if name == "weighted_bce":
        return MultiLabelLoss(pos_weight=inverse_frequency_pos_weight(pos_counts, total))
    if name == "focal":
        return MultiLabelLoss(gamma=gamma, alpha=alpha)
    if name == "cb_focal":
        return MultiLabelLoss(gamma=gamma, pos_weight=class_balanced_pos_weight(pos_counts, total, beta))
    raise ValueError(f"Unknown loss {name!r}")
