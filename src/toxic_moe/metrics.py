"""Evaluation metrics for multi-label toxicity classification (numpy / scikit-learn only).

Reported metrics
----------------
* ``roc_auc``  - per-label ROC-AUC and their mean (the official Kaggle metric).
* ``pr_auc``   - per-label average precision. More informative than ROC-AUC for
  labels with <1% positives, because it ignores the huge pool of easy negatives.
* ``f1@0.5``   - per-label / macro / micro F1 with a fixed 0.5 threshold.
* ``f1@tuned`` - same, with one threshold per label chosen on the *validation*
  split and then applied unchanged to the test split (no test-set peeking).

Element-wise "accuracy" is deliberately not reported: ~96% of all label entries
are 0, so predicting "clean" everywhere already scores ~0.96.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score

from . import LABELS


def _has_both_classes(y: np.ndarray) -> bool:
    return 0 < y.sum() < len(y)


def per_label_roc_auc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array(
        [roc_auc_score(y[:, j], p[:, j]) if _has_both_classes(y[:, j]) else np.nan for j in range(y.shape[1])]
    )


def per_label_pr_auc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array(
        [average_precision_score(y[:, j], p[:, j]) if y[:, j].sum() > 0 else np.nan for j in range(y.shape[1])]
    )


def tune_thresholds(y: np.ndarray, p: np.ndarray, default: float = 0.5) -> np.ndarray:
    """Per-label threshold maximising F1 on (y, p). Exact search over all cut points."""
    thresholds = np.full(y.shape[1], default, dtype=float)
    for j in range(y.shape[1]):
        if y[:, j].sum() == 0:
            continue
        precision, recall, cut = precision_recall_curve(y[:, j], p[:, j])
        # the last precision/recall pair has no threshold attached
        precision, recall = precision[:-1], recall[:-1]
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
        thresholds[j] = float(cut[int(np.argmax(f1))])
    return thresholds


def binarize(p: np.ndarray, thresholds: Sequence[float] | float) -> np.ndarray:
    return (p >= np.asarray(thresholds, dtype=float)).astype(np.int8)


def f1_report(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    return {
        "per_label": f1_score(y, pred, average=None, zero_division=0).tolist(),
        "macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro": float(f1_score(y, pred, average="micro", zero_division=0)),
    }


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
    labels: List[str] = LABELS,
) -> Dict[str, Any]:
    """Compute every metric above. ``thresholds`` should come from the validation split."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    roc = per_label_roc_auc(y, p)
    pr = per_label_pr_auc(y, p)
    out: Dict[str, Any] = {
        "n": int(len(y)),
        "labels": list(labels),
        "positives": y.sum(axis=0).astype(int).tolist(),
        "roc_auc": roc.tolist(),
        "roc_auc_mean": _nanmean(roc),
        "pr_auc": pr.tolist(),
        "pr_auc_mean": _nanmean(pr),
        "f1@0.5": f1_report(y, binarize(p, 0.5)),
    }
    if thresholds is not None:
        out["thresholds"] = [float(t) for t in thresholds]
        out["f1@tuned"] = f1_report(y, binarize(p, thresholds))
    return out


def pr_curves(y: np.ndarray, p: np.ndarray, points: int = 101) -> Dict[str, Dict[str, List[float]]]:
    """Down-sampled precision-recall curves (recall grid -> best precision), for plotting."""
    grid = np.linspace(0, 1, points)
    curves = {}
    for j, name in enumerate(LABELS[: y.shape[1]]):
        if y[:, j].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y[:, j], p[:, j])
        # interpolated precision: best precision achievable at recall >= r
        order = np.argsort(recall)
        r, pr = recall[order], precision[order]
        best = np.maximum.accumulate(pr[::-1])[::-1]
        interp = np.interp(grid, r, best)
        curves[name] = {"recall": grid.round(4).tolist(), "precision": interp.round(4).tolist()}
    return curves


def mean_pairwise_tv(rows: np.ndarray) -> float:
    """Mean total-variation distance over all pairs of rows, each row a distribution over experts.

    0 = every row puts its weight on the same experts; 1 = disjoint experts.
    """
    rows = np.asarray(rows, dtype=float)
    if len(rows) < 2:
        return 0.0
    d = 0.5 * np.abs(rows[:, None, :] - rows[None, :, :]).sum(-1)
    return float(d[np.triu_indices(len(rows), 1)].mean())


def gate_statistics(y: np.ndarray, gates: np.ndarray, labels: List[str] = LABELS) -> Dict[str, Any]:
    """How the MoE router distributes comments over experts.

    Returns the mean gate weight per expert for all comments, for clean comments,
    and for the comments carrying each label (rows), how often each expert is the
    top-1 choice, and ``label_divergence``: the mean pairwise total-variation distance
    between the labels' rows (0 = every label's comments go to the same experts).
    """
    y = np.asarray(y).astype(bool)
    gates = np.asarray(gates, dtype=float)
    rows = {"all": gates.mean(axis=0)}
    clean = ~y.any(axis=1)
    if clean.any():
        rows["clean"] = gates[clean].mean(axis=0)
    for j, name in enumerate(labels):
        if y[:, j].any():
            rows[name] = gates[y[:, j]].mean(axis=0)
    top1 = np.bincount(gates.argmax(axis=1), minlength=gates.shape[1]) / len(gates)
    return {
        "mean_gate": {k: v.round(5).tolist() for k, v in rows.items()},
        "top1_share": top1.round(5).tolist(),
        "label_divergence": round(mean_pairwise_tv(np.array([rows[k] for k in labels if k in rows])), 5),
    }


def label_gate_statistics(y: np.ndarray, gates: np.ndarray, labels: List[str] = LABELS) -> Dict[str, Any]:
    """Routing of the multi-gate (MMoE) head, where every label has its own gate.

    ``gates`` has shape (comments, labels, experts).

    * ``label_gates[l]``: label ``l``'s mean gate over the comments that carry ``l``.
    * ``label_gates_all[l]``: label ``l``'s mean gate over all comments.
    * ``label_divergence``: mean pairwise total-variation distance between the
      ``label_gates`` rows, on the same scale as :func:`gate_statistics`.
    * ``within_comment_divergence``: the same distance between the six labels' gates *for
      the same comment*, averaged over comments. A per-comment router scores 0 by
      construction; this is what a multi-gate head can do that a per-comment router cannot.
    """
    y = np.asarray(y).astype(bool)
    gates = np.asarray(gates, dtype=float)
    own = {name: gates[y[:, j], j].mean(axis=0) for j, name in enumerate(labels) if y[:, j].any()}
    overall = {name: gates[:, j].mean(axis=0) for j, name in enumerate(labels)}
    n_labels = gates.shape[1]
    within = [  # one label pair at a time keeps memory at O(comments x experts)
        0.5 * np.abs(gates[:, a] - gates[:, b]).sum(-1).mean() for a in range(n_labels) for b in range(a + 1, n_labels)
    ]
    return {
        "label_gates": {k: v.round(5).tolist() for k, v in own.items()},
        "label_gates_all": {k: v.round(5).tolist() for k, v in overall.items()},
        "label_divergence": round(mean_pairwise_tv(np.array(list(own.values()))), 5),
        "within_comment_divergence": round(float(np.mean(within)) if within else 0.0, 5),
    }


def routing_statistics(y: np.ndarray, gates: np.ndarray, labels: List[str] = LABELS) -> Dict[str, Any]:
    """:func:`gate_statistics` for (n, experts) gates, :func:`label_gate_statistics` for (n, labels, experts)."""
    gates = np.asarray(gates)
    return gate_statistics(y, gates, labels) if gates.ndim == 2 else label_gate_statistics(y, gates, labels)
