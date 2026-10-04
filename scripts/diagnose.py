#!/usr/bin/env python
"""Why didn't the MoE head beat the linear head? Diagnostics on the saved runs.

    # label-free parts: routing, threshold transfer, agreement between heads
    python scripts/diagnose.py --results results

    # everything, including the parts that need the test labels (CPU only, a few minutes)
    python scripts/diagnose.py --results /content/drive/MyDrive/toxic-moe/results \\
        --data /content/drive/MyDrive/toxic-moe/data

Reads ``<run>/metrics.json`` and, where present, ``<run>/predictions.npz``. Predictions are
not in git, so run this where the runs were trained (on Colab, with the results on Drive).
Needs numpy, pandas and scikit-learn only. Writes ``<results>/diagnosis.md`` and
``<results>/diagnosis.json``; ``docs/diagnosis.md`` discusses the findings.

The comparisons isolate the head: ``bert_bce`` (linear head) vs ``moe_bce`` (MoE head),
same loss, same seeds, same data. Follow-up heads (configs/heads.yaml) are included when
their runs exist.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toxic_moe import LABELS  # noqa: E402
from toxic_moe.metrics import mean_pairwise_tv  # noqa: E402

BASELINE, MOE = "bert_bce", "moe_bce"
FOLLOW_UP = ["mlp_bce", "moe_bce_noaux", "mmoe_bce", "labelattn_bce"]
NAMES = {
    "bert_bce": "Linear",
    "moe_bce": "MoE",
    "moe_cbfocal": "MoE + CB-focal",
    "mlp_bce": "MLP",
    "moe_bce_noaux": "MoE, no balancing",
    "mmoe_bce": "Multi-gate MoE",
    "labelattn_bce": "Label attention",
}
RARE = [LABELS.index(k) for k in ("severe_toxic", "threat", "identity_hate")]
SEED_SUFFIX = re.compile(r"_s\d+$")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
@dataclass
class Run:
    name: str
    dir: Path
    metrics: dict
    _preds: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)

    @property
    def group(self) -> str:
        return self.metrics.get("group") or SEED_SUFFIX.sub("", self.name)

    @property
    def seed(self) -> Optional[int]:
        return self.metrics.get("seed")

    @property
    def has_predictions(self) -> bool:
        return (self.dir / "predictions.npz").exists()

    def preds(self) -> Dict[str, np.ndarray]:
        """test_probs (n, 6), test_ids, optional test_gates; probabilities as float32."""
        if self._preds is None:
            with np.load(self.dir / "predictions.npz", allow_pickle=False) as z:
                self._preds = {k: z[k].astype(np.float32) if z[k].dtype.kind == "f" else z[k] for k in z.files}
        return self._preds


def load_runs(results: Path) -> Dict[str, List[Run]]:
    groups: Dict[str, List[Run]] = {}
    for path in sorted(results.glob("*/metrics.json")):
        run = Run(path.parent.name, path.parent, json.loads(path.read_text()))
        groups.setdefault(run.group, []).append(run)
    for runs in groups.values():
        runs.sort(key=lambda r: (r.seed is None, r.seed or 0))
    return groups


def run_config(run: Run) -> Dict[str, Any]:
    path = run.dir / "config.yaml"
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def with_predictions(runs: Sequence[Run]) -> List[Run]:
    return [r for r in runs if r.has_predictions]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)  # predictions are stored as float16, which rounds tiny values to 0
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def table(header: Sequence[str], rows: Sequence[Sequence[Any]], align: Optional[str] = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if a == "r" else "---" for a in align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pm(values: Sequence[float], digits: int = 3, signed: bool = False) -> str:
    v = np.asarray(values, dtype=float)
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    if len(v) == 0 or np.all(np.isnan(v)):
        return "–"
    if len(v) == 1:
        return fmt.format(v[0])
    return f"{fmt.format(np.nanmean(v))} ± {np.nanstd(v, ddof=1):.{digits}f}"


def name(group: str) -> str:
    return NAMES.get(group, group)


# ---------------------------------------------------------------------------
# 1. routing (metrics.json only)
# ---------------------------------------------------------------------------
def routing_specificity(groups: Dict[str, List[Run]]) -> Tuple[str, dict]:
    """Do the six labels use different experts, or one 'toxic' pair of experts?"""
    rows, out, n_exp = [], {}, 6
    for group, runs in sorted(groups.items()):
        stats = []
        for r in runs:
            if r.metrics.get("arch") != "moe" or r.metrics.get("encoder", "bert-base-uncased") != "bert-base-uncased":
                continue
            g = (r.metrics["test"].get("gates") or {}).get("mean_gate", {})
            if not all(k in g for k in ["clean", *LABELS]):
                continue
            n_exp = len(g["clean"])
            label_rows = np.array([g[k] for k in LABELS])
            top2 = np.argsort(g["toxic"])[-2:]  # the two experts toxic comments use most
            aux = r.metrics["history"][-1].get("train_aux_loss")
            stats.append(
                {
                    "label_divergence": mean_pairwise_tv(label_rows),
                    "toxic_vs_clean": 0.5 * float(np.abs(np.subtract(g["toxic"], g["clean"])).sum()),
                    "min_label_share_top2": float(label_rows[:, top2].sum(1).min()),
                    "clean_share_top2": float(np.sum(np.asarray(g["clean"])[top2])),
                    "aux_loss": aux if aux is not None else float("nan"),
                }
            )
        if not stats:
            continue
        out[group] = stats
        rows.append(
            [f"`{group}`", len(stats)]
            + [pm([s[k] for s in stats], 2) for k in ("label_divergence", "toxic_vs_clean", "min_label_share_top2")]
            + [pm([s["clean_share_top2"] for s in stats], 2), pm([s["aux_loss"] for s in stats], 3)]
        )
    if not rows:
        return "", {}
    md = "\n".join(
        [
            "## Do the labels get their own experts?",
            "",
            "Distances are total variation between mean router weights (0 = same experts, 1 = disjoint experts). "
            "*Label divergence*: mean distance between the six labels' positive comments. *Toxic vs clean*: "
            "distance between toxic and clean comments. *Top-2 share*: router weight that each label's positives "
            f"(minimum over labels) and clean comments put on the two experts toxic comments use most "
            f"(uniform routing: {2 / n_exp:.0%}).",
            "",
            table(
                [
                    "Setting",
                    "Runs",
                    "Label divergence",
                    "Toxic vs clean",
                    "Top-2 share, labels (min)",
                    "Top-2 share, clean",
                    "Aux loss (1.0 = balanced)",
                ],
                rows,
            ),  # fmt: skip
        ]
    )
    return md, out


def misrouting(groups: Dict[str, List[Run]]) -> Tuple[str, dict]:
    """Does the router send comments the linear head is sure about away from the 'toxic' experts?"""
    base = {r.seed: r for r in with_predictions(groups.get(BASELINE, []))}
    out, rows = {}, []
    for r in with_predictions(groups.get(MOE, [])):
        if "test_gates" not in r.preds() or r.seed not in base:
            continue
        toxic_experts = np.argsort(r.metrics["test"]["gates"]["mean_gate"]["toxic"])[-2:]
        w = r.preds()["test_gates"][:, toxic_experts].sum(1)
        sure = base[r.seed].preds()["test_probs"][:, 0] >= 0.9
        rec = {
            "share_mostly_toxic_experts": float((w > 0.5).mean()),
            "linear_sure_toxic": int(sure.sum()),
            "of_which_routed_away": int((sure & (w <= 0.5)).sum()),
        }
        out[r.name] = rec
        rows.append(
            [f"`{r.name}`", f"{rec['share_mostly_toxic_experts']:.1%}", f"{rec['linear_sure_toxic']:,}",
             f"{rec['of_which_routed_away']:,}"]
        )  # fmt: skip
    if not rows:
        return "", {}
    md = "\n".join(
        [
            "### Is anything routed to the wrong experts?",
            "",
            "Test comments that put more than half of their router weight on the two 'toxic' experts, and the "
            "comments the linear head (same seed) scores ≥ 0.9 for `toxic` that the router sends elsewhere.",
            "",
            table(["Run", "Mostly 'toxic' experts", "Linear head: p(toxic) ≥ 0.9", "…routed elsewhere"], rows),
        ]
    )
    return md, out


# ---------------------------------------------------------------------------
# 2. threshold transfer (metrics.json only)
# ---------------------------------------------------------------------------
def oracle_f1(curve: dict) -> float:
    """Best F1 on the saved (interpolated) test precision-recall curve: thresholds tuned on test itself."""
    r, p = np.asarray(curve["recall"]), np.asarray(curve["precision"])
    return float(np.max(np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)))


def threshold_transfer(groups: Dict[str, List[Run]]) -> Tuple[str, dict]:
    """How much F1 is lost because thresholds come from validation, and how noisy that makes F1 comparisons."""
    runs = [r for rs in groups.values() for r in rs if "pr_curves" in r.metrics["test"]]
    if not runs or BASELINE not in groups:
        return "", {}

    def f1(r: Run, kind: str, labels: Sequence[int]) -> float:
        if kind == "tuned":
            return float(np.mean([r.metrics["test"]["f1@tuned"]["per_label"][j] for j in labels]))
        return float(np.mean([oracle_f1(r.metrics["test"]["pr_curves"][LABELS[j]]) for j in labels]))

    lost = {
        lab: float(np.mean([f1(r, "oracle", [j]) - f1(r, "tuned", [j]) for r in runs])) for j, lab in enumerate(LABELS)
    }
    base = {r.seed: r for r in groups[BASELINE]}
    rows: List[List[Any]] = []
    paired: Dict[str, Dict[str, List[float]]] = {}
    for group, rs in sorted(groups.items(), key=lambda kv: (kv[0] != BASELINE, kv[0])):
        pairs = [(r, base[r.seed]) for r in rs if r.seed in base and "pr_curves" in r.metrics["test"]]
        if group == BASELINE or len(pairs) < 2:
            continue
        cells: List[str] = []
        rec: Dict[str, List[float]] = {}
        for kind in ("tuned", "oracle"):
            for what, labels in (("macro", range(6)), ("rare", RARE)):
                d = [f1(a, kind, labels) - f1(b, kind, labels) for a, b in pairs]
                rec[f"{kind}_{what}"] = d
                cells.append(pm(d, 4, signed=True))
        paired[group] = rec
        rows.append([f"`{group}`", len(pairs)] + cells)
    out = {"f1_lost_per_label": lost, "paired": paired}
    md = [
        "## How much of the F1 differences is threshold noise?",
        "",
        "F1 thresholds are tuned on the validation split and applied to test. *Oracle* thresholds are tuned on "
        "test itself, from the saved precision-recall curves: not a fair score, but it removes the noise of "
        "transferring thresholds. F1 lost by the transfer, mean over all runs: "
        + ", ".join(f"`{k}` {v:.3f}" for k, v in lost.items())
        + ".",
    ]
    if rows:
        md += [
            "",
            "Paired difference from the baseline (mean ± std over seeds). If the std shrinks a lot with oracle "
            "thresholds, most of the seed-to-seed spread in F1 comes from the thresholds, not the models.",
            "",
            table(
                [
                    "Setting",
                    "Seeds",
                    "Δ macro-F1, val thr.",
                    "Δ rare-F1, val thr.",
                    "Δ macro-F1, oracle thr.",
                    "Δ rare-F1, oracle thr.",
                ],
                rows,
            ),  # fmt: skip
        ]
    return "\n".join(md), out


def val_test_gap(groups: Dict[str, List[Run]]) -> Tuple[str, dict]:
    """Validation ROC-AUC minus test ROC-AUC: a ceiling shared by every model?"""
    runs = [
        r
        for rs in groups.values()
        for r in rs
        if "val" in r.metrics and r.metrics.get("encoder", "bert-base-uncased") == "bert-base-uncased"
    ]
    if not runs:
        return "", {}
    gaps = np.array([np.subtract(r.metrics["val"]["roc_auc"], r.metrics["test"]["roc_auc"]) for r in runs])
    per_run = gaps.mean(1)
    per_label: Dict[str, float] = dict(zip(LABELS, gaps.mean(0).round(4).tolist()))
    std = float(per_run.std(ddof=1)) if len(runs) > 1 else 0.0
    out = {"runs": len(runs), "mean_gap": float(per_run.mean()), "std_over_runs": std, "per_label": per_label}
    md = "\n".join(
        [
            "## Validation vs test",
            "",
            f"Over all {len(runs)} BERT-base runs, mean ROC-AUC is {per_run.mean():.4f} lower on the test set than on "
            f"the validation split (std over runs {std:.4f}). Per label: "
            + ", ".join(f"`{k}` {v:+.4f}" for k, v in per_label.items())
            + ". A gap that is the same for every head and loss is not something a different head can close.",
        ]
    )
    return md, out


# ---------------------------------------------------------------------------
# 3. agreement between heads (predictions)
# ---------------------------------------------------------------------------
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).correlation)


def agreement_pair(a: np.ndarray, b: np.ndarray, k: int) -> Tuple[float, float, float]:
    """(top-k overlap, Spearman on the union of both top-k sets, Spearman on the bottom 90% of both rankings)."""
    ta, tb = np.argpartition(-a, k)[:k], np.argpartition(-b, k)[:k]
    union = np.union1d(ta, tb)
    ra, rb = a.argsort().argsort() / len(a), b.argsort().argsort() / len(b)
    low = (ra < 0.9) & (rb < 0.9)
    return len(np.intersect1d(ta, tb)) / k, spearman(a[union], b[union]), spearman(a[low], b[low])


def agreement(groups: Dict[str, List[Run]]) -> Tuple[str, dict]:
    """Do two heads rank comments differently where the metrics are decided (the top of each ranking)?"""
    if BASELINE not in groups:
        return "", {}
    base = with_predictions(groups[BASELINE])
    if len(base) < 2:
        return "", {}
    positives = np.asarray(base[0].metrics["test"]["positives"])
    comparisons: List[Tuple[str, List[Tuple[Run, Run]]]] = [
        (f"{name(BASELINE)} vs {name(BASELINE)}", list(itertools.combinations(base, 2)))
    ]
    for group in [MOE, *FOLLOW_UP]:
        other = with_predictions(groups.get(group, []))
        if len(other) >= 2 and group == MOE:
            comparisons.append((f"{name(MOE)} vs {name(MOE)}", list(itertools.combinations(other, 2))))
        cross = [(a, b) for a in base for b in other if a.seed != b.seed]  # different seeds: like the first row
        if cross:
            comparisons.append((f"{name(BASELINE)} vs {name(group)}", cross))
    out: Dict[str, Any] = {}
    for title, pairs in comparisons:
        per_label = []
        n = len(pairs[0][0].preds()["test_probs"])
        for j in range(len(LABELS)):
            k = int(min(2 * positives[j], n - 1))
            vals = [agreement_pair(a.preds()["test_probs"][:, j], b.preds()["test_probs"][:, j], k) for a, b in pairs]
            per_label.append(np.mean(vals, axis=0))
        out[title] = {lab: dict(zip(("top_overlap", "top_spearman", "bottom_spearman"), map(float, v)))
                      for lab, v in zip(LABELS, per_label)}  # fmt: skip
    titles = list(out)
    rows_top = [[f"`{lab}`"] + [f"{out[t][lab]['top_overlap']:.3f}" for t in titles] for lab in LABELS]
    rows_bottom = [[f"`{lab}`"] + [f"{out[t][lab]['bottom_spearman']:.3f}" for t in titles] for lab in LABELS]
    md = "\n".join(
        [
            "## Do the heads rank comments differently?",
            "",
            "For each label, the *K* highest-scored test comments (K = 2 × the label's test positives, the region "
            "that decides PR-AUC and F1): the fraction both models put in their top K. Pairs always use different "
            "seeds, so the first column is the seed-to-seed baseline: a head that agrees with the linear head as "
            "much as another linear seed does is computing the same function where it matters.",
            "",
            table(["Label", *titles], rows_top),
            "",
            "The same pairs, Spearman correlation among the **bottom 90%** of both rankings (mostly clean comments, "
            "which hardly affect any metric):",
            "",
            table(["Label", *titles], rows_bottom),
        ]
    )
    return md, out


def expert_offsets(groups: Dict[str, List[Run]]) -> Tuple[str, dict]:
    """Among clean-looking comments, how much of the rare-label score depends on which expert was chosen?"""
    moe = [r for r in with_predictions(groups.get(MOE, [])) if "test_gates" in r.preds()]
    base = {r.seed: r for r in with_predictions(groups.get(BASELINE, []))}
    if not moe:
        return "", {}

    def explained(x: np.ndarray, group_ids: np.ndarray) -> float:
        between = sum(
            (group_ids == g).mean() * (x[group_ids == g].mean() - x.mean()) ** 2 for g in np.unique(group_ids)
        )
        return float(between / x.var()) if x.var() > 0 else float("nan")

    out: Dict[str, Dict[str, List[float]]] = {"moe": {}, "linear": {}}
    for r in moe:
        p, gates = r.preds()["test_probs"], r.preds()["test_gates"]
        clean = p[:, 0] < 0.01
        if clean.sum() < 10:
            continue
        top1 = gates.argmax(1)[clean]
        for j in RARE:
            out["moe"].setdefault(LABELS[j], []).append(explained(logit(p[clean, j]), top1))
            if r.seed in base:
                q = base[r.seed].preds()["test_probs"]
                out["linear"].setdefault(LABELS[j], []).append(explained(logit(q[clean, j]), top1))
    rows = [[f"`{lab}`", pm(out["moe"][lab], 2), pm(out["linear"].get(lab, []), 2)] for lab in out["moe"]]
    if not rows:
        return "", {}
    md = "\n".join(
        [
            "### Where the MoE's different ranking of clean comments comes from",
            "",
            "Test comments the MoE scores below 0.01 for `toxic`, grouped by their top-1 expert: the share of the "
            "variance of each rare-label logit explained by the expert. The linear head's logits for the same "
            "comments, grouped by the same experts, are the reference.",
            "",
            table(["Label", "MoE head", "Linear head (same grouping)"], rows),
        ]
    )
    return md, out


# ---------------------------------------------------------------------------
# 4. analyses that need the test labels
# ---------------------------------------------------------------------------
@dataclass
class Labels:
    y_test: np.ndarray  # aligned with the predictions' test_ids
    test_text: List[str]
    y_train: np.ndarray


def load_labels(data_dir: Path, ids: np.ndarray) -> Labels:
    from toxic_moe.data import ensure_extracted, load_test, load_train

    data_dir = ensure_extracted(data_dir)
    test = load_test(data_dir).set_index("id")
    missing = set(ids) - set(test.index)
    if missing:
        raise SystemExit(f"{len(missing)} prediction ids are not in the labelled test set; wrong --data folder?")
    test = test.loc[list(ids)]
    return Labels(
        test[LABELS].to_numpy().astype(int), test["comment_text"].tolist(), load_train(data_dir)[LABELS].to_numpy()
    )


def roc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def ap(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def label_structure(lab: Labels) -> Tuple[str, dict]:
    y = lab.y_train
    toxic = y[:, 0] == 1
    out = {
        k: {"positives": int(y[:, j].sum()), "p_toxic_given_label": float(toxic[y[:, j] == 1].mean())}
        for j, k in enumerate(LABELS)
        if j > 0
    }
    rows = [[f"`{k}`", f"{v['positives']:,}", f"{v['p_toxic_given_label']:.1%}"] for k, v in out.items()]
    md = "\n".join(
        [
            "## Labels",
            "",
            "In `train.csv`, how often each label comes with `toxic`. A router that picks one mixture of experts per "
            "comment cannot separate labels that almost always occur together.",
            "",
            table(["Label", "Positives", "Also `toxic`"], rows),
        ]
    )
    return md, out


def ensembles(groups: Dict[str, List[Run]], lab: Labels) -> Tuple[str, dict]:
    """Seed ensembles per setting, and whether an MoE adds more diversity to an ensemble than another linear seed."""
    y = lab.y_test
    out: Dict[str, Any] = {"seed_ensembles": {}, "pairs": {}}
    rows = []
    for group, runs in sorted(groups.items()):
        runs = with_predictions(runs)
        if len(runs) < 2:
            continue
        probs = [r.preds()["test_probs"] for r in runs]
        single = [[roc(y[:, j], p[:, j]) for j in range(6)] for p in probs]
        single_ap = [[ap(y[:, j], p[:, j]) for j in range(6)] for p in probs]
        mean_p = np.mean(probs, axis=0)
        ens = [roc(y[:, j], mean_p[:, j]) for j in range(6)]
        ens_ap = [ap(y[:, j], mean_p[:, j]) for j in range(6)]
        rec = {
            "n": len(runs),
            "single_roc": float(np.mean(np.nanmean(single, axis=1))),
            "ensemble_roc": float(np.nanmean(ens)),
            "single_pr": float(np.mean(np.nanmean(single_ap, axis=1))),
            "ensemble_pr": float(np.nanmean(ens_ap)),
        }
        out["seed_ensembles"][group] = rec
        rows.append(
            [f"`{group}`", rec["n"], f"{rec['single_roc']:.4f}", f"{rec['ensemble_roc']:.4f}",
             f"{rec['ensemble_roc'] - rec['single_roc']:+.4f}", f"{rec['single_pr']:.3f}", f"{rec['ensemble_pr']:.3f}",
             f"{rec['ensemble_pr'] - rec['single_pr']:+.3f}"]
        )  # fmt: skip
    base = with_predictions(groups.get(BASELINE, []))
    pair_rows = []
    for group in [BASELINE, MOE, *FOLLOW_UP]:
        other = with_predictions(groups.get(group, []))
        pairs = [(a, b) for a in base for b in other if a.seed != b.seed]
        if group == BASELINE:
            pairs = list(itertools.combinations(base, 2))
        if not pairs:
            continue
        scores = []
        for a, b in pairs:
            p = (a.preds()["test_probs"] + b.preds()["test_probs"]) / 2
            scores.append(
                [
                    np.nanmean([roc(y[:, j], p[:, j]) for j in range(6)]),
                    np.nanmean([ap(y[:, j], p[:, j]) for j in range(6)]),
                ]
            )
        s = np.mean(scores, axis=0)
        out["pairs"][group] = {"roc": float(s[0]), "pr": float(s[1]), "n_pairs": len(pairs)}
        pair_rows.append([f"{name(BASELINE)} + {name(group)}", len(pairs), f"{s[0]:.4f}", f"{s[1]:.3f}"])
    md = [
        "## Ensembles",
        "",
        "Averaging the probabilities of a setting's seeds (test ROC-AUC / PR-AUC, mean over labels):",
        "",
        table(["Setting", "Seeds", "Single ROC", "Ensemble ROC", "Gain", "Single PR", "Ensemble PR", "Gain"], rows),
    ]
    if len(pair_rows) > 1:
        md += [
            "",
            "Two-model ensembles: a linear head plus a second model trained with a different seed. If a head learned "
            "something the linear head did not, pairing it with a linear head should beat pairing two linear heads.",
            "",
            table(["Pair", "Pairs", "ROC-AUC", "PR-AUC"], pair_rows),
        ]
    return "\n".join(md), out


def subtype_auc(groups: Dict[str, List[Run]], lab: Labels) -> Tuple[str, dict]:
    """Among comments that are toxic, can the model tell which kind of toxicity it is?"""
    y = lab.y_test
    toxic = y[:, 0] == 1
    out, rows = {}, []
    for group in [BASELINE, MOE, "moe_cbfocal", *FOLLOW_UP]:
        runs = with_predictions(groups.get(group, []))
        if not runs:
            continue
        vals = [[roc(y[toxic, j], r.preds()["test_probs"][toxic, j]) for j in range(1, 6)] for r in runs]
        out[group] = dict(zip(LABELS[1:], np.mean(vals, axis=0).tolist()))
        rows.append([name(group)] + [pm([v[i] for v in vals], 4) for i in range(5)])
    md = "\n".join(
        [
            "## Telling kinds of toxicity apart",
            "",
            f"ROC-AUC for each sub-label, computed only over the {int(toxic.sum()):,} test comments labelled `toxic` "
            "(mean ± std over seeds). This is the job label-specialised experts were supposed to do.",
            "",
            table(["Head", *[f"`{k}`" for k in LABELS[1:]]], rows),
        ]
    )
    return md, out


def token_lengths(texts: List[str], encoder: str) -> Tuple[np.ndarray, str]:
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(encoder)
        lengths = [len(x) for x in tok(texts, add_special_tokens=True, truncation=False)["input_ids"]]
        return np.asarray(lengths), f"{encoder} tokens"
    except Exception:  # no transformers / no network: words are a rough proxy (~1.3 tokens per word)
        return np.asarray([int(len(t.split()) * 1.3) + 2 for t in texts]), "estimated tokens (1.3 × words)"


def length_effect(groups: Dict[str, List[Run]], lab: Labels) -> Tuple[str, dict]:
    """How much worse are predictions on comments longer than the 128-token input window?"""
    base = with_predictions(groups.get(BASELINE, []))
    if not base:
        return "", {}
    cfg_len = run_config(base[0]).get("max_length", 128)
    lengths, unit = token_lengths(lab.test_text, base[0].metrics.get("encoder", "bert-base-uncased"))
    long = lengths > cfg_len
    y = lab.y_test
    out: Dict[str, Any] = {"share_long": float(long.mean()), "unit": unit, "auc": {}}
    rows = []
    for group in [BASELINE, MOE, *FOLLOW_UP]:
        runs = with_predictions(groups.get(group, []))
        if not runs:
            continue
        p = np.mean([r.preds()["test_probs"] for r in runs], axis=0)
        for j, k in enumerate(LABELS):
            short_auc, long_auc = roc(y[~long, j], p[~long, j]), roc(y[long, j], p[long, j])
            out["auc"].setdefault(group, {})[k] = {"short": short_auc, "long": long_auc}
        rows.append(
            [name(group)]
            + [f"{out['auc'][group][k]['short']:.4f} / {out['auc'][group][k]['long']:.4f}" for k in LABELS]
        )
    pos_long = [f"`{k}` {long[y[:, j] == 1].mean():.1%}" for j, k in enumerate(LABELS)]
    md = "\n".join(
        [
            "## Long comments",
            "",
            f"{long.mean():.1%} of test comments are longer than the {cfg_len}-token input window ({unit}); the "
            f"model never sees anything after the first {cfg_len} tokens. Share of each label's positives that are "
            "long: " + ", ".join(pos_long) + ". ROC-AUC on short / long comments (seed-averaged predictions):",
            "",
            table(["Head", *[f"`{k}`" for k in LABELS]], rows),
        ]
    )
    return md, out


def routing_recall(groups: Dict[str, List[Run]], lab: Labels) -> Tuple[str, dict]:
    """Are positives that the router sends away from the 'toxic' experts missed?"""
    y = lab.y_test
    out, rows = {}, []
    for group in [MOE, "moe_cbfocal"]:
        for r in with_predictions(groups.get(group, [])):
            pr = r.preds()
            if "test_gates" not in pr or pr["test_gates"].ndim != 2:
                continue
            mean_gate = r.metrics["test"]["gates"]["mean_gate"]["toxic"]
            toxic_experts = np.argsort(mean_gate)[-2:]
            w = pr["test_gates"][:, toxic_experts].sum(1)
            thr = np.asarray(r.metrics["test"]["thresholds"])
            hit = pr["test_probs"] >= thr
            rec = {}
            for j, k in enumerate(LABELS):
                pos = y[:, j] == 1
                away = pos & (w < 0.5)
                rec[k] = {
                    "share_routed_away": float(away.sum() / max(pos.sum(), 1)),
                    "recall_routed_to_toxic_experts": float(hit[pos & (w >= 0.5), j].mean()) if (pos & (w >= 0.5)).any() else float("nan"),
                    "recall_routed_away": float(hit[away, j].mean()) if away.any() else float("nan"),
                }  # fmt: skip
            out[r.name] = rec
            rows.append(
                [f"`{r.name}`"]
                + [f"{rec[k]['share_routed_away']:.1%} ({_fmt(rec[k]['recall_routed_away'])} vs "
                   f"{_fmt(rec[k]['recall_routed_to_toxic_experts'])})" for k in LABELS]
            )  # fmt: skip
    if not rows:
        return "", {}
    md = "\n".join(
        [
            "## Routing mistakes",
            "",
            "Share of each label's positives that the router sends mostly (> 50% of the weight) away from the two "
            "experts toxic comments use most, and the recall at the tuned threshold for those positives vs the rest.",
            "",
            table(["Run", *[f"`{k}`" for k in LABELS]], rows),
        ]
    )
    return md, out


def _fmt(x: float) -> str:
    return "–" if np.isnan(x) else f"{x:.2f}"


def bootstrap(groups: Dict[str, List[Run]], lab: Labels, rounds: int, seed: int = 0) -> Tuple[str, dict]:
    """How large is the test-set sampling noise compared with the MoE-vs-linear difference?"""
    base, moe = with_predictions(groups.get(BASELINE, [])), with_predictions(groups.get(MOE, []))
    if not base or not moe or rounds <= 0:
        return "", {}
    y = lab.y_test
    pa = np.mean([r.preds()["test_probs"] for r in base], axis=0)
    pb = np.mean([r.preds()["test_probs"] for r in moe], axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(rounds, len(y)))
    out, rows = {}, []
    for j, k in enumerate(LABELS):
        d_roc = [roc(y[i, j], pb[i, j]) - roc(y[i, j], pa[i, j]) for i in idx]
        d_ap = [ap(y[i, j], pb[i, j]) - ap(y[i, j], pa[i, j]) for i in idx]
        point = (roc(y[:, j], pb[:, j]) - roc(y[:, j], pa[:, j]), ap(y[:, j], pb[:, j]) - ap(y[:, j], pa[:, j]))
        ci = (np.nanpercentile(d_roc, [2.5, 97.5]), np.nanpercentile(d_ap, [2.5, 97.5]))
        out[k] = {"d_roc": point[0], "d_roc_ci": ci[0].tolist(), "d_pr": point[1], "d_pr_ci": ci[1].tolist()}
        rows.append(
            [f"`{k}`", f"{point[0]:+.4f}", f"[{ci[0][0]:+.4f}, {ci[0][1]:+.4f}]", f"{point[1]:+.3f}",
             f"[{ci[1][0]:+.3f}, {ci[1][1]:+.3f}]"]
        )  # fmt: skip
    md = "\n".join(
        [
            "## Test-set sampling noise",
            "",
            f"MoE minus linear head, both averaged over their seeds, with 95% intervals from {rounds} paired "
            "bootstrap resamples of the test comments. An interval that contains 0 means the 64k test comments "
            "cannot tell the two heads apart.",
            "",
            table(["Label", "Δ ROC-AUC", "95% interval", "Δ PR-AUC", "95% interval"], rows),
        ]
    )
    return md, out


def disagreements(groups: Dict[str, List[Run]], lab: Labels, show: int) -> Tuple[str, dict]:
    """Comments where every model confidently disagrees with the label: a rough count of label noise."""
    runs = [
        r
        for g in groups.values()
        for r in with_predictions(g)
        if r.metrics.get("encoder", "bert-base-uncased") == "bert-base-uncased" and not r.metrics.get("freeze_encoder")
    ]
    if not runs:
        return "", {}
    p = np.mean([r.preds()["test_probs"][:, 0] for r in runs], axis=0)
    y = lab.y_test[:, 0]
    fp, fn = (p >= 0.9) & (y == 0), (p <= 0.05) & (y == 1)
    out = {"runs": len(runs), "confident_false_positives": int(fp.sum()), "confident_false_negatives": int(fn.sum()),
           "toxic_positives": int(y.sum())}  # fmt: skip
    if show:
        for title, mask in (("labelled clean, all models say toxic", fp), ("labelled toxic, all models say clean", fn)):
            print(f"\n--- {title} (first {show}) ---")
            for i in np.flatnonzero(mask)[:show]:
                print(f"[p={p[i]:.3f}] {lab.test_text[i][:300]!r}")
    md = "\n".join(
        [
            "## Shared errors",
            "",
            f"Averaging all {len(runs)} fine-tuned BERT-base runs: {out['confident_false_positives']:,} test comments labelled "
            f"clean get p(`toxic`) ≥ 0.9, and {out['confident_false_negatives']:,} of the "
            f"{out['toxic_positives']:,} comments labelled toxic get p(`toxic`) ≤ 0.05. Errors every model makes "
            "with confidence cannot be fixed by changing the head; some of them are label noise "
            "(`--show-examples 10` prints a few).",
        ]
    )
    return md, out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def fig_agreement(agree: dict, fig_dir: Path) -> List[Path]:
    """Dot plot: linear-vs-linear and linear-vs-MoE agreement, top of the ranking vs bottom 90%."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sys.path.insert(0, str(ROOT / "scripts"))
    import make_report as mr  # shared colours and chart chrome

    same, cross = f"{name(BASELINE)} vs {name(BASELINE)}", f"{name(BASELINE)} vs {name(MOE)}"
    if same not in agree or cross not in agree:
        return []
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    panels = [
        ("top_overlap", "Top of the ranking: share of the top K both put first", "decides PR-AUC and F1"),
        ("bottom_spearman", "Bottom 90%: rank correlation", "mostly clean comments; barely affects any metric"),
    ]
    for mode in mr.THEMES:
        th: Dict[str, Any] = mr.THEMES[mode]
        fig, axes = mr._figure(plt, th, 11, 4.6, ncols=2, sharey=True)
        fig.subplots_adjust(top=0.7, bottom=0.12, left=0.12, right=0.98, wspace=0.08)
        y = np.arange(len(LABELS))[::-1]
        colors = {same: th["series"]["bert"], cross: th["series"]["moe"]}
        for ax, (key, title, sub) in zip(axes, panels):
            mr._style(ax, th, ygrid=False, xgrid=True)
            for yi, lab in zip(y, LABELS):
                for k, dy in ((same, 0.14), (cross, -0.14)):  # offset so equal values stay visible
                    v = agree[k][lab][key]
                    ax.plot(v, yi + dy, "o", color=colors[k], markersize=8, markeredgecolor=th["surface"],
                            markeredgewidth=1.5, zorder=3)  # fmt: skip
            ax.set_xlim(0, 1.0)
            ax.set_title(f"{title}\n", loc="left", fontsize=10, color=th["ink"], pad=2)
            ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=th["ink2"], va="bottom")
            ax.tick_params(axis="x", labelsize=8)
        axes[0].set_yticks(y, LABELS, fontsize=9)
        mr._title(
            fig,
            th,
            "The MoE head agrees with the linear head where it counts",
            "Test-set rankings of two models trained with different seeds, per label. K = 2 × the label's positives.",
        )
        handles = [Line2D([], [], marker="o", linestyle="", color=colors[k], markersize=8) for k in (same, cross)]
        mr._legend(fig, th, handles, ["Linear head vs another linear seed", "Linear head vs MoE head"], y=0.875)
        path = fig_dir / f"diagnosis_agreement_{mode}.png"
        fig.savefig(path, dpi=160, facecolor=th["surface"])
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--data", help="folder with train.csv / test.csv / test_labels.csv (enables sections 4-10)")
    parser.add_argument("--bootstrap", type=int, default=200, help="bootstrap rounds for section 9 (0 = skip)")
    parser.add_argument("--show-examples", type=int, default=0, help="print N confidently-disagreeing comments")
    parser.add_argument("--out", help="output stem (default: <results>/diagnosis)")
    parser.add_argument("--figures", help="also draw the agreement figure into this folder (e.g. docs/figures)")
    args = parser.parse_args(argv)

    results = Path(args.results)
    groups = load_runs(results)
    if not groups:
        print(f"No */metrics.json under {results}")
        return 1
    n_pred = sum(r.has_predictions for rs in groups.values() for r in rs)
    print(f"{sum(len(v) for v in groups.values())} runs in {len(groups)} settings, {n_pred} with predictions")

    sections: List[Tuple[str, Callable[[], Tuple[str, dict]]]] = [
        ("routing", lambda: routing_specificity(groups)),
        ("misrouting", lambda: misrouting(groups)),
        ("agreement", lambda: agreement(groups)),
        ("expert_offsets", lambda: expert_offsets(groups)),
        ("thresholds", lambda: threshold_transfer(groups)),
        ("val_test_gap", lambda: val_test_gap(groups)),
    ]
    if args.data:
        any_run = next((r for rs in groups.values() for r in rs if r.has_predictions), None)
        if any_run is None:
            print("No predictions.npz files: skipping the label-based sections")
        else:
            ids = any_run.preds()["test_ids"]
            for rs in groups.values():
                for r in with_predictions(rs):
                    if not np.array_equal(r.preds()["test_ids"], ids):
                        raise SystemExit(f"{r.name}: test ids are in a different order from {any_run.name}")
            lab = load_labels(Path(args.data), ids)
            sections += [
                ("labels", lambda: label_structure(lab)),
                ("ensembles", lambda: ensembles(groups, lab)),
                ("subtypes", lambda: subtype_auc(groups, lab)),
                ("length", lambda: length_effect(groups, lab)),
                ("routing_recall", lambda: routing_recall(groups, lab)),
                ("bootstrap", lambda: bootstrap(groups, lab, args.bootstrap)),
                ("disagreements", lambda: disagreements(groups, lab, args.show_examples)),
            ]
    parts = [
        "# Diagnostics: why the MoE head did not beat the linear head",
        "",
        "Generated by `scripts/diagnose.py` from the saved runs; see `docs/diagnosis.md` for the discussion."
        + ("" if args.data else " The sections that need the test labels were skipped (run with `--data`)."),
    ]
    report: Dict[str, Any] = {}
    number = 0
    for key, fn in sections:
        print(f"... {key}", flush=True)
        md, data = fn()
        if md:
            if md.startswith("## "):
                number += 1
                md = f"## {number}. {md[3:]}"
            parts += ["", md]
            report[key] = data
    if args.figures and "agreement" in report:
        for path in fig_agreement(report["agreement"], Path(args.figures)):
            print("wrote", path)
    stem = Path(args.out) if args.out else results / "diagnosis"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text("\n".join(parts) + "\n")
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {stem.with_suffix('.md')} and {stem.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
