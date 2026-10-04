#!/usr/bin/env python
"""Aggregate results/<run>/metrics.json into tables + figures, and refresh the README.

    python scripts/make_report.py                       # results/ -> results/summary.md + docs/figures/*.png
    python scripts/make_report.py --update-readme       # also rewrite the RESULTS block in README.md
    python scripts/make_report.py --results /content/drive/MyDrive/toxic-moe/results

Runs that differ only by seed (``bert_bce_s42``, ``bert_bce_s43``, ...) form one
*setting*; tables show mean ± sample std over seeds. Because every setting uses the
same seeds and the same data split, settings are also compared *paired by seed*.
Only metrics.json files are read, so this runs anywhere (no GPU, no data).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toxic_moe import LABELS  # noqa: E402
from toxic_moe.metrics import mean_pairwise_tv  # noqa: E402

LOSS_ORDER = ["bce", "weighted_bce", "focal", "cb_focal"]
LOSS_NAMES = {"bce": "BCE", "weighted_bce": "Weighted BCE", "focal": "Focal", "cb_focal": "Class-balanced focal"}
HEAD_NAMES = {
    "bert": "linear head",
    "mlp": "MLP head",
    "moe": "MoE head",
    "mmoe": "multi-gate MoE head",
    "label_attn": "label-attention head",
}
HEAD_TITLES = {k: v[0].upper() + v[1:] for k, v in HEAD_NAMES.items()}
HEAD_ORDER = ["bert", "mlp", "moe", "mmoe", "label_attn"]
STUDY_ORDER = ["main", "heads", "probes", "encoders"]
MMOE = "mmoe_bce"
PROBE_BASELINE = "probe_linear"
# frozen-encoder probe -> the fine-tuned BERT-base + BCE setting with the same head
FINE_TUNED_OF = {
    "probe_linear": "bert_bce",
    "probe_mlp": "mlp_bce",
    "probe_moe": "moe_bce",
    "probe_mmoe": MMOE,
    "probe_labelattn": "labelattn_bce",
}
MAIN_ENCODER = "bert-base-uncased"
ENCODER_NAMES = {"bert-base-uncased": "BERT-base", "microsoft/deberta-v3-base": "DeBERTa-v3-base"}
BASELINE, HEADLINE = "bert_bce", "moe_cbfocal"
SEED_SUFFIX = re.compile(r"_s\d+$")
RARE = ("severe_toxic", "threat", "identity_hate")

Metric = Callable[[dict], float]
METRICS: Dict[str, Tuple[str, Metric, int]] = {
    "roc": ("Test ROC-AUC", lambda r: r["test"]["roc_auc_mean"], 4),
    "pr": ("Test PR-AUC", lambda r: r["test"]["pr_auc_mean"], 3),
    "f1_05": ("Macro-F1 @0.5", lambda r: r["test"]["f1@0.5"]["macro"], 3),
    "f1": ("Macro-F1 (val-tuned thr.)", lambda r: r["test"]["f1@tuned"]["macro"], 3),
    "rare": (
        "Rare-label F1¹",
        lambda r: float(np.mean([r["test"]["f1@tuned"]["per_label"][LABELS.index(k)] for k in RARE])),
        3,
    ),
}


def per_label(key: str, j: int) -> Metric:
    """Metric for label ``j``: ``key`` is "f1" (val-tuned thresholds), "pr_auc" or "roc_auc"."""
    if key == "f1":
        return lambda r: r["test"]["f1@tuned"]["per_label"][j]
    return lambda r: r["test"][key][j]


# Validated categorical palette (slot 1 blue, slot 2 orange, slot 3 aqua; the three pass
# the colour-blindness checks as a set) + chart chrome, stepped separately for light and
# dark surfaces. Heads are coloured by family: dense (linear, MLP) blue, routed (MoE,
# multi-gate MoE) orange, attention aqua.
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series={"bert": "#2a78d6", "mlp": "#2a78d6", "moe": "#eb6834", "mmoe": "#eb6834", "label_attn": "#1baf7a"},
        seq=["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series={"bert": "#3987e5", "mlp": "#3987e5", "moe": "#d95926", "mmoe": "#d95926", "label_attn": "#199e70"},
        seq=["#1a1a19", "#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
    ),
}


# ---------------------------------------------------------------------------
# loading + grouping
# ---------------------------------------------------------------------------
def load_runs(results_dir: Path) -> Dict[str, dict]:
    runs = {}
    for path in sorted(results_dir.glob("*/metrics.json")):
        with open(path) as f:
            runs[path.parent.name] = json.load(f)
    return runs


@dataclass
class Setting:
    """All seeds of one configuration."""

    name: str
    runs: List[dict]

    @property
    def first(self) -> dict:
        return self.runs[0]

    @property
    def arch(self) -> str:
        return self.first["arch"]

    @property
    def loss(self) -> str:
        return self.first["loss"]

    @property
    def encoder(self) -> str:
        return self.first.get("encoder", MAIN_ENCODER)

    @property
    def study(self) -> str:
        """Which table the setting belongs to; runs from before the field existed are main or encoders."""
        return self.first.get("study") or ("main" if self.encoder == MAIN_ENCODER else "encoders")

    @property
    def title(self) -> str:
        return (
            self.first.get("title")
            or f"{HEAD_TITLES.get(self.arch, self.arch)}, {LOSS_NAMES.get(self.loss, self.loss)}"
        )

    @property
    def seeds(self) -> List[Optional[int]]:
        return [r.get("seed") for r in self.runs]

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def label(self) -> str:
        enc = ENCODER_NAMES.get(self.encoder, self.encoder)
        return f"{enc} + {HEAD_NAMES.get(self.arch, self.arch)}, {LOSS_NAMES.get(self.loss, self.loss)}"

    def values(self, fn: Metric) -> np.ndarray:
        return np.array([fn(r) for r in self.runs], dtype=float)

    def mean(self, fn: Metric) -> float:
        return float(np.mean(self.values(fn)))

    def std(self, fn: Metric) -> float:
        v = self.values(fn)
        return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def group_runs(runs: Dict[str, dict]) -> Dict[str, Setting]:
    groups: Dict[str, List[dict]] = {}
    for name, r in runs.items():
        key = r.get("group") or SEED_SUFFIX.sub("", name)
        groups.setdefault(key, []).append(r)
    return {
        k: Setting(k, sorted(v, key=lambda r: (r.get("seed") is None, r.get("seed") or 0))) for k, v in groups.items()
    }


def sort_key(s: Setting):
    return (
        STUDY_ORDER.index(s.study) if s.study in STUDY_ORDER else 99,
        s.encoder != MAIN_ENCODER,
        s.encoder,
        HEAD_ORDER.index(s.arch) if s.arch in HEAD_ORDER else 99,
        LOSS_ORDER.index(s.loss) if s.loss in LOSS_ORDER else 99,
        s.name,
    )


def paired_delta(a: Setting, b: Setting, fn: Metric) -> Tuple[float, float, int]:
    """Mean ± std of (b - a) over the seeds both settings share."""
    va = {r.get("seed"): fn(r) for r in a.runs}
    diffs = np.array([fn(r) - va[r.get("seed")] for r in b.runs if r.get("seed") in va], dtype=float)
    if len(diffs) == 0:
        return float("nan"), float("nan"), 0
    return float(diffs.mean()), float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0, len(diffs)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _fmt(x: Optional[float], digits: int) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _cell(s: Setting, fn: Metric, digits: int) -> str:
    if s.n > 1:
        return f"{s.mean(fn):.{digits}f} ± {s.std(fn):.{digits}f}"
    return _fmt(s.mean(fn), digits)


def metric_table(settings: List[Setting], first_cols: Callable[[Setting], List[str]], header: List[str]) -> str:
    keys = ["roc", "pr", "f1_05", "f1", "rare"]
    lines = [
        "| " + " | ".join(header + [METRICS[k][0] for k in keys]) + " |",
        "|" + "---|" * len(header) + "---:|" * len(keys),
    ]
    best = {k: max(s.mean(METRICS[k][1]) for s in settings) for k in keys}
    for s in settings:
        cells = []
        for k in keys:
            _, fn, digits = METRICS[k]
            c = _cell(s, fn, digits)
            cells.append(f"**{c}**" if np.isclose(s.mean(fn), best[k]) and len(settings) > 1 else c)
        lines.append("| " + " | ".join(first_cols(s) + cells) + " |")
    return "\n".join(lines)


def per_label_table(a: Setting, b: Setting) -> str:
    lines = [
        f"| Label | Test positives | {a.name} PR-AUC | {b.name} PR-AUC | {a.name} F1 | {b.name} F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    pos = a.first["test"]["positives"]
    for j, label in enumerate(LABELS):
        pr, f1 = per_label("pr_auc", j), per_label("f1", j)
        lines.append(
            f"| `{label}` | {pos[j]:,} | {_cell(a, pr, 3)} | {_cell(b, pr, 3)} | {_cell(a, f1, 3)} | {_cell(b, f1, 3)} |"
        )
    return "\n".join(lines)


def delta_lines(a: Setting, b: Setting) -> List[str]:
    out = []
    for k in ("roc", "pr", "f1", "rare"):
        title, fn, digits = METRICS[k]
        m, sd, n = paired_delta(a, b, fn)
        if n:
            spread = f" ± {sd:.{digits}f}" if n > 1 else ""
            out.append(f"{title.replace('¹', '')}: **{m:+.{digits}f}**{spread}")
    n = paired_delta(a, b, METRICS["roc"][1])[2]
    seeds = f"paired over {n} seeds" if n > 1 else "single seed"
    return [f"**{b.label}** vs **{a.label}** ({seeds}): " + "; ".join(out) + "."]


DELTA_KEYS = ["roc", "pr", "f1", "rare"]


def _delta_cell(base: Setting, s: Setting, key: str) -> str:
    _, fn, digits = METRICS[key]
    m, sd, n = paired_delta(base, s, fn)
    return "–" if not n else (f"{m:+.{digits}f} ± {sd:.{digits}f}" if n > 1 else f"{m:+.{digits}f}")


def paired_table(
    base: Setting,
    others: List[Setting],
    first_cols: Optional[Callable[[Setting], List[str]]] = None,
    header: Optional[List[str]] = None,
) -> str:
    """Each setting minus the baseline, computed per seed, then mean ± std over seeds."""
    first_cols = first_cols or (lambda s: [HEAD_TITLES.get(s.arch, s.arch), LOSS_NAMES.get(s.loss, s.loss)])
    header = header or ["Head", "Loss"]
    titles = [METRICS[k][0].replace("¹", "").replace("Test ", "") for k in DELTA_KEYS]
    lines = [
        "| " + " | ".join(header + [f"Δ {t}" for t in titles]) + " |",
        "|" + "---|" * len(header) + "---:|" * len(DELTA_KEYS),
    ]
    for s in others:
        lines.append("| " + " | ".join(first_cols(s) + [_delta_cell(base, s, k) for k in DELTA_KEYS]) + " |")
    return "\n".join(lines)


def routing_summary(settings: Dict[str, Setting]) -> Optional[str]:
    """How concentrated the MoE router is: toxic comments vs clean comments."""
    toxic, clean, n_runs, n_experts = [], [], 0, 0
    for s in settings.values():
        if s.study != "main" or s.arch != "moe":
            continue
        for r in s.runs:
            g = r["test"].get("gates", {}).get("mean_gate", {})
            if "toxic" not in g or "clean" not in g:
                continue
            top2 = np.argsort(g["toxic"])[-2:]  # the two experts used most by toxic comments
            clean.append(float(np.sum(np.asarray(g["clean"])[top2])))
            toxic += [float(np.sum(np.asarray(g[lab])[top2])) for lab in LABELS if lab in g]
            n_runs, n_experts = n_runs + 1, len(g["clean"])
    if not n_runs:
        return None
    text = (
        f"**Routing.** Across all {n_runs} MoE runs, comments carrying any toxic label put on average "
        f"{np.mean(toxic):.0%} (minimum {np.min(toxic):.0%}) of their router weight on the same two experts, "
        f"whatever the kind of toxicity, while clean comments put {np.mean(clean):.0%} on those two experts "
        f"(uniform routing would be {2 / n_experts:.0%})."
    )
    if np.mean(toxic) >= 0.7 and np.mean(clean) <= 0.45:
        text += " The router learned a toxic-vs-clean split, not experts for specific kinds of toxicity."
    return text


def label_divergence(run: dict) -> Optional[float]:
    """How differently the labels use the experts (0 = same experts, 1 = disjoint), from metrics.json."""
    g = run["test"].get("gates") or {}
    if "label_divergence" in g:
        return float(g["label_divergence"])
    rows = [g["mean_gate"][k] for k in LABELS if k in g.get("mean_gate", {})]  # runs from before the field existed
    return mean_pairwise_tv(np.array(rows)) if len(rows) > 1 else None


def gating_summary(settings: Dict[str, Setting]) -> Optional[str]:
    """Per-comment router (MoE) vs per-label gates (multi-gate MoE): do the labels use different experts?"""

    def collect(arch: str, key: str) -> List[float]:
        out = []
        for s in settings.values():
            if (
                s.arch == arch
                and s.study in ("main", "heads")
                and s.encoder == MAIN_ENCODER
                and not s.first.get("freeze_encoder")
            ):
                for r in s.runs:
                    v = label_divergence(r) if key == "label" else (r["test"].get("gates") or {}).get(key)
                    if v is not None:
                        out.append(float(v))
        return out

    moe, mmoe = collect("moe", "label"), collect("mmoe", "label")
    within = collect("mmoe", "within_comment_divergence")
    if not mmoe or not moe:
        return None
    return (
        "**Do the labels use different experts?** Mean distance between the expert mixtures of the six labels' "
        f"positive comments (total variation; 0 = the same experts, 1 = disjoint experts): MoE router "
        f"{np.mean(moe):.2f}, multi-gate MoE {np.mean(mmoe):.2f}. For one and the same comment, the multi-gate "
        f"head's six label gates differ by {np.mean(within):.2f} on average; a per-comment router scores 0 by "
        "construction."
    )


def heads_section(settings: Dict[str, Setting]) -> List[str]:
    base = settings.get(BASELINE)
    heads = [s for s in sorted(settings.values(), key=sort_key) if s.study == "heads"]
    if base is None or not heads:
        return []
    rows = [settings[n] for n in ("moe_bce",) if n in settings] + heads
    n = base.n
    parts = [
        "",
        "**Follow-up: heads built around the MoE head's failure** (BERT-base, plain BCE, each minus the "
        f"linear-head baseline on the same seed; {'mean ± std over ' + str(n) + ' seeds' if n > 1 else 'one seed'}). "
        "Why these heads: `docs/diagnosis.md`.",
        "",
        paired_table(
            base, rows, lambda s: [s.title, f"{s.first.get('head_parameters', 0) / 1e6:.2f}M"], ["Head", "Head params"]
        ),
    ]
    gating = gating_summary(settings)
    if gating:
        parts += ["", gating]
    return parts


def probes_section(settings: Dict[str, Setting]) -> List[str]:
    probes = [s for s in sorted(settings.values(), key=sort_key) if s.study == "probes"]
    base = settings.get(PROBE_BASELINE)
    if not probes or base is None:
        return []
    roc, pr = METRICS["roc"][1], METRICS["pr"][1]
    lines = [
        "| Head | Frozen: test ROC-AUC | Frozen: test PR-AUC | Frozen: Δ ROC-AUC vs linear | "
        "Fine-tuned: Δ ROC-AUC vs linear |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in probes:
        frozen = "baseline" if s.name == base.name else _delta_cell(base, s, "roc")
        tuned = settings.get(FINE_TUNED_OF.get(s.name, ""))
        if s.name == PROBE_BASELINE:
            fine = "baseline"
        elif s.name not in FINE_TUNED_OF:
            fine = "–"
        elif tuned is not None and BASELINE in settings:
            fine = _delta_cell(settings[BASELINE], tuned, "roc")
        else:
            fine = "not run yet"
        lines.append(f"| {s.title} | {_cell(s, roc, 4)} | {_cell(s, pr, 3)} | {frozen} | {fine} |")
    return [
        "",
        "**Frozen encoder vs fine-tuned encoder.** The same heads trained on fixed BERT-base features "
        "(`configs/probes.yaml`). A head that helps only when the encoder cannot adapt is doing work that "
        "fine-tuning already does.",
        "",
        "\n".join(lines),
    ]


def details_table(settings: List[Setting]) -> str:
    lines = [
        "| Setting | Seeds | Best epoch per seed | Min / run | Precision | GPU |",
        "|---|---|---|---:|---|---|",
    ]
    for s in settings:
        env = s.first.get("environment", {})
        mins = np.mean([r.get("wall_clock_minutes", np.nan) for r in s.runs])
        lines.append(
            f"| `{s.name}` | {', '.join(str(x) for x in s.seeds)} | {', '.join(str(r.get('best_epoch', '–')) for r in s.runs)} "
            f"| {_fmt(float(mins), 1)} | {env.get('precision', '–')} | {env.get('gpu', env.get('device', '–'))} |"
        )
    return "\n".join(lines)


def picture(prefix: str, stem: str, alt: str) -> str:
    return (
        f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{prefix}/{stem}_dark.png">\n'
        f'  <img alt="{alt}" src="{prefix}/{stem}_light.png">\n</picture>'
    )


def summary_markdown(settings: Dict[str, Setting], figure_prefix: str, figure_dir: Path) -> str:
    ordered = sorted(settings.values(), key=sort_key)
    main = [s for s in ordered if s.study == "main"]
    others = [s for s in ordered if s.study == "encoders"]
    d = ordered[0].first["data"]
    seeds = sorted({x for s in ordered for x in s.seeds if x is not None})
    seed_note = (
        f"Every setting is trained with {len(seeds)} seeds ({', '.join(map(str, seeds))}) on the same data split; "
        "cells show mean ± sample std over seeds."
        if len(seeds) > 1
        else "Single seed per setting."
    )
    parts = [
        f"Trained on {d['train']:,} comments ({d['train_fraction']:.0%} of `train.csv` minus a {d['val']:,}-comment "
        f"validation split) and evaluated on the **{d['test']:,}-comment official Kaggle test set**. F1 thresholds are "
        f"tuned per label on the validation split, never on test. {seed_note} Best mean per column in bold; the paired "
        "table shows which differences are larger than seed-to-seed noise.",
    ]
    if BASELINE in settings and len(main) > 1:
        n = settings[BASELINE].n
        parts += [
            "",
            f"**Difference from the baseline (BERT + linear head + BCE), paired by seed.** Positive = better than the "
            f"baseline. {'Mean ± std over ' + str(n) + ' seeds.' if n > 1 else 'Single seed.'}",
            "",
            paired_table(settings[BASELINE], [x for x in main if x.name != BASELINE]),
        ]
    routing = routing_summary(settings)
    if routing:
        parts += ["", routing]
    if main:
        parts += [
            "",
            "**Loss × head ablation (BERT-base encoder)**",
            "",
            metric_table(main, lambda s: [HEAD_TITLES[s.arch], LOSS_NAMES.get(s.loss, s.loss)], ["Head", "Loss"]),
            "",
            "¹ Mean test F1 over the three rarest labels: `severe_toxic`, `threat`, `identity_hate`.",
        ]
    parts += heads_section(settings)
    parts += probes_section(settings)
    if others:
        enc_rows = [
            s
            for s in ordered
            if s.study in ("main", "encoders") and (s.arch, s.loss) in (("bert", "bce"), ("moe", "cb_focal"))
        ]
        parts += [
            "",
            "**Stronger encoder**",
            "",
            metric_table(
                enc_rows,
                lambda s: [
                    ENCODER_NAMES.get(s.encoder, s.encoder),
                    f"{HEAD_TITLES[s.arch]} + {LOSS_NAMES.get(s.loss, s.loss)}",
                ],
                ["Encoder", "Model"],
            ),
        ]
        for enc in sorted({s.encoder for s in others}):
            base = next((s for s in others if s.encoder == enc and s.arch == "bert" and s.loss == "bce"), None)
            head = next((s for s in others if s.encoder == enc and s.arch == "moe" and s.loss == "cb_focal"), None)
            if base and head:
                parts += ["", *delta_lines(base, head)]
    if BASELINE in settings and HEADLINE in settings:
        parts += [
            "",
            "**Per label (mean over seeds)**",
            "",
            per_label_table(settings[BASELINE], settings[HEADLINE]),
        ]
    figs = [
        ("ablation", "Each setting minus the BERT + BCE baseline, paired by seed"),
        ("heads", "Follow-up heads minus the BERT + BCE baseline, paired by seed"),
        ("probes", "Head gains with a frozen vs a fine-tuned encoder"),
        ("mmoe_gates", "Which experts each label's gate uses in the multi-gate MoE head"),
        ("encoders", "BERT-base vs DeBERTa-v3-base"),
        ("per_label_f1", "Per-label F1, baseline vs MoE + CB-focal"),
        ("pr_curves", "Precision–recall curves on the test set"),
        ("expert_routing", "Which experts the router uses for each label"),
    ]
    for stem, alt in figs:
        if (figure_dir / f"{stem}_light.png").exists():
            parts += ["", picture(figure_prefix, stem, alt)]
    parts += ["", "<details><summary>Run details</summary>", "", details_table(ordered), "", "</details>"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _style(ax, th, ygrid=True, xgrid=False):
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=th["muted"], labelcolor=th["ink2"], length=0, labelsize=9)
    ax.grid(False)
    if ygrid:
        ax.yaxis.grid(True, color=th["grid"], linewidth=1, linestyle="-")
    if xgrid:
        ax.xaxis.grid(True, color=th["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def _figure(plt, th, w, h, ncols=1, nrows=1, **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=th["surface"], **kw)
    return fig, np.atleast_1d(axes).ravel()


def _title(fig, th, title, subtitle=None):
    fig.text(0.01, 0.98, title, ha="left", va="top", fontsize=13, fontweight="bold", color=th["ink"])
    if subtitle:
        fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=th["ink2"])


def _legend(fig, th, handles, labels, y=0.86):
    leg = fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.01, y),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        handlelength=1.2,
        columnspacing=1.5,
    )
    for t in leg.get_texts():
        t.set_color(th["ink2"])


def _grouped_bars(ax, th, x_labels, series, width=0.36):
    """series: list of (arch_for_color, means, stds). Returns bar handles and best (value, x)."""
    x = np.arange(len(x_labels))
    handles: list = []
    all_vals: list = []
    best: tuple = (-np.inf, None, 0.0)
    for k, (arch, means, stds) in enumerate(series):
        offs = (k - (len(series) - 1) / 2) * (width + 0.04)
        bars = ax.bar(x + offs, means, width, color=th["series"][arch], edgecolor=th["surface"], linewidth=2)
        if np.any(np.asarray(stds) > 0):
            ax.errorbar(x + offs, means, yerr=stds, fmt="none", ecolor=th["ink2"], elinewidth=1, capsize=3)
        handles.append(bars[0])
        for xi, m, s in zip(x + offs, means, stds):
            if not np.isnan(m):
                all_vals += [m - s, m + s]
                if m > best[0]:
                    best = (m, xi, s)
    ax.set_xticks(x, x_labels, fontsize=9)
    return handles, all_vals, best


def _zoom(ax, vals):
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.6 + 1e-3
    ax.set_ylim(max(0, lo - pad), min(1, hi + pad))


FAMILIES = [
    ("Dense head", ("bert", "mlp")),
    ("Routed head (MoE family)", ("moe", "mmoe")),
    ("Attention head", ("label_attn",)),
]


def _forest(plt, th, base: Setting, rows: List[Setting], row_labels: List[str], title: str, sub: str, out: Path):
    """Forest plot: each row minus the baseline, paired by seed (dot = mean, line = ±1 std)."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator

    keys = ["roc", "pr", "f1"]
    height = 2.0 + 0.42 * len(rows)
    left = min(0.36, 0.06 + 0.0068 * max(len(t) for t in row_labels))
    fig, axes = _figure(plt, th, 12, height, ncols=3, sharey=True)
    fig.subplots_adjust(top=1 - 1.2 / height, bottom=0.35 / height + 0.04, left=left, right=0.98, wspace=0.12)
    y = np.arange(len(rows))[::-1]
    for ax, key in zip(axes, keys):
        title_k, fn, digits = METRICS[key]
        _style(ax, th, ygrid=False, xgrid=True)
        ax.axvline(0, color=th["ink2"], linewidth=1, zorder=1)
        lim = 0.0
        for yi, setting in zip(y, rows):
            m, sd, n = paired_delta(base, setting, fn)
            if not n:
                continue
            color = th["series"].get(setting.arch, th["ink2"])
            ax.errorbar(
                m,
                yi,
                xerr=sd,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=2,
                capsize=0,
                markersize=7,
                markeredgecolor=th["surface"],
                markeredgewidth=1.5,
                zorder=3,
            )
            lim = max(lim, abs(m) + sd)
        lim = lim * 1.3 or 10**-digits
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter(f"%+.{digits}f"))
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(f"Δ {title_k.replace('Test ', 'test ')}", loc="left", fontsize=10, color=th["ink"], pad=8)
    axes[0].set_yticks(y, row_labels, fontsize=9)
    axes[0].set_ylim(-0.7, len(rows) - 0.3)
    if base.n > 1:
        sub += f" Dot: mean of {base.n} seeds; line: ±1 std."
    _title(fig, th, title, sub)
    present = {r.arch for r in rows} | {base.arch}
    legend = []  # one entry per head family; named after the head when only one head of the family is shown
    for family, archs in FAMILIES:
        shown = [a for a in archs if a in present]
        if shown:
            legend.append((HEAD_TITLES[shown[0]] if len(shown) == 1 else family, shown[0]))
    handles = [Line2D([], [], marker="o", linestyle="", color=th["series"][a], markersize=7) for _, a in legend]
    _legend(fig, th, handles, [name for name, _ in legend], y=1 - 0.62 / height)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def fig_ablation(plt, settings: Dict[str, Setting], th, out: Path):
    base = settings[BASELINE]
    rows = [x for x in sorted(settings.values(), key=sort_key) if x.study == "main" and x.name != BASELINE]
    labels = [f"{HEAD_TITLES[x.arch]}, {LOSS_NAMES.get(x.loss, x.loss)}" for x in rows]
    sub = "Each setting minus BERT + linear head + BCE on the same seed. Right of 0 = better than the baseline."
    _forest(plt, th, base, rows, labels, "Difference from the BERT + BCE baseline", sub, out)


def fig_heads(plt, settings: Dict[str, Setting], th, out: Path):
    base = settings[BASELINE]
    rows = [settings[n] for n in ("moe_bce",) if n in settings]
    rows += [x for x in sorted(settings.values(), key=sort_key) if x.study == "heads"]
    sub = "BERT-base, plain BCE. Each head minus the linear head on the same seed. Right of 0 = better."
    _forest(plt, th, base, rows, [x.title for x in rows], "Follow-up heads vs the linear head", sub, out)


def fig_probes(plt, settings: Dict[str, Setting], th, out: Path):
    """Dumbbell: gain over the linear head with a frozen encoder vs with a fine-tuned encoder."""
    from matplotlib.lines import Line2D

    roc = METRICS["roc"][1]
    rows = []
    for s in sorted(settings.values(), key=sort_key):
        if s.study != "probes" or s.name == PROBE_BASELINE:
            continue
        frozen = paired_delta(settings[PROBE_BASELINE], s, roc)
        tuned = settings.get(FINE_TUNED_OF.get(s.name, ""))
        fine = paired_delta(settings[BASELINE], tuned, roc) if tuned is not None and BASELINE in settings else None
        rows.append((s.title, frozen, fine))
    height = 2.2 + 0.5 * len(rows)
    fig, axes = _figure(plt, th, 9, height)
    ax = axes[0]
    fig.subplots_adjust(top=1 - 1.25 / height, bottom=0.55 / height + 0.04, left=0.26, right=0.97)
    _style(ax, th, ygrid=False, xgrid=True)
    ax.axvline(0, color=th["ink2"], linewidth=1, zorder=1)
    y = np.arange(len(rows))[::-1]
    c_frozen, c_fine = th["series"]["moe"], th["series"]["bert"]
    for yi, (_, frozen, fine) in zip(y, rows):
        points = [(frozen, c_frozen)]
        if fine is not None and fine[2]:
            points.append((fine, c_fine))
            ax.plot([frozen[0], fine[0]], [yi, yi], color=th["axis"], linewidth=2, zorder=2)
        for (m, sd, n), color in points:
            if n:
                ax.errorbar(m, yi, xerr=sd, fmt="o", color=color, ecolor=color, elinewidth=2, markersize=8,
                            markeredgecolor=th["surface"], markeredgewidth=1.5, zorder=3)  # fmt: skip
                ax.text(m, yi + 0.2, f"{m:+.4f}", ha="center", va="bottom", fontsize=8, color=th["ink2"])
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.set_xlabel("Δ test ROC-AUC vs the linear head (same encoder setting)", color=th["ink2"], fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    _title(fig, th, "Does the head matter once the encoder is fine-tuned?",
           "Frozen: head trained on fixed BERT-base features. Fine-tuned: encoder trained with the head.")  # fmt: skip
    handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=8) for c in (c_frozen, c_fine)]
    _legend(fig, th, handles, ["Frozen encoder", "Fine-tuned encoder"], y=1 - 0.68 / height)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def fig_encoders(plt, settings: Dict[str, Setting], th, out: Path):
    settings = {k: s for k, s in settings.items() if s.study in ("main", "encoders")}  # not probes / follow-ups
    encoders = sorted({s.encoder for s in settings.values()}, key=lambda e: (e != MAIN_ENCODER, e))
    combos = [("bert", "bce"), ("moe", "cb_focal")]
    fig, axes = _figure(plt, th, 9, 4.2, ncols=2)
    fig.subplots_adjust(top=0.72, bottom=0.14, left=0.08, right=0.99, wspace=0.3)
    handles: list = []
    for ax, key in zip(axes, ["roc", "f1"]):
        title, fn, digits = METRICS[key]
        _style(ax, th)
        series = []
        for arch, loss in combos:
            found = [
                next((s for s in settings.values() if s.encoder == e and s.arch == arch and s.loss == loss), None)
                for e in encoders
            ]
            series.append(
                (arch, [s.mean(fn) if s else np.nan for s in found], [s.std(fn) if s else 0.0 for s in found])
            )
        h, vals, best = _grouped_bars(ax, th, [ENCODER_NAMES.get(e, e) for e in encoders], series)
        handles = handles or h
        _zoom(ax, vals)
        if best[1] is not None:
            ax.text(
                best[1],
                best[0] + best[2],
                f"{best[0]:.{digits}f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=th["ink"],
                fontweight="bold",
            )
        ax.set_title(title, loc="left", fontsize=10, color=th["ink"], pad=8)
    _title(fig, th, "Does the gain hold on a stronger encoder?", "Mean of seeds, whiskers ±1 std. Y-axes are zoomed.")
    _legend(fig, th, handles, ["Linear head + BCE", "MoE head + class-balanced focal"], y=0.86)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def fig_per_label(plt, settings: Dict[str, Setting], th, out: Path):
    pair = [settings[n] for n in (BASELINE, HEADLINE) if n in settings]
    fig, axes = _figure(plt, th, 9, 4.2)
    ax = axes[0]
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.07, right=0.99)
    _style(ax, th)
    series = []
    for s in pair:
        means = [s.mean(per_label("f1", j)) for j in range(len(LABELS))]
        stds = [s.std(per_label("f1", j)) for j in range(len(LABELS))]
        series.append((s.arch, means, stds))
    pos = pair[0].first["test"]["positives"]
    handles, _, _ = _grouped_bars(ax, th, [f"{l}\n({p:,} pos.)" for l, p in zip(LABELS, pos)], series)
    for k, (_, means, stds) in enumerate(series):
        offs = (k - (len(series) - 1) / 2) * 0.4
        for xi, m, sd in zip(np.arange(len(LABELS)) + offs, means, stds):
            ax.text(xi, m + sd + 0.01, f"{m:.2f}", ha="center", va="bottom", fontsize=7.5, color=th["ink2"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1", color=th["ink2"], fontsize=9)
    n = max(s.n for s in pair)
    sub = "Thresholds tuned on the validation split." + (f" Mean of {n} seeds, whiskers ±1 std." if n > 1 else "")
    _title(fig, th, "Per-label F1 on the test set", sub)
    _legend(fig, th, handles, [s.label for s in pair], y=0.87)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def fig_pr_curves(plt, settings: Dict[str, Setting], th, out: Path):
    pair = [settings[n] for n in (BASELINE, HEADLINE) if n in settings and "pr_curves" in settings[n].first["test"]]
    fig, axes = _figure(plt, th, 12, 6.2, ncols=3, nrows=2)
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.05, right=0.99, hspace=0.45, wspace=0.18)
    handles = []
    for j, (ax, label) in enumerate(zip(axes, LABELS)):
        _style(ax, th, ygrid=True, xgrid=True)
        for s in pair:
            curves = [r["test"]["pr_curves"][label] for r in s.runs if label in r["test"].get("pr_curves", {})]
            if not curves:
                continue
            recall = curves[0]["recall"]
            precision = np.mean([c["precision"] for c in curves], axis=0)  # same recall grid for every seed
            (line,) = ax.plot(recall, precision, color=th["series"][s.arch], linewidth=2, solid_capstyle="round")
            if j == 0:
                handles.append(line)
        ap = "  ".join(_fmt(s.mean(per_label("pr_auc", j)), 3) for s in pair)
        ax.set_title(f"{label}   AP {ap}", loc="left", fontsize=9.5, color=th["ink"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        if j >= 3:
            ax.set_xlabel("Recall", color=th["ink2"], fontsize=9)
        if j % 3 == 0:
            ax.set_ylabel("Precision", color=th["ink2"], fontsize=9)
    n = max(s.n for s in pair)
    _title(
        fig,
        th,
        "Precision–recall on the test set",
        "Interpolated precision at each recall"
        + (f", averaged over {n} seeds" if n > 1 else "")
        + ". AP = average precision, listed in legend order.",
    )
    _legend(fig, th, handles, [s.label for s in pair], y=0.905)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def _heatmap(plt, th, mat: np.ndarray, rows: List[str], title: str, sub: str, out: Path):
    from matplotlib.colors import LinearSegmentedColormap

    n_exp = mat.shape[1]
    cmap = LinearSegmentedColormap.from_list("seq", th["seq"])
    fig, axes = _figure(plt, th, max(7.5, 2.4 + 0.85 * n_exp), 1.75 + 0.33 * len(rows))
    ax = axes[0]
    fig.subplots_adjust(top=1 - 0.95 / (1.75 + 0.33 * len(rows)), bottom=0.08, left=0.22, right=0.97)
    ax.set_facecolor(th["surface"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(float(mat.max()), 2.0 / n_exp), aspect="auto")
    for i in range(mat.shape[0]):
        for e in range(n_exp):
            v = mat[i, e]
            rgba = im.cmap(im.norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                e, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color="#0b0b0b" if lum > 0.55 else "#ffffff"
            )
    ax.set_xticks(range(n_exp), [f"E{e + 1}" for e in range(n_exp)])
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0, labelcolor=th["ink2"], labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_exp, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=th["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)
    _title(fig, th, title, sub)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def fig_routing(plt, settings: Dict[str, Setting], th, out: Path):
    # Expert indices are arbitrary per seed (any permutation is equivalent), so routing
    # is shown for a single run rather than averaged.
    r = settings[HEADLINE].first
    gates = r["test"]["gates"]["mean_gate"]
    rows = [k for k in ["all", "clean", *LABELS] if k in gates]
    seed = f", seed {r['seed']}" if r.get("seed") is not None else ""
    _heatmap(
        plt,
        th,
        np.array([gates[k] for k in rows]),
        [("all comments" if k == "all" else k) for k in rows],
        f"Expert routing (MoE + CB-focal{seed})",
        "Mean router weight per expert, over test comments with each label. Expert numbers are arbitrary per seed.",
        out,
    )


def fig_mmoe_gates(plt, settings: Dict[str, Setting], th, out: Path):
    r = settings[MMOE].first
    gates = r["test"]["gates"]["label_gates"]
    rows = [k for k in LABELS if k in gates]
    seed = f", seed {r['seed']}" if r.get("seed") is not None else ""
    _heatmap(
        plt,
        th,
        np.array([gates[k] for k in rows]),
        rows,
        f"Per-label gates (multi-gate MoE{seed})",
        "Each label's own gate weight per expert, averaged over the test comments carrying that label.",
        out,
    )


def make_figures(settings: Dict[str, Setting], fig_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    fig_dir.mkdir(parents=True, exist_ok=True)
    main = [s for s in settings.values() if s.study == "main"]
    has_pair = BASELINE in settings and HEADLINE in settings
    mmoe = settings.get(MMOE)
    jobs: list = [
        ("ablation", fig_ablation, BASELINE in settings and len(main) >= 2),
        ("heads", fig_heads, BASELINE in settings and any(s.study == "heads" for s in settings.values())),
        ("probes", fig_probes, PROBE_BASELINE in settings and sum(s.study == "probes" for s in settings.values()) > 1),
        ("mmoe_gates", fig_mmoe_gates, mmoe is not None and "label_gates" in (mmoe.first["test"].get("gates") or {})),
        ("encoders", fig_encoders, len({s.encoder for s in settings.values()}) > 1),
        ("per_label_f1", fig_per_label, has_pair),
        ("pr_curves", fig_pr_curves, has_pair),
        ("expert_routing", fig_routing, HEADLINE in settings and "gates" in settings[HEADLINE].first["test"]),
    ]
    written = []
    for mode, th in THEMES.items():
        for stem, fn, ok in jobs:
            if ok:
                path = fig_dir / f"{stem}_{mode}.png"
                fn(plt, settings, th, path)
                written.append(path)
    return written


def update_readme(readme: Path, block: str) -> None:
    text = readme.read_text()
    pattern = re.compile(r"(<!-- RESULTS:START -->)(.*?)(<!-- RESULTS:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"No <!-- RESULTS:START --> ... <!-- RESULTS:END --> block in {readme}")
    readme.write_text(pattern.sub(lambda m: f"{m.group(1)}\n{block}\n{m.group(3)}", text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--figures", default=str(ROOT / "docs" / "figures"))
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    runs = load_runs(Path(args.results))
    if not runs:
        print(f"No */metrics.json found under {args.results}")
        return 1
    settings = group_runs(runs)
    print(
        f"Found {len(runs)} runs in {len(settings)} settings: "
        + ", ".join(f"{k} (n={v.n})" for k, v in settings.items())
    )
    fig_dir = Path(args.figures).resolve()
    if not args.no_figures:
        for p in make_figures(settings, fig_dir):
            print("wrote", p)
    fig_prefix = fig_dir.relative_to(ROOT).as_posix() if fig_dir.is_relative_to(ROOT) else str(fig_dir)
    block = summary_markdown(settings, fig_prefix, fig_dir)
    summary = Path(args.results) / "summary.md"
    # summary.md sits in results/, one level below the repo root
    block_for_summary = (
        block if Path(fig_prefix).is_absolute() else block.replace(f"{fig_prefix}/", f"../{fig_prefix}/")
    )
    summary.write_text("# Results\n\n" + block_for_summary + "\n")
    print("wrote", summary)
    if args.update_readme:
        update_readme(ROOT / "README.md", block)
        print("updated README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
