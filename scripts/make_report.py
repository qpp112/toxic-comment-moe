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

LOSS_ORDER = ["bce", "weighted_bce", "focal", "cb_focal"]
LOSS_NAMES = {"bce": "BCE", "weighted_bce": "Weighted BCE", "focal": "Focal", "cb_focal": "Class-balanced focal"}
HEAD_NAMES = {"bert": "linear head", "moe": "MoE head"}
HEAD_TITLES = {"bert": "Linear head", "moe": "MoE head"}
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


# Validated two-series categorical palette (slot 1 blue, slot 2 orange) + chart chrome,
# stepped separately for light and dark surfaces.
THEMES = {
    "light": dict(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series={"bert": "#2a78d6", "moe": "#eb6834"},
        seq=["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    ),
    "dark": dict(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series={"bert": "#3987e5", "moe": "#d95926"},
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
        s.encoder != MAIN_ENCODER,
        s.encoder,
        s.arch != "bert",
        LOSS_ORDER.index(s.loss) if s.loss in LOSS_ORDER else 99,
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
    main = [s for s in ordered if s.encoder == MAIN_ENCODER]
    others = [s for s in ordered if s.encoder != MAIN_ENCODER]
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
        f"tuned per label on the validation split, never on test. {seed_note} Best mean per column in bold.",
    ]
    if BASELINE in settings and HEADLINE in settings:
        parts += ["", *delta_lines(settings[BASELINE], settings[HEADLINE])]
    if main:
        parts += [
            "",
            "**Loss × head ablation (BERT-base encoder)**",
            "",
            metric_table(main, lambda s: [HEAD_TITLES[s.arch], LOSS_NAMES.get(s.loss, s.loss)], ["Head", "Loss"]),
            "",
            "¹ Mean test F1 over the three rarest labels: `severe_toxic`, `threat`, `identity_hate`.",
        ]
    if others:
        enc_rows = [s for s in ordered if (s.arch, s.loss) in (("bert", "bce"), ("moe", "cb_focal"))]
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
        ("ablation", "Loss × head ablation on the test set"),
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


def fig_ablation(plt, settings: Dict[str, Setting], th, out: Path):
    main = [s for s in settings.values() if s.encoder == MAIN_ENCODER]
    losses = [l for l in LOSS_ORDER if any(s.loss == l for s in main)]
    archs = [a for a in ("bert", "moe") if any(s.arch == a for s in main)]
    fig, axes = _figure(plt, th, 12, 4.2, ncols=3)
    fig.subplots_adjust(top=0.72, bottom=0.2, left=0.05, right=0.99, wspace=0.28)
    handles: list = []
    for ax, key in zip(axes, ["roc", "pr", "f1"]):
        title, fn, digits = METRICS[key]
        _style(ax, th)
        series = []
        for arch in archs:
            found = [next((s for s in main if s.arch == arch and s.loss == l), None) for l in losses]
            series.append(
                (
                    arch,
                    [s.mean(fn) if s else np.nan for s in found],
                    [s.std(fn) if s else 0.0 for s in found],
                )
            )
        h, vals, best = _grouped_bars(ax, th, [LOSS_NAMES[l].replace("Class-balanced", "CB") for l in losses], series)
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
    n = max(s.n for s in main)
    sub = "Higher is better. Y-axes are zoomed; the best mean in each panel is labelled."
    if n > 1:
        sub += f" Bars: mean of {n} seeds, whiskers: ±1 std."
    _title(fig, th, "Loss × head ablation (BERT-base)", sub)
    _legend(fig, th, handles, [f"BERT + {HEAD_NAMES[a]}" for a in archs], y=0.86)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def fig_encoders(plt, settings: Dict[str, Setting], th, out: Path):
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


def fig_routing(plt, settings: Dict[str, Setting], th, out: Path):
    from matplotlib.colors import LinearSegmentedColormap

    # Expert indices are arbitrary per seed (any permutation is equivalent), so routing
    # is shown for a single run rather than averaged.
    r = settings[HEADLINE].first
    gates = r["test"]["gates"]["mean_gate"]
    rows = [k for k in ["all", "clean", *LABELS] if k in gates]
    mat = np.array([gates[k] for k in rows])
    n_exp = mat.shape[1]
    cmap = LinearSegmentedColormap.from_list("seq", th["seq"])
    fig, axes = _figure(plt, th, 2.4 + 0.85 * n_exp, 4.4)
    ax = axes[0]
    fig.subplots_adjust(top=0.8, bottom=0.08, left=0.22, right=0.97)
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
    ax.set_yticks(range(len(rows)), [("all comments" if k == "all" else k) for k in rows])
    ax.tick_params(length=0, labelcolor=th["ink2"], labelsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_exp, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=th["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)
    seed = f", seed {r['seed']}" if r.get("seed") is not None else ""
    _title(
        fig,
        th,
        f"Expert routing (MoE + CB-focal{seed})",
        "Mean router weight per expert, over test comments with each label.",
    )
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def make_figures(settings: Dict[str, Setting], fig_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    fig_dir.mkdir(parents=True, exist_ok=True)
    main = [s for s in settings.values() if s.encoder == MAIN_ENCODER]
    has_pair = BASELINE in settings and HEADLINE in settings
    jobs: list = [
        ("ablation", fig_ablation, len(main) >= 2),
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
