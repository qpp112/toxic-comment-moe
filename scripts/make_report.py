#!/usr/bin/env python
"""Aggregate results/<run>/metrics.json into tables + figures, and refresh the README.

    python scripts/make_report.py                       # results/ -> results/summary.md + docs/figures/*.png
    python scripts/make_report.py --update-readme       # also rewrite the RESULTS block in README.md
    python scripts/make_report.py --results /content/drive/MyDrive/toxic-moe/results

Only metrics.json files are read, so this runs anywhere (no GPU, no data).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from toxic_moe import LABELS  # noqa: E402

LOSS_ORDER = ["bce", "weighted_bce", "focal", "cb_focal"]
LOSS_NAMES = {"bce": "BCE", "weighted_bce": "Weighted BCE", "focal": "Focal", "cb_focal": "Class-balanced focal"}
ARCH_NAMES = {"bert": "BERT + linear head", "moe": "BERT + MoE head"}
BASELINE, HEADLINE = "bert_bce", "moe_cbfocal"

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
# loading + tables
# ---------------------------------------------------------------------------
def load_runs(results_dir: Path) -> Dict[str, dict]:
    runs = {}
    for path in sorted(results_dir.glob("*/metrics.json")):
        with open(path) as f:
            runs[path.parent.name] = json.load(f)
    return runs


def _fmt(x: Optional[float], digits: int) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def _bold_best(values: List[Optional[float]], cells: List[str]) -> List[str]:
    finite = [v for v in values if v is not None and not np.isnan(v)]
    if not finite:
        return cells
    best = max(finite)
    return [f"**{c}**" if v is not None and v == best else c for v, c in zip(values, cells)]


def sort_key(item):
    name, r = item
    return (0 if r["arch"] == "bert" else 1, LOSS_ORDER.index(r["loss"]) if r["loss"] in LOSS_ORDER else 99, name)


def ablation_table(runs: Dict[str, dict]) -> str:
    items = sorted(runs.items(), key=sort_key)
    cols = [
        ("Test ROC-AUC", lambda r: r["test"]["roc_auc_mean"], 4),
        ("Test PR-AUC", lambda r: r["test"]["pr_auc_mean"], 3),
        ("Macro-F1 @0.5", lambda r: r["test"]["f1@0.5"]["macro"], 3),
        ("Macro-F1 (val-tuned thr.)", lambda r: r["test"]["f1@tuned"]["macro"], 3),
        (
            "Rare-label F1¹",
            lambda r: float(
                np.mean(
                    [
                        r["test"]["f1@tuned"]["per_label"][LABELS.index(k)]
                        for k in ("severe_toxic", "threat", "identity_hate")
                    ]
                )
            ),
            3,
        ),
    ]
    header = "| Model | Loss | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|---|---|" + "---:|" * len(cols)
    columns = []
    for _, fn, digits in cols:
        values = [fn(r) for _, r in items]
        columns.append(_bold_best(values, [_fmt(v, digits) for v in values]))
    lines = [header, sep]
    for i, (_, r) in enumerate(items):
        row = [ARCH_NAMES.get(r["arch"], r["arch"]), LOSS_NAMES.get(r["loss"], r["loss"])]
        row += [col[i] for col in columns]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def per_label_table(runs: Dict[str, dict], names: List[str]) -> str:
    names = [n for n in names if n in runs]
    if not names:
        return ""
    header = "| Label | Test positives | " + " | ".join(f"{n} ROC-AUC | {n} PR-AUC | {n} F1" for n in names) + " |"
    sep = "|---|---:|" + "---:|---:|---:|" * len(names)
    lines = [header, sep]
    pos = runs[names[0]]["test"]["positives"]
    for j, label in enumerate(LABELS):
        row = [f"`{label}`", f"{pos[j]:,}"]
        for n in names:
            t = runs[n]["test"]
            row += [_fmt(t["roc_auc"][j], 4), _fmt(t["pr_auc"][j], 3), _fmt(t["f1@tuned"]["per_label"][j], 3)]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run_details(runs: Dict[str, dict]) -> str:
    lines = [
        "| Run | Best epoch | Val ROC-AUC per epoch | Train time / epoch (min) | Params (head) | Wall clock (min) | GPU |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for name, r in sorted(runs.items(), key=sort_key):
        hist = r.get("history", [])
        curve = " → ".join(_fmt(h["val_roc_auc_mean"], 4) for h in hist)
        per_epoch = np.mean([h["train_seconds"] for h in hist]) / 60 if hist else float("nan")
        lines.append(
            f"| `{name}` | {r.get('best_epoch', '–')} | {curve} | {_fmt(per_epoch, 1)} | "
            f"{r.get('head_parameters', 0):,} | {_fmt(r.get('wall_clock_minutes'), 1)} | "
            f"{r.get('environment', {}).get('gpu', r.get('environment', {}).get('device', '–'))} |"
        )
    return "\n".join(lines)


def summary_markdown(runs: Dict[str, dict], figure_prefix: str = "docs/figures") -> str:
    any_run = next(iter(runs.values()))
    d = any_run["data"]
    parts = [
        f"Trained on {d['train']:,} comments ({d['train_fraction']:.0%} of `train.csv` minus a "
        f"{d['val']:,}-comment validation split), evaluated on the **{d['test']:,}-comment official "
        "Kaggle test set** (rows with released labels). Thresholds for F1 are tuned per label on the "
        "validation split, never on test. Best value per column in bold.",
        "",
        ablation_table(runs),
        "",
        "¹ Mean test F1 over the three rarest labels: `severe_toxic`, `threat`, `identity_hate`.",
    ]
    if (BASELINE in runs) or (HEADLINE in runs):
        parts += ["", "**Per label: baseline vs. headline model**", "", per_label_table(runs, [BASELINE, HEADLINE])]
    figs = [
        ("ablation", "Ablation: every loss × architecture on the test set"),
        ("per_label_f1", "Per-label F1 (val-tuned thresholds), baseline vs. MoE + CB-focal"),
        ("pr_curves", "Precision–recall curves on the test set"),
        ("expert_routing", "Which experts the router uses for each label"),
    ]
    for stem, alt in figs:
        if (ROOT / figure_prefix / f"{stem}_light.png").exists():
            parts += ["", picture(figure_prefix, stem, alt)]
    parts += ["", "<details><summary>Training details</summary>", "", run_details(runs), "", "</details>"]
    return "\n".join(parts)


def picture(prefix: str, stem: str, alt: str) -> str:
    return (
        f'<picture>\n  <source media="(prefers-color-scheme: dark)" srcset="{prefix}/{stem}_dark.png">\n'
        f'  <img alt="{alt}" src="{prefix}/{stem}_light.png">\n</picture>'
    )


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


def fig_ablation(plt, runs, th, out: Path):
    metrics = [
        ("Test ROC-AUC (Kaggle metric)", lambda r: r["test"]["roc_auc_mean"], "{:.4f}"),
        ("Test PR-AUC (mean over labels)", lambda r: r["test"]["pr_auc_mean"], "{:.3f}"),
        ("Test macro-F1 (val-tuned thr.)", lambda r: r["test"]["f1@tuned"]["macro"], "{:.3f}"),
    ]
    losses = [l for l in LOSS_ORDER if any(r["loss"] == l for r in runs.values())]
    archs = [a for a in ("bert", "moe") if any(r["arch"] == a for r in runs.values())]
    fig, axes = _figure(plt, th, 12, 4.2, ncols=3)
    fig.subplots_adjust(top=0.72, bottom=0.2, left=0.05, right=0.99, wspace=0.28)
    width = 0.36
    x = np.arange(len(losses))
    handles = []
    for ax, (title, fn, fmt) in zip(axes, metrics):
        _style(ax, th)
        vals_all: list = []
        best: tuple = (-np.inf, None, None)
        for k, arch in enumerate(archs):
            vals = [
                next((fn(r) for r in runs.values() if r["arch"] == arch and r["loss"] == l), np.nan) for l in losses
            ]
            vals_all += [v for v in vals if not np.isnan(v)]
            offs = (k - (len(archs) - 1) / 2) * (width + 0.04)
            bars = ax.bar(x + offs, vals, width, color=th["series"][arch], edgecolor=th["surface"], linewidth=2)
            if ax is axes[0]:
                handles.append(bars[0])
            for xi, v in zip(x + offs, vals):
                if not np.isnan(v) and v > best[0]:
                    best = (v, xi, fmt)
        if best[1] is not None:  # label only the single best bar in each panel
            ax.text(
                best[1],
                best[0],
                best[2].format(best[0]),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=th["ink"],
                fontweight="bold",
            )
        lo, hi = min(vals_all), max(vals_all)
        pad = (hi - lo) * 0.6 + 1e-3
        ax.set_ylim(max(0, lo - pad), min(1, hi + pad))  # zoomed axis; values are labelled
        ax.set_xticks(x, [LOSS_NAMES[l].replace("Class-balanced", "CB") for l in losses], fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=th["ink"], pad=8)
    _title(
        fig,
        th,
        "Loss × architecture ablation",
        "Higher is better. Y-axes are zoomed to show differences; the best bar in each panel is labelled.",
    )
    _legend(fig, th, handles, [ARCH_NAMES[a] for a in archs], y=0.86)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def fig_per_label(plt, runs, th, out: Path):
    names = [n for n in (BASELINE, HEADLINE) if n in runs]
    fig, axes = _figure(plt, th, 9, 4.2)
    ax = axes[0]
    fig.subplots_adjust(top=0.74, bottom=0.14, left=0.07, right=0.99)
    _style(ax, th)
    x = np.arange(len(LABELS))
    width = 0.36
    handles = []
    for k, name in enumerate(names):
        r = runs[name]
        vals = r["test"]["f1@tuned"]["per_label"]
        offs = (k - (len(names) - 1) / 2) * (width + 0.04)
        bars = ax.bar(x + offs, vals, width, color=th["series"][r["arch"]], edgecolor=th["surface"], linewidth=2)
        handles.append(bars[0])
        for xi, v in zip(x + offs, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=th["ink2"])
    pos = runs[names[0]]["test"]["positives"]
    ax.set_xticks(x, [f"{l}\n({p:,} pos.)" for l, p in zip(LABELS, pos)], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1", color=th["ink2"], fontsize=9)
    labels = [f"{ARCH_NAMES[runs[n]['arch']]}, {LOSS_NAMES[runs[n]['loss']]}" for n in names]
    _title(
        fig,
        th,
        "Per-label F1 on the test set",
        "Thresholds tuned on the validation split. Labels ordered as in Kaggle.",
    )
    _legend(fig, th, handles, labels, y=0.87)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def fig_pr_curves(plt, runs, th, out: Path):
    names = [n for n in (BASELINE, HEADLINE) if n in runs and "pr_curves" in runs[n]["test"]]
    fig, axes = _figure(plt, th, 12, 6.2, ncols=3, nrows=2)
    fig.subplots_adjust(top=0.82, bottom=0.08, left=0.05, right=0.99, hspace=0.45, wspace=0.18)
    handles = []
    for j, (ax, label) in enumerate(zip(axes, LABELS)):
        _style(ax, th, ygrid=True, xgrid=True)
        for name in names:
            r = runs[name]
            c = r["test"]["pr_curves"].get(label)
            if not c:
                continue
            (line,) = ax.plot(
                c["recall"], c["precision"], color=th["series"][r["arch"]], linewidth=2, solid_capstyle="round"
            )
            if j == 0:
                handles.append(line)
        ap = "  ".join(f"{_fmt(runs[n]['test']['pr_auc'][j], 3)}" for n in names)
        ax.set_title(f"{label}   AP {ap}", loc="left", fontsize=9.5, color=th["ink"])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        if j >= 3:
            ax.set_xlabel("Recall", color=th["ink2"], fontsize=9)
        if j % 3 == 0:
            ax.set_ylabel("Precision", color=th["ink2"], fontsize=9)
    labels = [f"{ARCH_NAMES[runs[n]['arch']]}, {LOSS_NAMES[runs[n]['loss']]}" for n in names]
    _title(
        fig,
        th,
        "Precision–recall on the test set",
        "Interpolated precision at each recall. AP = average precision, listed in legend order.",
    )
    _legend(fig, th, handles, labels, y=0.905)
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def fig_routing(plt, runs, th, out: Path):
    from matplotlib.colors import LinearSegmentedColormap

    r = runs[HEADLINE]
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
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_exp, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=th["surface"], linewidth=2)
    ax.tick_params(which="minor", length=0)
    _title(
        fig, th, "Expert routing (MoE + CB-focal)", "Mean router weight per expert, over test comments with each label."
    )
    fig.savefig(out, dpi=160, facecolor=th["surface"])
    plt.close(fig)


def make_figures(runs: Dict[str, dict], fig_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for mode, th in THEMES.items():
        jobs = [("ablation", fig_ablation, len(runs) >= 2)]
        jobs.append(("per_label_f1", fig_per_label, BASELINE in runs or HEADLINE in runs))
        jobs.append(("pr_curves", fig_pr_curves, BASELINE in runs or HEADLINE in runs))
        jobs.append(("expert_routing", fig_routing, HEADLINE in runs and "gates" in runs[HEADLINE]["test"]))
        for stem, fn, ok in jobs:
            if ok:
                path = fig_dir / f"{stem}_{mode}.png"
                fn(plt, runs, th, path)
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
    print(f"Found {len(runs)} runs: {', '.join(runs)}")
    if not args.no_figures:
        for p in make_figures(runs, Path(args.figures)):
            print("wrote", p)
    fig_prefix = (
        Path(args.figures).resolve().relative_to(ROOT).as_posix()
        if Path(args.figures).resolve().is_relative_to(ROOT)
        else args.figures
    )
    block = summary_markdown(runs, fig_prefix)
    summary = Path(args.results) / "summary.md"
    if not Path(fig_prefix).is_absolute():  # summary.md lives one level down, in results/
        block_for_summary = block.replace(f"{fig_prefix}/", f"../{fig_prefix}/")
    else:
        block_for_summary = block
    summary.write_text("# Results\n\n" + block_for_summary + "\n")
    print("wrote", summary)
    if args.update_readme:
        update_readme(ROOT / "README.md", block)
        print("updated README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
