"""The report script on synthetic metrics.json files (no torch, no data)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from toxic_moe import LABELS
from toxic_moe.metrics import evaluate, gate_statistics, label_gate_statistics, pr_curves, tune_thresholds

ROOT = Path(__file__).resolve().parents[1]
RATES = np.array([0.095, 0.0057, 0.058, 0.0033, 0.054, 0.011])


def _load_report_module():
    spec = importlib.util.spec_from_file_location("make_report", ROOT / "scripts" / "make_report.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_report"] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _fake_run(out, group, arch, loss, encoder, seed, skill, **extra):
    rng = np.random.default_rng(seed * 100 + len(group))
    y_val = (rng.random((1500, 6)) < RATES * 4).astype(int)
    y_test = (rng.random((3000, 6)) < RATES * 4).astype(int)
    p_val = 1 / (1 + np.exp(-(rng.normal(0, 1, y_val.shape) + skill * 3 * y_val - 2)))
    p_test = 1 / (1 + np.exp(-(rng.normal(0, 1, y_test.shape) + skill * 3 * y_test - 2)))
    thresholds = tune_thresholds(y_val, p_val)
    test = evaluate(y_test, p_test, thresholds)
    test["pr_curves"] = pr_curves(y_test, p_test)
    if arch == "moe":
        test["gates"] = gate_statistics(y_test, rng.dirichlet(np.ones(4), size=len(y_test)))
    if arch == "mmoe":
        test["gates"] = label_gate_statistics(y_test, rng.dirichlet(np.ones(4), size=(len(y_test), 6)))
    name = f"{group}_s{seed}"
    record = {
        "name": name, "group": group, "arch": arch, "loss": loss, "encoder": encoder, "seed": seed,
        "best_epoch": 2, "history": [{"epoch": 1, "train_seconds": 60.0, "val_roc_auc_mean": 0.9}],
        "val": evaluate(y_val, p_val, thresholds), "test": test,
        "data": {"train": 1000, "val": 100, "test": 3000, "train_fraction": 1.0},
        "wall_clock_minutes": 3.0, "environment": {"device": "cuda", "gpu": "fake", "precision": "bf16"},
        "head_parameters": 1_000_000, **extra,
    }  # fmt: skip
    (out / name).mkdir(parents=True)
    (out / name / "metrics.json").write_text(json.dumps(record))


@pytest.fixture
def results_dir(tmp_path):
    out = tmp_path / "results"
    specs = [("bert_bce", "bert", "bce", "bert-base-uncased", 1.0), ("moe_cbfocal", "moe", "cb_focal", "bert-base-uncased", 1.3),
             ("deberta_bce", "bert", "bce", "microsoft/deberta-v3-base", 1.2),
             ("deberta_moe_cbfocal", "moe", "cb_focal", "microsoft/deberta-v3-base", 1.5)]  # fmt: skip
    for seed in (42, 43, 44):
        for group, arch, loss, encoder, skill in specs:
            _fake_run(out, group, arch, loss, encoder, seed, skill)
    return out


def test_grouping_and_paired_delta(results_dir):
    report = _load_report_module()
    settings = report.group_runs(report.load_runs(results_dir))
    assert set(settings) == {"bert_bce", "moe_cbfocal", "deberta_bce", "deberta_moe_cbfocal"}
    assert all(s.n == 3 and s.seeds == [42, 43, 44] for s in settings.values())
    roc = report.METRICS["roc"][1]
    mean, std, n = report.paired_delta(settings["bert_bce"], settings["moe_cbfocal"], roc)
    expected = np.mean(settings["moe_cbfocal"].values(roc) - settings["bert_bce"].values(roc))
    assert n == 3 and mean == pytest.approx(expected) and mean > 0 and std >= 0


def test_group_falls_back_to_name_without_seed_suffix(tmp_path):
    report = _load_report_module()
    runs = {"bert_bce_s42": {"arch": "bert", "loss": "bce", "seed": 42}, "solo": {"arch": "moe", "loss": "bce"}}
    assert set(report.group_runs(runs)) == {"bert_bce", "solo"}


def test_summary_and_figures(results_dir, tmp_path):
    pytest.importorskip("matplotlib")
    report = _load_report_module()
    settings = report.group_runs(report.load_runs(results_dir))
    fig_dir = tmp_path / "figs"
    written = report.make_figures(settings, fig_dir)
    stems = {p.stem.rsplit("_", 1)[0] for p in written}
    assert stems == {"ablation", "encoders", "per_label_f1", "pr_curves", "expert_routing"}
    assert len(written) == 10  # light + dark
    md = report.summary_markdown(settings, "docs/figures", fig_dir)
    assert "± " in md and "Stronger encoder" in md and "paired over 3 seeds" in md
    for label in LABELS:
        assert f"`{label}`" in md


def test_update_readme_replaces_only_the_block(tmp_path):
    report = _load_report_module()
    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro\n")
    report.update_readme(readme, "new table")
    assert readme.read_text() == "intro\n<!-- RESULTS:START -->\nnew table\n<!-- RESULTS:END -->\noutro\n"


@pytest.fixture
def follow_up_dir(tmp_path):
    """Main-study baseline + MoE, the heads study (3 seeds) and the frozen-encoder probes (1 seed)."""
    out = tmp_path / "results"
    enc = "bert-base-uncased"
    for seed in (42, 43, 44):
        _fake_run(out, "bert_bce", "bert", "bce", enc, seed, 1.0)
        _fake_run(out, "moe_bce", "moe", "bce", enc, seed, 1.0)
        for group, arch, skill in [("mmoe_bce", "mmoe", 1.1), ("labelattn_bce", "label_attn", 1.2),
                                   ("mlp_bce", "mlp", 1.0), ("moe_bce_noaux", "moe", 1.0)]:  # fmt: skip
            _fake_run(out, group, arch, "bce", enc, seed, skill, study="heads", title=f"{arch} head")
    for group, arch, skill in [("probe_linear", "bert", 0.6), ("probe_linear_mean", "bert", 0.7),
                               ("probe_moe", "moe", 0.8), ("probe_mmoe", "mmoe", 0.8),
                               ("probe_labelattn", "label_attn", 0.9)]:  # fmt: skip
        _fake_run(out, group, arch, "bce", enc, 42, skill, study="probes", title=group, freeze_encoder=True)
    return out


def test_follow_up_sections_and_figures(follow_up_dir, tmp_path):
    pytest.importorskip("matplotlib")
    report = _load_report_module()
    settings = report.group_runs(report.load_runs(follow_up_dir))
    assert settings["mmoe_bce"].study == "heads" and settings["bert_bce"].study == "main"
    assert settings["probe_moe"].title == "probe_moe"
    fig_dir = tmp_path / "figs"
    stems = {p.stem.rsplit("_", 1)[0] for p in report.make_figures(settings, fig_dir)}
    assert stems == {"ablation", "heads", "probes", "mmoe_gates"}  # no moe_cbfocal runs -> no routing figure
    md = report.summary_markdown(settings, "docs/figures", fig_dir)
    assert "Follow-up: heads built around" in md and "Frozen encoder vs fine-tuned encoder" in md
    assert "Do the labels use different experts?" in md
    # the main-study table must not mix in follow-up or probe runs
    main_table = md.split("**Loss × head ablation")[1].split("**Follow-up")[0]
    assert "mmoe" not in main_table and "probe" not in main_table
    # probes are compared with their fine-tuned counterparts; the mean-pooling control has none
    probes = md.split("Frozen encoder vs fine-tuned encoder")[1]
    assert "| probe_linear_mean |" in probes and "| – |" in probes
