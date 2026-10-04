"""scripts/diagnose.py on synthetic runs (no torch needed)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from toxic_moe import LABELS
from toxic_moe.data import load_test
from toxic_moe.metrics import evaluate, gate_statistics, pr_curves, tune_thresholds

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("diagnose", ROOT / "scripts" / "diagnose.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["diagnose"] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _write_run(out, group, arch, seed, y, ids, rng):
    """A run whose predictions are noisy copies of the labels; MoE runs route toxic comments to experts 0-1."""
    logits = 4 * y - 3 + rng.normal(0, 1.2, y.shape)
    probs = 1 / (1 + np.exp(-logits))
    probs[y[:, 0] == 0, 0] *= 0.02  # plenty of confidently clean comments
    thresholds = tune_thresholds(y, probs)
    test = evaluate(y, probs, thresholds)
    test["pr_curves"] = pr_curves(y, probs)
    arrays = {"val_probs": probs[:20].astype(np.float16), "test_probs": probs.astype(np.float16), "test_ids": ids}
    if arch == "moe":
        gates = rng.dirichlet(np.ones(4), size=len(y))
        gates[y[:, 0] == 1, :2] += 3
        gates /= gates.sum(1, keepdims=True)
        test["gates"] = gate_statistics(y, gates)
        arrays["test_gates"] = gates.astype(np.float16)
    name = f"{group}_s{seed}"
    (out / name).mkdir(parents=True)
    record = {
        "name": name, "group": group, "arch": arch, "loss": "bce", "encoder": "bert-base-uncased", "seed": seed,
        "history": [{"epoch": 1, "train_aux_loss": 1.01 if arch == "moe" else None}], "test": test, "val": test,
    }  # fmt: skip
    (out / name / "metrics.json").write_text(json.dumps(record))
    (out / name / "config.yaml").write_text("max_length: 8\n")
    np.savez_compressed(out / name / "predictions.npz", **arrays)


@pytest.fixture
def runs_and_data(data_dir, tmp_path):
    test = load_test(data_dir)
    y = test[LABELS].to_numpy().astype(int)
    ids = test["id"].to_numpy().astype(str)
    rng = np.random.default_rng(0)
    out = tmp_path / "results"
    for seed in (42, 43):
        _write_run(out, "bert_bce", "bert", seed, y, ids, rng)
        _write_run(out, "moe_bce", "moe", seed, y, ids, rng)
    return out, data_dir


def test_label_free_sections(runs_and_data):
    diag = _load()
    results, _ = runs_and_data
    assert diag.main(["--results", str(results)]) == 0
    md = (results / "diagnosis.md").read_text()
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "### Is anything routed"):
        assert heading in md
    assert "## 5." not in md  # the rest needs labels
    report = json.loads((results / "diagnosis.json").read_text())
    assert set(report) >= {"routing", "misrouting", "agreement", "expert_offsets", "thresholds", "val_test_gap"}
    assert report["misrouting"]["moe_bce_s42"]["linear_sure_toxic"] >= 0
    linear = report["agreement"]["Linear vs Linear"]["toxic"]
    assert 0 <= linear["top_overlap"] <= 1


def test_label_based_sections(runs_and_data):
    diag = _load()
    diag.token_lengths = lambda texts, encoder: (np.array([len(t.split()) for t in texts]), "words")  # no download
    results, data_dir = runs_and_data
    out = results / "diag"
    assert diag.main(["--results", str(results), "--data", str(data_dir), "--bootstrap", "5", "--out", str(out)]) == 0
    md = out.with_suffix(".md").read_text()
    for k in range(1, 12):
        assert f"## {k}." in md
    report = json.loads(out.with_suffix(".json").read_text())
    ens = report["ensembles"]["seed_ensembles"]["bert_bce"]
    assert ens["n"] == 2 and 0.5 < ens["single_roc"] <= 1 and 0.5 < ens["ensemble_roc"] <= 1
    assert set(report["ensembles"]["pairs"]) == {"bert_bce", "moe_bce"}
    assert set(report["routing_recall"]) == {"moe_bce_s42", "moe_bce_s43"}
    assert set(report["bootstrap"]) == set(LABELS)
    assert report["labels"]["threat"]["p_toxic_given_label"] == pytest.approx(1.0)  # the synthetic data's rule


def test_oracle_f1_on_a_perfect_curve():
    diag = _load()
    assert diag.oracle_f1({"recall": [0.0, 0.5, 1.0], "precision": [1.0, 1.0, 1.0]}) == pytest.approx(1.0)
    assert diag.oracle_f1({"recall": [0.0, 1.0], "precision": [0.5, 0.5]}) == pytest.approx(2 / 3)
