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


def _encode(body, window, mode="head"):
    from toxic_moe.data import cut_tokens

    return np.asarray([1, *cut_tokens(body, window - 2, mode), 2])  # [CLS] body [SEP]


def test_check_tokens_accepts_consistent_windows_and_flags_changes():
    diag = _load()
    rng = np.random.default_rng(0)
    bodies = [list(rng.integers(10, 99, size=n)) for n in (3, 6, 7, 12, 20, 40)]  # an 8-token window keeps 6
    base = [_encode(b, 8) for b in bodies]
    longer = [_encode(b, 16) for b in bodies]
    head_tail = [_encode(b, 8, "head_tail") for b in bodies]
    assert diag.check_tokens(base, longer, 8) == {"comments": 6, "mismatches": 0, "tails_checked": 0}
    # the tails of the 7- and 12-token comments can be checked against the 16-token window
    assert diag.check_tokens(base, head_tail, 8, "head_tail", longer, 16)["mismatches"] == 0
    assert diag.check_tokens(base, head_tail, 8, "head_tail", longer, 16)["tails_checked"] == 2

    changed = [x.copy() for x in longer]
    changed[0][1] += 1  # a short comment tokenized differently
    changed[4][2] += 1  # a long comment that starts differently
    assert diag.check_tokens(base, changed, 8)["mismatches"] == 2
    wrong_tail = [x.copy() for x in head_tail]
    wrong_tail[3][5] += 1  # the 12-token comment's tail
    assert diag.check_tokens(base, wrong_tail, 8, "head_tail", longer, 16)["mismatches"] == 1
    plain_head = [_encode(b, 8) for b in bodies]  # "head + tail" that kept only the head
    assert diag.check_tokens(base, plain_head, 8, "head_tail", longer, 16)["mismatches"] == 2


def test_token_check_reads_the_training_caches(tiny_encoder_dir, tmp_path):
    pytest.importorskip("torch")
    from transformers import AutoTokenizer

    from toxic_moe.batching import encode_texts

    diag = _load()
    tok = AutoTokenizer.from_pretrained(tiny_encoder_dir)
    words = ["hello", "thanks", "article", "edit", "page", "great", "idiot", "stupid", "kill", "hate", "you", "wiki"]
    texts = [" ".join(words[i % len(words)] for i in range(n)) for n in (2, 6, 9, 13, 30)]
    cache = tmp_path / "data" / "cache"
    for split in ("train", "test"):
        for length, mode in ((8, "head"), (16, "head"), (8, "head_tail")):
            encode_texts(tok, texts, length, cache_dir=cache, tag=split, truncation=mode)
    md, out = diag.token_check(tmp_path / "data", str(tiny_encoder_dir), window=8)
    assert "## Same tokens?" in md
    assert set(out) == {f"{s}: {x}" for s in ("train", "test") for x in ("16 tokens", "8 tokens, first + last half")}
    assert all(v["comments"] == 5 and v["mismatches"] == 0 for v in out.values())
    assert out["test: 8 tokens, first + last half"]["tails_checked"] == 2  # the 9- and 13-word comments
    assert diag.token_check(tmp_path / "nowhere", str(tiny_encoder_dir)) == ("", {})


def test_length_section_pairs_longer_inputs_by_seed(runs_and_data):
    diag = _load()
    diag.token_lengths = lambda texts, encoder: (np.array([len(t.split()) for t in texts]), "words")  # no download
    results, data_dir = runs_and_data
    test = load_test(data_dir)
    y, ids = test[LABELS].to_numpy().astype(int), test["id"].to_numpy().astype(str)
    rng = np.random.default_rng(1)
    for seed in (42, 43):
        _write_run(results, "bert_bce_len512", "bert", seed, y, ids, rng)
    out = results / "diag"
    assert diag.main(["--results", str(results), "--data", str(data_dir), "--bootstrap", "0", "--out", str(out)]) == 0
    gain = json.loads(out.with_suffix(".json").read_text())["length"]["length_runs"]["bert_bce_len512"]
    assert gain["seeds"] == [42, 43] and len(gain["d_roc_short"]) == len(gain["d_roc_long"]) == 2
    assert set(gain["d_roc_long_per_label"]) == set(LABELS)
    md = out.with_suffix(".md").read_text()
    assert "| Linear, 512 tokens | 2 |" in md and "Short comments" in md
