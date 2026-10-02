"""End-to-end smoke test: the full training/evaluation pipeline on a tiny model, on CPU."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from toxic_moe.batching import (  # noqa: E402
    Collator,
    LengthGroupedBatchSampler,
    SortedBatchSampler,
    TokenizedDataset,
    encode_texts,
)
from toxic_moe.config import Config  # noqa: E402
from toxic_moe.experiment import is_complete, run  # noqa: E402


def test_length_grouped_sampler_covers_every_index_once():
    lengths = np.random.default_rng(0).integers(1, 100, size=1003)
    sampler = LengthGroupedBatchSampler(lengths, batch_size=16, mega=4, seed=1)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(1003))
    assert len(batches) == len(sampler)
    sampler.set_epoch(1)
    assert [i for b in sampler for i in b] != flat  # reshuffled next epoch


def test_sorted_sampler_and_collator():
    ids = [np.arange(n, dtype=np.int32) + 1 for n in (3, 7, 5)]
    ds = TokenizedDataset(ids, np.zeros((3, 6)))
    batches = list(SortedBatchSampler(ds.lengths(), batch_size=2))
    assert batches[0] == [1, 2]
    batch = Collator(pad_token_id=0)([ds[i] for i in batches[0]])
    assert batch["input_ids"].shape == (2, 7)
    assert batch["attention_mask"].sum().item() == 12
    assert batch["labels"].shape == (2, 6)


def test_tokenization_cache_round_trip(tiny_encoder_dir, tmp_path):
    tok = transformers.AutoTokenizer.from_pretrained(tiny_encoder_dir)
    texts = ["hello you", "kill people please", "thanks"]
    first = encode_texts(tok, texts, 16, cache_dir=tmp_path / "cache", tag="t")
    second = encode_texts(tok, texts, 16, cache_dir=tmp_path / "cache", tag="t")
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


@pytest.mark.parametrize("arch,loss", [("bert", "bce"), ("moe", "cb_focal")])
def test_run_end_to_end(arch, loss, data_dir, tiny_encoder_dir, tmp_path):
    cfg = Config(
        name=f"{arch}_{loss}",
        data_dir=str(data_dir),
        encoder=str(tiny_encoder_dir),
        arch=arch,
        loss=loss,
        epochs=2,
        batch_size=16,
        eval_batch_size=32,
        lr=1e-3,
        head_lr=1e-3,
        max_length=32,
        num_workers=0,
        num_experts=3,
        expert_hidden=16,
        top_k=2,
        log_every=0,
        output_dir=str(tmp_path / "results"),
        save_checkpoint=(arch == "moe"),
        fp16=True,
    )
    results = run(cfg, log=lambda m: None)
    assert is_complete(cfg)
    saved = json.loads((cfg.run_dir / "metrics.json").read_text())
    assert saved["test"]["n"] == 150 and saved["data"]["test"] == 150
    assert (
        saved["group"] == cfg.name
        and saved["seed"] == 42
        and saved["environment"]["precision"] in ("fp32", "fp16", "bf16")
    )
    assert 0.0 <= saved["test"]["roc_auc_mean"] <= 1.0
    assert len(saved["history"]) == 2 and saved["best_epoch"] in (1, 2)
    assert len(saved["test"]["thresholds"]) == 6
    assert (cfg.run_dir / "predictions.npz").exists()
    if arch == "moe":
        assert "gates" in saved["test"] and len(saved["test"]["gates"]["top1_share"]) == 3
        from toxic_moe.predict import ToxicityPredictor

        pred = ToxicityPredictor(cfg.run_dir, device="cpu")
        out = pred(["you idiot", "thanks for the edit"])
        assert set(out[0]["probabilities"]) == set(results["test"]["labels"])
        assert len(out[0]["expert_weights"]) == 3


def test_training_learns_signal(data_dir, tiny_encoder_dir, tmp_path):
    """On the synthetic data, the tiny model should clearly beat chance."""
    cfg = Config(
        name="learn",
        data_dir=str(data_dir),
        encoder=str(tiny_encoder_dir),
        arch="moe",
        loss="bce",
        pooling="mean",
        epochs=6,
        batch_size=16,
        lr=2e-3,
        head_lr=2e-3,
        max_length=32,
        num_workers=0,
        num_experts=3,
        log_every=0,
        output_dir=str(tmp_path / "r"),
        save_predictions=False,
    )
    results = run(cfg, log=lambda m: None)
    assert results["test"]["roc_auc_mean"] > 0.75
