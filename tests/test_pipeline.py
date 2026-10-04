"""End-to-end smoke test: the full training/evaluation pipeline on a tiny model, on CPU."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from toxic_moe import LABELS  # noqa: E402
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


@pytest.mark.parametrize(
    "arch,loss,freeze",
    [
        ("bert", "bce", False),
        ("moe", "cb_focal", False),
        ("mmoe", "bce", False),
        ("label_attn", "bce", False),
        ("mlp", "bce", True),
    ],
)
def test_run_end_to_end(arch, loss, freeze, data_dir, tiny_encoder_dir, tmp_path):
    cfg = Config(
        name=f"{arch}_{loss}",
        data_dir=str(data_dir),
        encoder=str(tiny_encoder_dir),
        arch=arch,
        loss=loss,
        freeze_encoder=freeze,
        epochs=2,
        batch_size=16,
        eval_batch_size=32,
        lr=1e-3,
        head_lr=1e-3,
        max_length=32,
        num_workers=0,
        num_experts=3,
        expert_hidden=16,
        mlp_hidden=16,
        attn_hidden=8,
        top_k=2,
        log_every=0,
        output_dir=str(tmp_path / "results"),
        save_checkpoint=(arch != "bert"),
        fp16=True,
    )
    results = run(cfg, log=lambda m: None)
    assert is_complete(cfg)
    saved = json.loads((cfg.run_dir / "metrics.json").read_text())
    assert saved["test"]["n"] == 150 and saved["data"]["test"] == 150
    assert (
        saved["group"] == cfg.name
        and saved["seed"] == 42
        and saved["freeze_encoder"] is freeze
        and saved["environment"]["precision"] in ("fp32", "fp16", "bf16")
    )
    assert 0.0 <= saved["test"]["roc_auc_mean"] <= 1.0
    assert len(saved["history"]) == 2 and saved["best_epoch"] in (1, 2)
    assert len(saved["test"]["thresholds"]) == 6
    assert (cfg.run_dir / "predictions.npz").exists()
    gates = saved["test"].get("gates")
    if arch == "moe":
        assert len(gates["top1_share"]) == 3 and 0 <= gates["label_divergence"] <= 1
    elif arch == "mmoe":
        assert set(gates["label_gates_all"]) == set(LABELS) and 0 <= gates["within_comment_divergence"] <= 1
        assert np.load(cfg.run_dir / "predictions.npz")["test_gates"].shape == (150, 6, 3)
    else:
        assert gates is None
    if cfg.save_checkpoint:
        from toxic_moe.predict import ToxicityPredictor

        pred = ToxicityPredictor(cfg.run_dir, device="cpu")
        out = pred(["you idiot", "thanks for the edit"])
        assert set(out[0]["probabilities"]) == set(results["test"]["labels"])
        if arch == "moe":
            assert len(out[0]["expert_weights"]) == 3
        if arch == "mmoe":
            assert set(out[0]["expert_weights"]) == set(LABELS) and len(out[0]["expert_weights"]["threat"]) == 3
        if arch == "label_attn":
            words = [w for w, _ in out[0]["evidence"]["toxic"]]
            assert sorted(words) == ["idiot", "you"]  # special tokens dropped


class BagOfEmbeddings(torch.nn.Module):
    """A trivial encoder (token embeddings, no attention).

    Training a randomly initialised transformer on 360 examples is itself unreliable, so
    this test swaps in an encoder that is guaranteed to be learnable. Everything else --
    data loading, tokenization, batching, the MoE head, the training loop, prediction
    re-ordering and evaluation -- is the real pipeline, so any misalignment between
    predictions and labels would show up as chance-level ROC-AUC.
    """

    def __init__(self, vocab_size: int, hidden: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.embed = torch.nn.Embedding(vocab_size, hidden)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


@pytest.mark.parametrize("arch", ["moe", "mmoe", "label_attn"])
def test_pipeline_learns_signal(arch, data_dir, tiny_encoder_dir, tmp_path):
    """On the synthetic data (a bag-of-words baseline scores ~0.99), the model must clearly learn."""
    vocab_size = len(transformers.AutoTokenizer.from_pretrained(tiny_encoder_dir))
    cfg = Config(
        name=f"learn_{arch}",
        data_dir=str(data_dir),
        encoder=str(tiny_encoder_dir),  # tokenizer only
        arch=arch,
        loss="bce",
        pooling="mean",
        epochs=8,
        batch_size=16,
        lr=1e-2,
        head_lr=1e-2,
        max_length=32,
        num_workers=0,
        num_experts=3,
        attn_hidden=16,
        log_every=0,
        output_dir=str(tmp_path / "r"),
        save_predictions=False,
    )
    torch.manual_seed(0)
    results = run(cfg, log=lambda m: None, encoder=BagOfEmbeddings(vocab_size))
    history = [round(h["val_roc_auc_mean"], 3) for h in results["history"]]
    # severe_toxic and obscene are coin flips within their group in the synthetic data (and have
    # only a handful of test positives), so judge the four labels that words fully determine
    clean = [LABELS.index(k) for k in ("toxic", "threat", "insult", "identity_hate")]
    roc = float(np.mean([results["test"]["roc_auc"][j] for j in clean]))
    assert roc > 0.85, f"test ROC-AUC {roc:.3f} on the clean labels, val mean per epoch {history}"


class NaNEncoder(BagOfEmbeddings):
    """Produces NaN activations, like a numerically unstable encoder."""

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids) * float("nan"))


def test_non_finite_training_stops_early(data_dir, tiny_encoder_dir, tmp_path):
    vocab_size = len(transformers.AutoTokenizer.from_pretrained(tiny_encoder_dir))
    cfg = Config(
        name="nan",
        data_dir=str(data_dir),
        encoder=str(tiny_encoder_dir),
        arch="moe",
        epochs=1,
        batch_size=8,
        max_length=32,
        num_workers=0,
        num_experts=3,
        log_every=0,
        output_dir=str(tmp_path / "r"),
    )
    with pytest.raises(FloatingPointError):
        run(cfg, log=lambda m: None, encoder=NaNEncoder(vocab_size))
    assert not is_complete(cfg)  # a failed run is retried next time, never reported


def test_deberta_defaults_to_full_precision():
    from toxic_moe.train import resolve_amp_dtype

    gpu = torch.device("cuda")  # only the device *type* is inspected on these code paths
    deberta = Config(encoder="microsoft/deberta-v3-base")
    assert resolve_amp_dtype(deberta, gpu) is None
    assert resolve_amp_dtype(deberta.replace(amp_dtype="bf16"), gpu) is torch.bfloat16  # explicit opt-in
    assert resolve_amp_dtype(Config(fp16=False), gpu) is None
    assert resolve_amp_dtype(Config(), torch.device("cpu")) is None
