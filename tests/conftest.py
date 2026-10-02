import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxic_moe import LABELS  # noqa: E402

WORDS = [
    "hello",
    "thanks",
    "article",
    "edit",
    "page",
    "great",
    "idiot",
    "stupid",
    "kill",
    "hate",
    "you",
    "the",
    "is",
    "a",
    "wiki",
    "source",
    "please",
    "dumb",
    "die",
    "people",
]


def make_split(n: int, seed: int, with_ids: str = "tr") -> pd.DataFrame:
    """Synthetic Jigsaw-shaped data: toxic words drive the labels, so a model can learn them."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        toks = list(rng.choice(WORDS[:6] + WORDS[10:16], size=rng.integers(3, 15)))
        labels = dict.fromkeys(LABELS, 0)
        r = rng.random()
        if r < 0.25:
            toks += ["idiot", "stupid"]
            labels.update(toxic=1, insult=1)
        elif r < 0.35:
            toks += ["kill", "die"]
            labels.update(toxic=1, threat=1, severe_toxic=int(rng.random() < 0.5))
        elif r < 0.45:
            toks += ["hate", "people"]
            labels.update(toxic=1, identity_hate=1, obscene=int(rng.random() < 0.5))
        rng.shuffle(toks)
        text = " ".join(toks)
        if i % 7 == 0:
            text = f'"{text}"\nsecond line, with a comma'  # quoted multi-line comment
        rows.append({"id": f"{with_ids}{i:05d}", "comment_text": text, **labels})
    return pd.DataFrame(rows)


@pytest.fixture
def data_dir(tmp_path):
    """A tiny but complete Kaggle-style data folder (train.csv, test.csv, test_labels.csv)."""
    d = tmp_path / "data"
    d.mkdir()
    make_split(400, 0, "tr").to_csv(d / "train.csv", index=False)
    test = make_split(200, 1, "te")
    test[["id", "comment_text"]].to_csv(d / "test.csv", index=False)
    labels = test[["id", *LABELS]].copy()
    labels.loc[labels.index % 4 == 0, LABELS] = -1  # unscored rows, as in the real test_labels.csv
    labels.to_csv(d / "test_labels.csv", index=False)
    return d


@pytest.fixture
def tiny_encoder_dir(tmp_path):
    """A 2-layer, 32-dim BERT + word-level tokenizer saved to disk (no network needed)."""
    transformers = pytest.importorskip("transformers")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ",", ".", '"', "second", "line", "with", "comma", *WORDS]
    out = tmp_path / "tiny-bert"
    out.mkdir()
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tok = transformers.BertTokenizerFast(vocab_file=str(out / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(out)
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    transformers.BertModel(config).save_pretrained(out)
    return out
