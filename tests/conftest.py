import json
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
    """A 2-layer, 32-dim BERT + word-level tokenizer saved to disk (no network needed).

    The tokenizer is written the way Hub checkpoints store it (tokenizer.json, vocab.txt and
    tokenizer_config.json) and read back with AutoTokenizer, which works the same under
    transformers 4 and 5. (Transformers 5 dropped the ``vocab_file`` constructor argument:
    ``BertTokenizerFast(vocab_file=...)`` silently gives a 5-token vocabulary there.)
    """
    transformers = pytest.importorskip("transformers")
    tk = pytest.importorskip("tokenizers")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ",", ".", '"', "second", "line", "with", "comma", *WORDS]
    ids = {token: i for i, token in enumerate(vocab)}
    out = tmp_path / "tiny-bert"
    out.mkdir()

    backend = tk.Tokenizer(tk.models.WordPiece(ids, unk_token="[UNK]"))
    backend.normalizer = tk.normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = tk.pre_tokenizers.BertPreTokenizer()
    backend.post_processor = tk.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", ids["[CLS]"]), ("[SEP]", ids["[SEP]"])],
    )
    backend.decoder = tk.decoders.WordPiece(prefix="##")
    backend.add_special_tokens(vocab[:5])  # already in the vocabulary: marks them special, adds no ids
    backend.save(str(out / "tokenizer.json"))
    (out / "vocab.txt").write_text("\n".join(vocab) + "\n")
    special = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    settings = {"tokenizer_class": "BertTokenizer", "do_lower_case": True, "model_max_length": 64, **special}
    (out / "tokenizer_config.json").write_text(json.dumps(settings))

    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    transformers.BertModel(config).save_pretrained(out)

    tok = transformers.AutoTokenizer.from_pretrained(out)
    assert len(tok) == len(vocab), f"the tiny tokenizer lost its vocabulary: {len(tok)} of {len(vocab)} tokens"
    assert tok.convert_ids_to_tokens(tok("You idiot")["input_ids"]) == ["[CLS]", "you", "idiot", "[SEP]"]
    return out
