"""Inference on raw text with a saved checkpoint.

python -m toxic_moe.predict --run-dir results/moe_cbfocal_s42 "you are a wonderful person" "..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

from . import LABELS
from .config import Config
from .models import build_model


def word_attention(tokens: Sequence[str], weights: np.ndarray, skip: Set[str]) -> List[Tuple[str, float]]:
    """Merge sub-word pieces into words, summing their attention; drop special tokens.

    Handles WordPiece (BERT: ``kill ##ing``) and SentencePiece (DeBERTa-v3: ``▁kill ing``).
    """
    sentencepiece = any(t.startswith("\u2581") for t in tokens)
    words: List[Tuple[str, float]] = []
    for tok, w in zip(tokens, weights):
        if tok in skip:
            continue
        if sentencepiece:
            starts_word, piece = tok.startswith("\u2581"), tok.lstrip("\u2581")
        else:
            starts_word, piece = not tok.startswith("##"), tok[2:] if tok.startswith("##") else tok
        if starts_word or not words:
            words.append((piece, float(w)))
        else:
            word, total = words[-1]
            words[-1] = (word + piece, total + float(w))
    return [(word, total) for word, total in words if word]


class ToxicityPredictor:
    def __init__(self, run_dir: str | Path, device: str | None = None):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        run_dir = Path(run_dir)
        ckpt = torch.load(run_dir / "model.pt", map_location="cpu")
        self.cfg = Config(**ckpt["config"])
        self.labels: List[str] = ckpt.get("labels", LABELS)
        self.thresholds = np.asarray(ckpt.get("thresholds", [0.5] * len(self.labels)))
        self.tokenizer = AutoTokenizer.from_pretrained(run_dir / "tokenizer")
        # build the encoder skeleton from its config; weights come from the checkpoint
        config_src = run_dir / "encoder_config"
        encoder_config = AutoConfig.from_pretrained(config_src if config_src.exists() else self.cfg.encoder)
        encoder = AutoModel.from_config(encoder_config)
        self.model = build_model(self.cfg, num_labels=len(self.labels), encoder=encoder)
        self.model.load_state_dict(
            {k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def __call__(self, texts: Sequence[str], top_words: int = 5) -> List[Dict[str, Any]]:
        """Probabilities and flagged labels per text, plus what the head did:

        * MoE head: ``expert_weights``, the router's weight on each expert.
        * Multi-gate head: ``expert_weights`` per label.
        * Label-attention head: ``evidence``, the ``top_words`` words each label attended to most.
        """
        enc = self.tokenizer(
            list(texts), truncation=True, max_length=self.cfg.max_length, padding=True, return_tensors="pt"
        )
        out = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        probs = torch.sigmoid(out.logits.float()).cpu().numpy()
        gates = None if out.gates is None else out.gates.float().cpu().numpy()
        attention = None if out.attention is None else out.attention.float().cpu().numpy()
        skip = set(self.tokenizer.all_special_tokens)
        results = []
        for i, text in enumerate(texts):
            row: Dict[str, Any] = {
                "text": text,
                "probabilities": {lab: round(float(p), 4) for lab, p in zip(self.labels, probs[i])},
                "flagged": [lab for lab, p, t in zip(self.labels, probs[i], self.thresholds) if p >= t],
            }
            if gates is not None and gates.ndim == 2:
                row["expert_weights"] = [round(float(g), 3) for g in gates[i]]
            elif gates is not None:
                row["expert_weights"] = {
                    lab: [round(float(g), 3) for g in gates[i, j]] for j, lab in enumerate(self.labels)
                }
            if attention is not None:
                n_tok = int(enc["attention_mask"][i].sum())
                tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][i][:n_tok].tolist())
                row["evidence"] = {}
                for j, lab in enumerate(self.labels):
                    words = word_attention(tokens, attention[i, j, :n_tok], skip)
                    top = sorted(words, key=lambda w: -w[1])[:top_words]
                    row["evidence"][lab] = [(w, round(a, 3)) for w, a in top]
            results.append(row)
        return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="results/<run> folder containing model.pt")
    parser.add_argument("texts", nargs="+")
    args = parser.parse_args(argv)
    predictor = ToxicityPredictor(args.run_dir)
    print(json.dumps(predictor(args.texts), indent=2))


if __name__ == "__main__":
    main()
