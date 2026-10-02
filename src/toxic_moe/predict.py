"""Inference on raw text with a saved checkpoint.

python -m toxic_moe.predict --run-dir results/moe_cbfocal "you are a wonderful person" "..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from . import LABELS
from .config import Config
from .models import build_model


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
    def __call__(self, texts: Sequence[str]) -> List[Dict[str, Any]]:
        enc = self.tokenizer(
            list(texts), truncation=True, max_length=self.cfg.max_length, padding=True, return_tensors="pt"
        )
        out = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        probs = torch.sigmoid(out.logits.float()).cpu().numpy()
        gates = None if out.gates is None else out.gates.float().cpu().numpy()
        results = []
        for i, text in enumerate(texts):
            row: Dict[str, Any] = {
                "text": text,
                "probabilities": {lab: round(float(p), 4) for lab, p in zip(self.labels, probs[i])},
                "flagged": [lab for lab, p, t in zip(self.labels, probs[i], self.thresholds) if p >= t],
            }
            if gates is not None:
                row["expert_weights"] = [round(float(g), 3) for g in gates[i]]
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
