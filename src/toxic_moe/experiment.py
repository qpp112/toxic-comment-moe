"""End-to-end run: data -> train -> pick thresholds on val -> evaluate on the official test set."""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from . import LABELS, __version__
from .batching import TokenizedDataset, encode_texts, make_loader
from .config import Config, is_complete, save_config  # noqa: F401 (is_complete re-exported)
from .data import describe, ensure_extracted, label_counts, load_test, load_train, subsample, train_val_split
from .losses import build_loss
from .metrics import evaluate, gate_statistics, pr_curves, tune_thresholds
from .models import build_model
from .train import predict, resolve_amp_dtype, train

logger = logging.getLogger("toxic_moe")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _environment(device: torch.device) -> Dict[str, Any]:
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "toxic_moe": __version__,
        "device": device.type,
    }
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:  # pragma: no cover
        pass
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def run(cfg: Config, log=None, encoder=None) -> Dict[str, Any]:
    """Train and evaluate one configuration; write everything to ``cfg.run_dir``.

    ``encoder`` optionally replaces the pretrained encoder named by ``cfg.encoder``
    (the tokenizer is still loaded from ``cfg.encoder``); the tests use this.
    """
    log = log or (lambda msg: print(f"[{cfg.name}] {msg}", flush=True))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # we tokenize before forking workers
    cfg.validate()
    set_seed(cfg.seed)
    device = get_device()
    if device.type == "cuda":  # TF32 matmuls: near-fp32 accuracy, much faster on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    t_start = time.time()

    # ---------------- data ----------------
    data_dir = ensure_extracted(cfg.data_dir)
    train_df = subsample(load_train(data_dir), cfg.train_fraction, cfg.split_seed)
    test_df = load_test(data_dir)
    tr_idx, va_idx = train_val_split(train_df, cfg.val_fraction, cfg.split_seed)
    y_all = train_df[LABELS].to_numpy()
    y_tr, y_va, y_te = y_all[tr_idx], y_all[va_idx], test_df[LABELS].to_numpy()
    precision = {None: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[resolve_amp_dtype(cfg, device)]
    log(f"train {len(tr_idx):,} | val {len(va_idx):,} | test {len(test_df):,} | {device} {precision} | seed {cfg.seed}")
    log("train label counts:\n" + describe(train_df.iloc[tr_idx]).to_string())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder)
    cache_dir = cfg.cache_dir or str(Path(cfg.data_dir) / "cache")
    t0 = time.time()
    ids_all = encode_texts(tokenizer, train_df["comment_text"].tolist(), cfg.max_length, cache_dir, "train")
    ids_test = encode_texts(tokenizer, test_df["comment_text"].tolist(), cfg.max_length, cache_dir, "test")
    log(f"tokenized in {time.time() - t0:.0f}s")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_ds = TokenizedDataset([ids_all[i] for i in tr_idx], y_tr)
    val_ds = TokenizedDataset([ids_all[i] for i in va_idx], y_va)
    test_ds = TokenizedDataset(ids_test, y_te)
    train_loader = make_loader(train_ds, pad_id, cfg.batch_size, train=True, seed=cfg.seed, num_workers=cfg.num_workers)
    val_loader = make_loader(val_ds, pad_id, cfg.eval_batch_size, train=False, num_workers=cfg.num_workers)
    test_loader = make_loader(test_ds, pad_id, cfg.eval_batch_size, train=False, num_workers=cfg.num_workers)

    # ---------------- model + loss ----------------
    pos, total = label_counts(y_tr)
    loss_fn = build_loss(cfg.loss, pos, total, gamma=cfg.focal_gamma, alpha=cfg.focal_alpha, beta=cfg.cb_beta)
    log(f"loss: {loss_fn.extra_repr()}")
    model = build_model(cfg, num_labels=len(LABELS), encoder=encoder)

    # ---------------- train ----------------
    best_state, history, best_epoch = train(model, cfg, train_loader, val_loader, y_va, loss_fn, device, log)
    model.load_state_dict(best_state)

    # ---------------- thresholds on val, final numbers on test ----------------
    amp_dtype = resolve_amp_dtype(cfg, device)
    val_probs, _ = predict(model, val_loader, device, amp_dtype)
    thresholds = tune_thresholds(y_va, val_probs)
    t0 = time.time()
    test_probs, test_gates = predict(model, test_loader, device, amp_dtype)
    log(f"test inference: {len(test_ds):,} comments in {time.time() - t0:.0f}s")

    val_metrics = evaluate(y_va, val_probs, thresholds)
    test_metrics = evaluate(y_te, test_probs, thresholds)
    test_metrics["pr_curves"] = pr_curves(y_te, test_probs)
    if test_gates is not None:
        test_metrics["gates"] = gate_statistics(y_te, test_gates)

    results = {
        "name": cfg.name,
        "group": cfg.group_name,
        "arch": cfg.arch,
        "loss": cfg.loss,
        "encoder": cfg.encoder,
        "seed": cfg.seed,
        "split_seed": cfg.split_seed,
        "best_epoch": best_epoch,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
        "data": {
            "train": int(len(tr_idx)),
            "val": int(len(va_idx)),
            "test": int(len(test_df)),
            "train_fraction": cfg.train_fraction,
        },
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "head_parameters": int(sum(p.numel() for p in model.head.parameters())),
        "wall_clock_minutes": round((time.time() - t_start) / 60, 1),
        "environment": {**_environment(device), "precision": precision},
    }

    if cfg.save_predictions:
        arrays = {
            "val_probs": val_probs.astype(np.float16),
            "test_probs": test_probs.astype(np.float16),
            "test_ids": test_df["id"].to_numpy().astype(str),
        }
        if test_gates is not None:
            arrays["test_gates"] = test_gates.astype(np.float16)
        np.savez_compressed(run_dir / "predictions.npz", **arrays)
    if cfg.save_checkpoint:
        torch.save(
            {
                "state_dict": {k: v.half() if v.is_floating_point() else v for k, v in best_state.items()},
                "config": cfg.to_dict(),
                "thresholds": thresholds.tolist(),
                "labels": LABELS,
            },
            run_dir / "model.pt",
        )
        tokenizer.save_pretrained(run_dir / "tokenizer")
        if hasattr(model.encoder.config, "save_pretrained"):
            model.encoder.config.save_pretrained(run_dir / "encoder_config")

    # metrics.json is written last: its presence marks the run as complete
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    t = test_metrics
    log(
        f"TEST  ROC-AUC {t['roc_auc_mean']:.4f} | PR-AUC {t['pr_auc_mean']:.4f} | "
        f"macro-F1@0.5 {t['f1@0.5']['macro']:.4f} | macro-F1@tuned {t['f1@tuned']['macro']:.4f}"
    )

    del model, best_state, loss_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results
