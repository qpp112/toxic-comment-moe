"""Training loop: mixed precision, linear warm-up/decay, best-epoch checkpointing."""

from __future__ import annotations

import contextlib
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import evaluate

NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")


def build_optimizer(model, lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    encoder_params, head_params = model.param_groups()
    groups = []
    for params, group_lr in ((encoder_params, lr), (head_params, head_lr)):
        decay = [p for n, p in params if p.requires_grad and not any(k in n for k in NO_DECAY)]
        no_decay = [p for n, p in params if p.requires_grad and any(k in n for k in NO_DECAY)]
        groups.append({"params": decay, "lr": group_lr, "weight_decay": weight_decay})
        groups.append({"params": no_decay, "lr": group_lr, "weight_decay": 0.0})
    return torch.optim.AdamW([g for g in groups if g["params"]])


def linear_warmup_decay(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _grad_scaler(enabled: bool):
    try:  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def predict(
    model, loader: DataLoader, device: torch.device, fp16: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return sigmoid probabilities (and MoE gates) in the dataset's original order."""
    model.eval()
    n = len(loader.dataset)
    probs: Optional[np.ndarray] = None
    gates: Optional[np.ndarray] = None
    for batch in loader:
        idx = batch["index"].numpy()
        with _autocast(device, fp16):
            out = model(
                batch["input_ids"].to(device, non_blocking=True), batch["attention_mask"].to(device, non_blocking=True)
            )
        p = torch.sigmoid(out.logits.float()).cpu().numpy()
        if probs is None:
            probs = np.zeros((n, p.shape[1]), dtype=np.float32)
        probs[idx] = p
        if out.gates is not None:
            g = out.gates.float().cpu().numpy()
            if gates is None:
                gates = np.zeros((n, g.shape[1]), dtype=np.float32)
            gates[idx] = g
    assert probs is not None, "empty loader"
    return probs, gates


def train(
    model,
    cfg,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val: np.ndarray,
    loss_fn,
    device: torch.device,
    log: Callable[[str], None] = print,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, float]], int]:
    """Train for ``cfg.epochs`` epochs; return (best state_dict on CPU, history, best epoch).

    The best epoch is chosen on the validation split by ``cfg.selection_metric``.
    Its weights are *deep-copied* to CPU -- the course notebook used
    ``model.state_dict().copy()``, a shallow copy whose tensors kept changing, so its
    "best model" was silently always the last epoch.
    """
    use_amp = bool(cfg.fp16 and device.type == "cuda")
    model.to(device)
    loss_fn.to(device)
    optimizer = build_optimizer(model, cfg.lr, cfg.head_lr, cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = linear_warmup_decay(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    scaler = _grad_scaler(use_amp)

    best_score, best_state, best_epoch, history = -math.inf, None, 0, []
    for epoch in range(cfg.epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        model.train()
        t0, running, running_aux, seen = time.time(), 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with _autocast(device, use_amp):
                out = model(input_ids, attention_mask)
            task_loss = loss_fn(out.logits.float(), labels)
            loss = task_loss
            if out.aux_loss is not None and cfg.aux_loss_weight > 0:
                loss = loss + cfg.aux_loss_weight * out.aux_loss.float()
                running_aux += float(out.aux_loss)
            scaler.scale(loss / cfg.grad_accum_steps).backward()

            is_update = (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(train_loader)
            if is_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running += float(task_loss)
            seen += 1
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                rate = seen / max(time.time() - t0, 1e-9)
                msg = f"epoch {epoch + 1} step {step + 1}/{len(train_loader)} loss {running / seen:.4f}"
                if running_aux:
                    msg += f" aux {running_aux / seen:.3f}"
                log(msg + f" ({rate:.1f} it/s)")

        train_time = time.time() - t0
        val_probs, _ = predict(model, val_loader, device, use_amp)
        val = evaluate(y_val, val_probs)
        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(seen, 1),
            "train_aux_loss": running_aux / max(seen, 1) if running_aux else None,
            "train_seconds": round(train_time, 1),
            "val_roc_auc_mean": val["roc_auc_mean"],
            "val_pr_auc_mean": val["pr_auc_mean"],
            "val_f1_macro@0.5": val["f1@0.5"]["macro"],
        }
        history.append(record)
        score = val[cfg.selection_metric] if cfg.selection_metric in val else val["roc_auc_mean"]
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch + 1
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        log(
            f"epoch {epoch + 1}: train_loss {record['train_loss']:.4f} | val ROC-AUC {val['roc_auc_mean']:.4f} "
            f"| val PR-AUC {val['pr_auc_mean']:.4f} | val macro-F1@0.5 {val['f1@0.5']['macro']:.4f}"
            f" | {train_time / 60:.1f} min{'  <- best' if improved else ''}"
        )

    assert best_state is not None
    return best_state, history, best_epoch
