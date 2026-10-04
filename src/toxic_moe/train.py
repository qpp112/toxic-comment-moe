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
MAX_BAD_STEPS = 20  # abort a run after this many NaN/inf steps instead of training on garbage


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


def resolve_amp_dtype(cfg, device: torch.device) -> Optional[torch.dtype]:
    """Mixed-precision dtype to use, or None for full precision (fp32, TF32 matmuls on Ampere+).

    * bf16 on Ampere or newer GPUs (A100, L4, H100): same speed as fp16, no loss scaling.
    * fp16 + GradScaler on older GPUs (T4, V100).
    * DeBERTa-v3 always runs in full precision unless ``amp_dtype`` is set explicitly:
      it produced NaN losses within its first few hundred steps under bf16 autocast, and it is known
      to overflow in fp16.
    """
    if not cfg.fp16 or device.type != "cuda":
        return None
    if cfg.amp_dtype == "bf16":
        return torch.bfloat16
    if cfg.amp_dtype == "fp16":
        return torch.float16
    if "deberta" in str(cfg.encoder).lower():
        return None
    if torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _autocast(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _grad_scaler(enabled: bool):
    try:  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def predict(
    model, loader: DataLoader, device: torch.device, amp_dtype: Optional[torch.dtype] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return sigmoid probabilities (and MoE gates) in the dataset's original order."""
    model.eval()
    n = len(loader.dataset)
    probs: Optional[np.ndarray] = None
    gates: Optional[np.ndarray] = None
    for batch in loader:
        idx = batch["index"].numpy()
        with _autocast(device, amp_dtype):
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
    amp_dtype = resolve_amp_dtype(cfg, device)
    model.to(device)
    loss_fn.to(device)
    optimizer = build_optimizer(model, cfg.lr, cfg.head_lr, cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = linear_warmup_decay(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    scaler = _grad_scaler(amp_dtype == torch.float16)  # loss scaling is only needed for fp16

    best_score, best_state, best_epoch, history = -math.inf, None, 0, []
    bad_steps = 0

    def bad_step(epoch: int, step: int, what: str) -> None:
        nonlocal bad_steps
        bad_steps += 1
        optimizer.zero_grad(set_to_none=True)  # never apply a NaN/inf update
        if bad_steps == 1:
            log(f"WARNING: non-finite {what} at epoch {epoch + 1} step {step + 1}; skipping that update")
        if bad_steps >= MAX_BAD_STEPS:
            precision = {None: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}[amp_dtype]
            raise FloatingPointError(
                f"{cfg.name}: {bad_steps} training steps had a NaN/inf {what} (precision {precision}). "
                "Stopping this run early instead of training on garbage. "
                "Try full precision with --set fp16=false, or a lower learning rate."
            )

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
            with _autocast(device, amp_dtype):
                out = model(input_ids, attention_mask)
            task_loss = loss_fn(out.logits.float(), labels)
            if not torch.isfinite(task_loss):
                bad_step(epoch, step, "loss")
                continue
            loss = task_loss
            if out.aux_loss is not None and cfg.aux_loss_weight > 0:
                loss = loss + cfg.aux_loss_weight * out.aux_loss.float()
                running_aux += out.aux_loss.detach().item()
            scaler.scale(loss / cfg.grad_accum_steps).backward()

            is_update = (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(train_loader)
            if is_update:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                if not scaler.is_enabled() and not torch.isfinite(grad_norm):
                    # with fp16 the GradScaler already skips inf/NaN steps and lowers its scale
                    bad_step(epoch, step, "gradient")
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running += task_loss.detach().item()
            seen += 1
            if cfg.log_every and (step + 1) % cfg.log_every == 0:
                rate = seen / max(time.time() - t0, 1e-9)
                msg = f"epoch {epoch + 1} step {step + 1}/{len(train_loader)} loss {running / seen:.4f}"
                if running_aux:
                    msg += f" aux {running_aux / seen:.3f}"
                log(msg + f" ({rate:.1f} it/s)")

        train_time = time.time() - t0
        val_probs, _ = predict(model, val_loader, device, amp_dtype)
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
