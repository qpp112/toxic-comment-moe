"""Experiment configuration: one flat dataclass, loadable from YAML with CLI overrides."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ARCHS = ("bert", "moe")
LOSSES = ("bce", "weighted_bce", "focal", "cb_focal")
AMP_DTYPES = ("auto", "bf16", "fp16")


@dataclass
class Config:
    # --- identity -----------------------------------------------------------
    name: str = "bert_bce"
    group: Optional[str] = None  # runs that differ only by seed share a group (e.g. "bert_bce")

    # --- data ---------------------------------------------------------------
    data_dir: str = "data"
    cache_dir: Optional[str] = None  # tokenization cache; defaults to <data_dir>/cache
    train_fraction: float = 1.0  # 0.05 reproduces the course-project setting
    val_fraction: float = 0.1  # held out from train.csv for model selection + thresholds
    max_length: int = 128

    # --- model --------------------------------------------------------------
    arch: str = "bert"  # "bert" (linear head) | "moe" (mixture-of-experts head)
    encoder: str = "bert-base-uncased"  # any HF encoder name or local path
    pooling: str = "cls"  # "cls" | "mean"
    dropout: float = 0.1
    num_experts: int = 6
    expert_hidden: int = 256
    top_k: int = 2  # experts used per comment; 0 = dense (all experts)
    aux_loss_weight: float = 0.01  # load-balancing loss coefficient (MoE only)

    # --- loss ---------------------------------------------------------------
    loss: str = "bce"  # "bce" | "weighted_bce" | "focal" | "cb_focal"
    focal_gamma: float = 2.0
    focal_alpha: Optional[float] = None  # only used by plain "focal"
    cb_beta: float = 0.9999  # effective-number hyper-parameter (Cui et al., 2019)

    # --- optimisation -------------------------------------------------------
    epochs: int = 2
    batch_size: int = 32
    eval_batch_size: int = 128
    lr: float = 2e-5  # encoder learning rate
    head_lr: float = 1e-4  # classifier / MoE head learning rate
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    fp16: bool = True  # mixed precision on/off (ignored on CPU)
    amp_dtype: str = "auto"  # "auto" = bf16 on Ampere+ GPUs (A100, L4), fp16 otherwise (T4)
    selection_metric: str = "roc_auc_mean"  # validation metric used to keep the best epoch
    log_every: int = 200

    # --- misc ---------------------------------------------------------------
    seed: int = 42  # model init, dropout and batch order
    split_seed: int = 42  # data subsample + train/val split; fixed across seeds so runs are paired
    num_workers: int = 2
    output_dir: str = "results"
    save_checkpoint: bool = False
    save_predictions: bool = True

    def validate(self) -> Config:
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {self.loss!r}")
        if self.pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {self.pooling!r}")
        if not 0 < self.train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1]")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")
        if self.top_k < 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [0, num_experts]")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.amp_dtype not in AMP_DTYPES:
            raise ValueError(f"amp_dtype must be one of {AMP_DTYPES}, got {self.amp_dtype!r}")
        return self

    @property
    def group_name(self) -> str:
        return self.group or self.name

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def replace(self, **overrides: Any) -> Config:
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise KeyError(f"Unknown config keys: {sorted(unknown)}")
        return dataclasses.replace(self, **overrides).validate()


def _cast(field: dataclasses.Field, raw: str) -> Any:
    """Cast a CLI string to the declared type of ``field``."""
    if raw.lower() in ("none", "null"):
        return None
    type_str = str(field.type)
    if "bool" in type_str:
        if raw.lower() in ("1", "true", "yes", "y"):
            return True
        if raw.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"Cannot parse boolean for {field.name}: {raw!r}")
    if "int" in type_str:
        return int(raw)
    if "float" in type_str:
        return float(raw)
    return raw


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Any]:
    """Parse ``["key=value", ...]`` into a typed dict."""
    if not items:
        return {}
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in fields:
            raise KeyError(f"Unknown config key: {key!r}")
        out[key] = _cast(fields[key], raw.strip())
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path | None = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load a single-run config. ``path`` may point to a flat YAML of Config fields."""
    values: Dict[str, Any] = {}
    if path is not None:
        values.update(load_yaml(path))
    if overrides:
        values.update(overrides)
    return Config().replace(**values)


def load_ablation(
    path: str | Path,
    overrides: Optional[Dict[str, Any]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Config]:
    """Load an ablation file and expand it over random seeds.

    File format::

        base: base.yaml            # or an inline dict of Config fields
        seeds: [42, 43, 44]        # optional; omitted -> one run per entry, no suffix
        runs:
          - {name: bert_bce, arch: bert, loss: bce}
          ...

    Each run becomes ``<name>_s<seed>`` with ``group=<name>``. Runs are ordered
    seed-major (every run for the first seed, then every run for the second, ...),
    so a partially finished study is still a complete single-seed ablation. Only the
    first seed keeps ``save_checkpoint``. ``overrides`` (e.g. from the CLI) apply to
    every run; ``seeds`` overrides the file's list.
    """
    path = Path(path)
    spec = load_yaml(path)
    base = spec.get("base", {})
    if isinstance(base, str):
        base = load_yaml(path.parent / base)
    runs = spec.get("runs") or []
    if not runs:
        raise ValueError(f"No runs listed in {path}")
    seed_list = list(seeds) if seeds else spec.get("seeds")

    configs = []
    if not seed_list:
        for run in runs:
            configs.append(Config().replace(**{**base, **run, **(overrides or {})}))
    else:
        for i, seed in enumerate(seed_list):
            for run in runs:
                values = {**base, **run, **(overrides or {})}
                values.update(group=values["name"], name=f"{values['name']}_s{seed}", seed=int(seed))
                if i > 0:
                    values["save_checkpoint"] = False
                configs.append(Config().replace(**values))
    names = [c.name for c in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate run names in {path}: {names}")
    return configs


def is_complete(cfg: Config) -> bool:
    """A run is finished once its metrics.json exists (it is written last)."""
    return (cfg.run_dir / "metrics.json").exists()


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
