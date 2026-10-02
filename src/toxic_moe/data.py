"""Loading, cleaning and splitting the Jigsaw Toxic Comment data (no torch needed here)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import LABELS

KAGGLE_COMPETITION = "jigsaw-toxic-comment-classification-challenge"
REQUIRED_FILES = ("train.csv", "test.csv", "test_labels.csv")


def ensure_extracted(data_dir: str | Path) -> Path:
    """Unzip Kaggle's (possibly nested) archives in ``data_dir`` until the CSVs exist."""
    data_dir = Path(data_dir)
    for _ in range(2):  # the competition zip contains one zip per CSV
        if all((data_dir / f).exists() for f in REQUIRED_FILES):
            break
        for archive in sorted(data_dir.glob("*.zip")):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(data_dir)
    missing = [f for f in REQUIRED_FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. Download the data with "
            f"`bash scripts/download_data.sh` (Kaggle API) or place the CSVs there manually."
        )
    return data_dir


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a Jigsaw CSV.

    Uses pandas' default C parser, which handles quoted comments that span several
    lines, and fails loudly on a malformed file. (The course MoE notebook used
    ``engine="python", on_bad_lines="warn"``, which skipped lines it could not parse
    and produced a 1/20 sample of 8,219 rows instead of the expected 7,979.)
    ``keep_default_na=False`` keeps comments such as "NA" or "null" as text.
    """
    return pd.read_csv(path, dtype={"id": str, "comment_text": str}, keep_default_na=False)


def _check_columns(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")


def load_train(data_dir: str | Path) -> pd.DataFrame:
    df = read_csv(Path(data_dir) / "train.csv")
    _check_columns(df, ["id", "comment_text", *LABELS], "train.csv")
    df[LABELS] = df[LABELS].astype(np.int8)
    return df.reset_index(drop=True)


def load_test(data_dir: str | Path) -> pd.DataFrame:
    """Join Kaggle's test comments with the labels released after the competition.

    Rows labelled ``-1`` were never scored by Kaggle and are dropped, leaving the
    63,978 comments of the official private+public test set.
    """
    data_dir = Path(data_dir)
    text = read_csv(data_dir / "test.csv")
    labels = read_csv(data_dir / "test_labels.csv")
    _check_columns(text, ["id", "comment_text"], "test.csv")
    _check_columns(labels, ["id", *LABELS], "test_labels.csv")
    df = text.merge(labels, on="id", how="inner", validate="one_to_one")
    df[LABELS] = df[LABELS].astype(int)
    df = df[(df[LABELS] != -1).all(axis=1)].copy()
    df[LABELS] = df[LABELS].astype(np.int8)
    return df.reset_index(drop=True)


def stratify_keys(y: np.ndarray, min_count: int = 2) -> np.ndarray:
    """One key per label combination, with combinations rarer than ``min_count`` pooled.

    Stratifying on the full 6-bit label combination keeps the rare labels (``threat``
    is ~0.3% of comments) represented in proportion in both splits.
    """
    keys = np.array(["".join(map(str, row)) for row in y.astype(int)])
    values, counts = np.unique(keys, return_counts=True)
    rare = set(values[counts < min_count])
    if rare:
        keys = np.array([("rare" if k in rare else k) for k in keys])
        # if the pooled bucket itself is too small, fold it into the most common key
        if (keys == "rare").sum() < min_count:
            values, counts = np.unique(keys, return_counts=True)
            keys[keys == "rare"] = values[np.argmax(counts)]
    return keys


def subsample(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return df.reset_index(drop=True)
    keys = stratify_keys(df[LABELS].to_numpy())
    keep, _ = train_test_split(np.arange(len(df)), train_size=fraction, random_state=seed, stratify=keys)
    return df.iloc[np.sort(keep)].reset_index(drop=True)


def train_val_split(df: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted row indices for a label-combination-stratified train/val split."""
    keys = stratify_keys(df[LABELS].to_numpy())
    train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=val_fraction, random_state=seed, stratify=keys)
    return np.sort(train_idx), np.sort(val_idx)


def label_counts(y: np.ndarray) -> Tuple[np.ndarray, int]:
    """Positive count per label and number of samples."""
    return y.sum(axis=0).astype(np.int64), int(y.shape[0])


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Per-label positive counts and rates, for logging."""
    pos = df[LABELS].sum()
    return pd.DataFrame({"positives": pos, "rate_%": (100 * pos / len(df)).round(2)})
