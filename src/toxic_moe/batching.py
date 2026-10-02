"""Tokenization (with an on-disk cache), datasets and length-aware batching."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def _cache_file(cache_dir: Path, tokenizer_name: str, max_length: int, texts: Sequence[str], tag: str) -> Path:
    digest = hashlib.sha1()
    digest.update(f"{tokenizer_name}|{max_length}|{len(texts)}".encode())
    for t in texts[:: max(1, len(texts) // 1000)]:  # cheap content fingerprint
        digest.update(t.encode("utf-8", "ignore"))
    safe = tokenizer_name.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{tag}_{safe}_{max_length}_{digest.hexdigest()[:10]}.npz"


def encode_texts(
    tokenizer,
    texts: Sequence[str],
    max_length: int,
    cache_dir: Optional[str | Path] = None,
    tag: str = "texts",
    chunk_size: int = 20_000,
) -> List[np.ndarray]:
    """Tokenize ``texts`` once (truncation, no padding) and cache the token ids.

    Padding happens per batch in :class:`Collator`, so short comments are not padded
    to ``max_length`` -- most Jigsaw comments are far shorter than 128 tokens.
    """
    texts = list(texts)
    path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_file(cache_dir, getattr(tokenizer, "name_or_path", "tok"), max_length, texts, tag)
        if path.exists():
            blob = np.load(path)
            flat, offsets = blob["flat"], blob["offsets"]
            return [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]

    ids: List[np.ndarray] = []
    for start in range(0, len(texts), chunk_size):
        enc = tokenizer(texts[start : start + chunk_size], truncation=True, max_length=max_length)
        ids.extend(np.asarray(x, dtype=np.int32) for x in enc["input_ids"])

    if path is not None:
        offsets = np.zeros(len(ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(x) for x in ids])
        flat = np.concatenate(ids) if ids else np.zeros(0, dtype=np.int32)
        tmp = path.with_name(path.stem + ".partial.npz")
        np.savez(tmp, flat=flat, offsets=offsets)
        os.replace(tmp, path)  # atomic: a crash mid-write never leaves a corrupt cache
    return ids


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------
class TokenizedDataset(Dataset):
    def __init__(self, input_ids: Sequence[np.ndarray], labels: Optional[np.ndarray] = None):
        if labels is not None and len(labels) != len(input_ids):
            raise ValueError("input_ids and labels must have the same length")
        self.input_ids = input_ids
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.input_ids)

    def lengths(self) -> np.ndarray:
        return np.fromiter((len(x) for x in self.input_ids), dtype=np.int64, count=len(self.input_ids))

    def __getitem__(self, idx: int):
        item = {"index": idx, "input_ids": self.input_ids[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class Collator:
    """Pads a batch to its longest sequence and builds the attention mask."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.as_tensor(b["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = 1
        out = {
            "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "labels" in batch[0]:
            out["labels"] = torch.as_tensor(np.stack([b["labels"] for b in batch]), dtype=torch.float32)
        return out


# ---------------------------------------------------------------------------
# Length-aware batch samplers
# ---------------------------------------------------------------------------
class LengthGroupedBatchSampler(Sampler[List[int]]):
    """Random batches of similar-length comments (less padding -> ~2x faster).

    Shuffle all indices, cut them into "mega-batches" of ``batch_size * mega``,
    sort each mega-batch by length, slice it into batches, then shuffle the batches.
    Every epoch sees every sample exactly once.
    """

    def __init__(self, lengths: np.ndarray, batch_size: int, mega: int = 50, seed: int = 0, drop_last: bool = False):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.mega = mega
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = rng.permutation(len(self.lengths))
        chunk = self.batch_size * self.mega
        batches = []
        for start in range(0, len(perm), chunk):
            idx = perm[start : start + chunk]
            idx = idx[np.argsort(-self.lengths[idx], kind="stable")]
            for b in range(0, len(idx), self.batch_size):
                batch = idx[b : b + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch.tolist())
        order = rng.permutation(len(batches))
        return iter([batches[i] for i in order])

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


class SortedBatchSampler(Sampler[List[int]]):
    """Deterministic batches sorted by length, for fast inference."""

    def __init__(self, lengths: np.ndarray, batch_size: int):
        self.order = np.argsort(-np.asarray(lengths), kind="stable")
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        for b in range(0, len(self.order), self.batch_size):
            yield self.order[b : b + self.batch_size].tolist()

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.batch_size)


def make_loader(
    dataset: TokenizedDataset,
    pad_token_id: int,
    batch_size: int,
    train: bool,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    lengths = dataset.lengths()
    sampler = (
        LengthGroupedBatchSampler(lengths, batch_size, seed=seed) if train else SortedBatchSampler(lengths, batch_size)
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=Collator(pad_token_id),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
