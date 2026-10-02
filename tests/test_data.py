import numpy as np
import pytest

from toxic_moe import LABELS
from toxic_moe.data import (
    ensure_extracted,
    label_counts,
    load_test,
    load_train,
    stratify_keys,
    subsample,
    train_val_split,
)


def test_load_train_keeps_multiline_comments(data_dir):
    df = load_train(data_dir)
    assert len(df) == 400
    assert df["comment_text"].str.contains("\n").any()
    assert set(df[LABELS].to_numpy().ravel()) <= {0, 1}


def test_load_test_drops_unscored_rows(data_dir):
    df = load_test(data_dir)
    assert len(df) == 150  # 200 rows, every 4th labelled -1
    assert (df[LABELS].to_numpy() >= 0).all()


def test_ensure_extracted_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure_extracted(tmp_path)


def test_ensure_extracted_unzips_nested_archives(data_dir, tmp_path):
    import zipfile

    nested = tmp_path / "zipped"
    nested.mkdir()
    for name in ("train.csv", "test.csv", "test_labels.csv"):
        with zipfile.ZipFile(nested / f"{name}.zip", "w") as zf:
            zf.write(data_dir / name, arcname=name)
    with zipfile.ZipFile(nested / "competition.zip", "w") as zf:
        for name in ("train.csv", "test.csv", "test_labels.csv"):
            zf.write(nested / f"{name}.zip", arcname=f"{name}.zip")
    for name in ("train.csv", "test.csv", "test_labels.csv"):
        (nested / f"{name}.zip").unlink()
    ensure_extracted(nested)
    assert len(load_train(nested)) == 400


def test_split_is_stratified_and_disjoint(data_dir):
    df = load_train(data_dir)
    tr, va = train_val_split(df, 0.2, seed=0)
    assert len(set(tr) & set(va)) == 0
    assert len(tr) + len(va) == len(df)
    y = df[LABELS].to_numpy()
    # every label with enough positives appears in both splits at a similar rate
    for j in range(len(LABELS)):
        if y[:, j].sum() >= 10:
            assert y[va, j].sum() > 0
            assert abs(y[tr, j].mean() - y[va, j].mean()) < 0.05


def test_split_is_deterministic(data_dir):
    df = load_train(data_dir)
    a = train_val_split(df, 0.2, seed=3)
    b = train_val_split(df, 0.2, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_stratify_keys_pools_singletons():
    y = np.array([[0, 0], [0, 0], [1, 0], [1, 0], [0, 1], [1, 1]])
    keys = stratify_keys(y)
    _, counts = np.unique(keys, return_counts=True)
    assert counts.min() >= 2


def test_subsample_fraction(data_dir):
    df = load_train(data_dir)
    assert len(subsample(df, 0.5, seed=0)) == 200
    assert len(subsample(df, 1.0, seed=0)) == 400


def test_label_counts():
    pos, n = label_counts(np.array([[1, 0], [1, 1], [0, 0]]))
    assert pos.tolist() == [2, 1] and n == 3
