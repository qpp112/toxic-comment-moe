import numpy as np
import pytest

from toxic_moe.metrics import evaluate, gate_statistics, pr_curves, tune_thresholds


def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 6)) < [0.1, 0.01, 0.05, 0.005, 0.05, 0.01]).astype(int)
    p = np.clip(0.6 * y + rng.normal(0.2, 0.15, size=y.shape), 0, 1)
    return y, p


def test_perfect_scores():
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    m = evaluate(y, y.astype(float), thresholds=[0.5, 0.5], labels=["a", "b"])
    assert m["roc_auc_mean"] == pytest.approx(1.0)
    assert m["pr_auc_mean"] == pytest.approx(1.0)
    assert m["f1@0.5"]["macro"] == pytest.approx(1.0)
    assert m["f1@tuned"]["macro"] == pytest.approx(1.0)


def test_label_without_positives_is_nan_not_crash():
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    p = np.array([[0.9, 0.1], [0.2, 0.2], [0.8, 0.3], [0.1, 0.1]])
    m = evaluate(y, p, labels=["a", "b"])
    assert np.isnan(m["roc_auc"][1])
    assert m["roc_auc_mean"] == pytest.approx(1.0)  # nan-mean over defined labels


def test_tuned_thresholds_beat_or_match_default_on_same_data():
    y, p = _toy()
    t = tune_thresholds(y, p)
    m = evaluate(y, p, thresholds=t)
    assert m["f1@tuned"]["macro"] >= m["f1@0.5"]["macro"] - 1e-12
    assert all(0 <= x <= 1 for x in t)


def test_tuned_threshold_is_optimal_for_single_label():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.2).astype(int)
    p = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
    t = tune_thresholds(y[:, None], p[:, None])[0]
    from sklearn.metrics import f1_score

    best_grid = max(f1_score(y, (p >= c).astype(int), zero_division=0) for c in np.unique(p))
    assert f1_score(y, (p >= t).astype(int)) == pytest.approx(best_grid)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate(np.zeros((3, 6)), np.zeros((3, 5)))


def test_pr_curves_are_monotone_non_increasing():
    y, p = _toy()
    for curve in pr_curves(y, p).values():
        prec = np.array(curve["precision"])
        assert np.all(np.diff(prec) <= 1e-9)


def test_gate_statistics_rows_sum_to_one():
    y, _ = _toy(500)
    rng = np.random.default_rng(0)
    g = rng.dirichlet(np.ones(4), size=500)
    stats = gate_statistics(y, g)
    for row in stats["mean_gate"].values():
        assert sum(row) == pytest.approx(1.0, abs=1e-3)
    assert sum(stats["top1_share"]) == pytest.approx(1.0, abs=1e-3)
