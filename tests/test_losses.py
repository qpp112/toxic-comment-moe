import numpy as np
import pytest

torch = pytest.importorskip("torch")

from toxic_moe.losses import (  # noqa: E402
    MultiLabelLoss,
    build_loss,
    class_balanced_pos_weight,
    effective_number,
    focal_bce,
    inverse_frequency_pos_weight,
)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(16, 6, generator=g) * 3
    targets = (torch.rand(16, 6, generator=g) < 0.2).float()
    return logits, targets


def test_gamma_zero_equals_bce():
    logits, y = _batch()
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert torch.allclose(MultiLabelLoss()(logits, y), ref, atol=1e-6)


def test_pos_weight_matches_torch_bce_with_logits():
    logits, y = _batch(1)
    pw = torch.tensor([1.0, 5.0, 2.0, 30.0, 2.0, 8.0])
    ref = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, y)
    assert torch.allclose(MultiLabelLoss(pos_weight=pw.numpy())(logits, y), ref, atol=1e-5)


def test_focal_downweights_easy_examples():
    y = torch.tensor([[1.0], [1.0]])
    logits = torch.tensor([[4.0], [-1.0]])  # easy (p~0.98) vs hard (p~0.27) positive
    bce = focal_bce(logits, y, gamma=0.0)
    foc = focal_bce(logits, y, gamma=2.0)
    assert (foc <= bce).all()
    assert (foc[0] / bce[0]) < (foc[1] / bce[1])  # easy example shrinks much more


def test_focal_gradients_finite_for_extreme_logits():
    logits = torch.tensor([[50.0, -50.0, 0.0]], requires_grad=True)
    y = torch.tensor([[0.0, 1.0, 1.0]])
    focal_bce(logits, y, gamma=2.0, pos_weight=torch.tensor([10.0, 10.0, 10.0])).mean().backward()
    assert torch.isfinite(logits.grad).all()


def test_effective_number_limits():
    n = np.array([1, 10, 1000])
    assert np.allclose(effective_number(n, 0.0), 1.0)
    assert np.allclose(effective_number(n, 1.0), n)
    e = effective_number(n, 0.999)
    assert np.all(np.diff(e) > 0) and np.all(e <= n) and e[-1] < 1 / (1 - 0.999)


def test_class_balanced_weights_are_softer_than_inverse_frequency():
    pos = np.array([15000, 1500, 450])
    total = 150_000
    inv = inverse_frequency_pos_weight(pos, total)
    cb = class_balanced_pos_weight(pos, total, beta=0.9999)
    assert np.all(cb < inv)
    assert np.all(np.diff(cb) > 0)  # rarer label -> larger weight
    assert cb[-1] == pytest.approx(22.72, rel=1e-3)


@pytest.mark.parametrize("name", ["bce", "weighted_bce", "focal", "cb_focal"])
def test_build_loss_variants_run(name):
    logits, y = _batch(2)
    loss = build_loss(name, np.array([30, 5, 20, 2, 18, 6]), 200)
    value = loss(logits.requires_grad_(), y)
    value.backward()
    assert value.ndim == 0 and torch.isfinite(value)


def test_pos_weight_moves_with_module():
    loss = build_loss("cb_focal", np.array([10, 1]), 100)
    assert "pos_weight" in dict(loss.named_buffers())
