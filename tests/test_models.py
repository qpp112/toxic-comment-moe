import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from toxic_moe.config import Config  # noqa: E402
from toxic_moe.models import MoEHead, ToxicityClassifier, build_model, load_balancing_loss  # noqa: E402


def tiny_encoder():
    cfg = transformers.BertConfig(
        vocab_size=50,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    return transformers.BertModel(cfg)


def _inputs(batch=5, length=12):
    ids = torch.randint(5, 50, (batch, length))
    mask = torch.ones_like(ids)
    mask[0, 8:] = 0
    return ids, mask


def test_linear_head_shapes():
    model = ToxicityClassifier(tiny_encoder(), arch="bert")
    out = model(*_inputs())
    assert out.logits.shape == (5, 6) and out.gates is None and out.aux_loss is None


@pytest.mark.parametrize("top_k", [0, 1, 2, 6])
def test_moe_gates_are_distributions_with_k_nonzeros(top_k):
    head = MoEHead(32, 6, num_experts=6, expert_hidden=16, top_k=top_k)
    out = head(torch.randn(7, 32))
    assert out.logits.shape == (7, 6)
    assert torch.allclose(out.gates.sum(-1), torch.ones(7), atol=1e-5)
    nonzero = (out.gates > 0).sum(-1)
    expected = 6 if top_k in (0, 6) else top_k
    assert (nonzero == expected).all()


def test_moe_logits_are_gate_weighted_expert_outputs():
    head = MoEHead(16, 3, num_experts=4, expert_hidden=8, top_k=2).eval()
    h = torch.randn(3, 16)
    out = head(h)
    manual = sum(out.gates[:, i : i + 1] * head.experts[i](h) for i in range(4))
    assert torch.allclose(out.logits, manual, atol=1e-5)


def test_unselected_experts_get_no_gradient_from_a_sample():
    head = MoEHead(16, 3, num_experts=4, expert_hidden=8, top_k=1, dropout=0.0)
    h = torch.randn(1, 16)
    out = head(h)
    out.logits.sum().backward()
    chosen = int(out.gates.argmax())
    for i, expert in enumerate(head.experts):
        g = expert.net[0].weight.grad
        if i == chosen:
            assert g is not None and g.abs().sum() > 0
        else:
            assert g is None or g.abs().sum() == 0


def test_load_balancing_loss_is_one_when_uniform_and_larger_when_collapsed():
    uniform = torch.full((8, 4), 0.25)
    uniform[torch.arange(8), torch.arange(8) % 4] += 1e-3  # break argmax ties evenly
    uniform = uniform / uniform.sum(-1, keepdim=True)
    collapsed = torch.tensor([[0.97, 0.01, 0.01, 0.01]]).repeat(8, 1)
    assert load_balancing_loss(uniform).item() == pytest.approx(1.0, abs=1e-2)
    assert load_balancing_loss(collapsed).item() > 3.5


def test_mean_pooling_ignores_padding():
    model = ToxicityClassifier(tiny_encoder(), arch="moe", pooling="mean").eval()
    ids, mask = _inputs(batch=1, length=10)
    padded_ids = torch.cat([ids, torch.zeros(1, 5, dtype=torch.long)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(1, 5, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a = model(ids, mask).logits
        b = model(padded_ids, padded_mask).logits
    assert torch.allclose(a, b, atol=1e-4)


def test_build_model_uses_config():
    cfg = Config(arch="moe", num_experts=3, top_k=1, expert_hidden=8)
    model = build_model(cfg, encoder=tiny_encoder())
    assert len(model.head.experts) == 3 and model.head.top_k == 1
    enc, head = model.param_groups()
    assert enc and head
