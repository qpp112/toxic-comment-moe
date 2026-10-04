import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from toxic_moe.config import Config  # noqa: E402
from toxic_moe.models import (  # noqa: E402
    HEADS,
    LabelAttentionHead,
    MLPHead,
    MMoEHead,
    MoEHead,
    ToxicityClassifier,
    build_model,
    load_balancing_loss,
)
from toxic_moe.train import build_optimizer  # noqa: E402


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


SMALL = dict(num_experts=3, expert_hidden=8, mlp_hidden=16, attn_hidden=8)


@pytest.mark.parametrize("arch", HEADS)
def test_every_head_gives_six_finite_logits_and_gradients(arch):
    torch.manual_seed(0)
    model = ToxicityClassifier(tiny_encoder(), arch=arch, **SMALL)
    out = model(*_inputs())
    assert out.logits.shape == (5, 6) and out.logits.dtype == torch.float32
    assert torch.isfinite(out.logits).all()
    out.logits.sum().backward()
    head_grad = sum(float(p.grad.abs().sum()) for p in model.head.parameters() if p.grad is not None)
    encoder_grad = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
    assert head_grad > 0 and encoder_grad > 0


def test_mmoe_gates_are_one_distribution_per_label():
    head = MMoEHead(16, 6, num_experts=4, expert_hidden=8).eval()
    out = head(torch.randn(7, 16))
    assert out.logits.shape == (7, 6) and out.gates.shape == (7, 6, 4) and out.aux_loss is None
    assert torch.allclose(out.gates.sum(-1), torch.ones(7, 6), atol=1e-5)
    # unlike the per-comment router, different labels mix the experts differently for one comment
    assert not torch.allclose(out.gates[:, 0], out.gates[:, 1])


def test_mmoe_logits_match_manual_computation():
    head = MMoEHead(16, 3, num_experts=4, expert_hidden=8).eval()
    h = torch.randn(5, 16)
    out = head(h)
    with torch.no_grad():
        z = torch.stack([expert(h) for expert in head.experts], dim=1)  # (5, 4, 8)
        for label in range(3):
            mixed = (out.gates[:, label, :, None] * z).sum(dim=1)  # (5, 8)
            manual = mixed @ head.towers.weight[label] + head.towers.bias[label]
            assert torch.allclose(out.logits[:, label], manual, atol=1e-5)


def test_label_attention_is_a_distribution_over_real_tokens():
    torch.manual_seed(0)
    head = LabelAttentionHead(16, 6, attn_hidden=8).eval()
    states = torch.randn(2, 10, 16)
    mask = torch.ones(2, 10, dtype=torch.long)
    mask[0, 6:] = 0
    out = head(states, mask)
    assert out.logits.shape == (2, 6) and out.attention.shape == (2, 6, 10)
    assert torch.allclose(out.attention.sum(-1), torch.ones(2, 6), atol=1e-5)
    assert (out.attention[0, :, 6:] == 0).all()
    changed = states.clone()
    changed[0, 6:] = 100.0  # garbage in the padding must not change the prediction
    assert torch.allclose(head(changed, mask).logits[0], out.logits[0], atol=1e-5)


def test_label_attention_logit_is_attention_weighted_states():
    head = LabelAttentionHead(16, 3, attn_hidden=8).eval()
    states = torch.randn(1, 5, 16)
    out = head(states, torch.ones(1, 5, dtype=torch.long))
    with torch.no_grad():
        for label in range(3):
            r = (out.attention[0, label, :, None] * states[0]).sum(dim=0)
            manual = r @ head.out.weight[label] + head.out.bias[label]
            assert torch.allclose(out.logits[0, label], manual, atol=1e-5)


def test_mlp_control_has_about_the_moe_head_parameter_count():
    mlp = sum(p.numel() for p in MLPHead(768, 6, mlp_hidden=1536).parameters())
    moe = sum(p.numel() for p in MoEHead(768, 6, num_experts=6, expert_hidden=256).parameters())
    assert abs(mlp - moe) / moe < 0.01


@pytest.mark.parametrize("arch", ["bert", "label_attn"])
def test_frozen_encoder_trains_only_the_head(arch):
    model = ToxicityClassifier(tiny_encoder(), arch=arch, freeze_encoder=True, **SMALL)
    model.train()
    assert not model.encoder.training and model.head.training  # a frozen encoder runs without dropout
    model(*_inputs()).logits.sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.head.parameters())
    optimizer = build_optimizer(model, lr=1e-5, head_lr=1e-3, weight_decay=0.01)
    in_optimizer = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    assert in_optimizer == sum(p.numel() for p in model.head.parameters())


def test_build_model_passes_the_new_options():
    model = build_model(Config(arch="label_attn", attn_hidden=8, freeze_encoder=True), encoder=tiny_encoder())
    assert isinstance(model.head, LabelAttentionHead) and model.head.proj.out_features == 8
    assert model.freeze_encoder and not any(p.requires_grad for p in model.encoder.parameters())
    model = build_model(Config(arch="mlp", mlp_hidden=12), encoder=tiny_encoder())
    assert model.head.net.net[0].out_features == 12
    model = build_model(Config(arch="mmoe", num_experts=5, expert_hidden=8), encoder=tiny_encoder())
    assert len(model.head.experts) == 5 and model.head.gate.out_features == 6 * 5


def test_word_attention_merges_word_pieces():
    from toxic_moe.predict import word_attention

    skip = {"[CLS]", "[SEP]", "<s>", "</s>"}
    bert = word_attention(["[CLS]", "you", "kill", "##ing", "[SEP]"], np.array([0.1, 0.2, 0.3, 0.3, 0.1]), skip)
    assert bert == [("you", pytest.approx(0.2)), ("killing", pytest.approx(0.6))]
    deberta = word_attention(
        ["<s>", "\u2581you", "\u2581kill", "ing", "</s>"], np.array([0.1, 0.2, 0.3, 0.3, 0.1]), skip
    )
    assert deberta == [("you", pytest.approx(0.2)), ("killing", pytest.approx(0.6))]
