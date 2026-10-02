from pathlib import Path

import pytest

from toxic_moe.config import Config, load_ablation, load_config, parse_overrides

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_valid():
    Config().validate()


@pytest.mark.parametrize(
    "bad",
    [{"arch": "rnn"}, {"loss": "mse"}, {"top_k": 9}, {"train_fraction": 0}, {"val_fraction": 1.0}, {"pooling": "max"}],
)
def test_invalid_values_rejected(bad):
    with pytest.raises(ValueError):
        Config().replace(**bad)


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        Config().replace(learning_rate=1e-3)


def test_parse_overrides_casts_types():
    out = parse_overrides(["epochs=1", "lr=3e-5", "fp16=false", "focal_alpha=none", "name=x"])
    assert out == {"epochs": 1, "lr": 3e-5, "fp16": False, "focal_alpha": None, "name": "x"}


def test_shipped_ablation_parses():
    runs = load_ablation(ROOT / "configs" / "ablation.yaml")
    assert len(runs) == 8
    assert {(r.arch, r.loss) for r in runs} == {
        (a, l) for a in ("bert", "moe") for l in ("bce", "weighted_bce", "focal", "cb_focal")
    }
    assert all(isinstance(r.lr, float) for r in runs)
    assert sum(r.save_checkpoint for r in runs) == 1


def test_cli_overrides_apply_to_every_run():
    runs = load_ablation(ROOT / "configs" / "ablation.yaml", {"epochs": 1, "train_fraction": 0.01})
    assert all(r.epochs == 1 and r.train_fraction == 0.01 for r in runs)


def test_moe_ablation_parses():
    assert len(load_ablation(ROOT / "configs" / "moe_ablation.yaml")) == 4


def test_load_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: demo\narch: moe\nloss: cb_focal\n")
    cfg = load_config(p, {"epochs": 3})
    assert (cfg.name, cfg.arch, cfg.loss, cfg.epochs) == ("demo", "moe", "cb_focal", 3)
