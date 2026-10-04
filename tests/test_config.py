from pathlib import Path

import pytest

from toxic_moe.config import Config, load_ablation, load_config, parse_overrides

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_valid():
    Config().validate()


@pytest.mark.parametrize(
    "bad",
    [
        {"arch": "rnn"},
        {"loss": "mse"},
        {"top_k": 9},
        {"train_fraction": 0},
        {"val_fraction": 1.0},
        {"pooling": "max"},
        {"amp_dtype": "fp8"},
        {"mlp_hidden": 0},
        {"arch": "transformer"},
    ],
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


def test_shipped_ablation_expands_over_seeds():
    runs = load_ablation(ROOT / "configs" / "ablation.yaml")
    assert len(runs) == 24
    groups = {r.group for r in runs}
    assert len(groups) == 8
    assert {(r.arch, r.loss) for r in runs} == {
        (a, l) for a in ("bert", "moe") for l in ("bce", "weighted_bce", "focal", "cb_focal")
    }
    # seed-major order: the first 8 runs are a complete single-seed ablation
    assert {r.seed for r in runs[:8]} == {42} and {r.group for r in runs[:8]} == groups
    assert runs[0].name == "bert_bce_s42" and runs[0].group_name == "bert_bce"
    # the data split never changes across seeds, so runs are paired
    assert {r.split_seed for r in runs} == {42}
    assert all(isinstance(r.lr, float) for r in runs)
    assert [r.name for r in runs if r.save_checkpoint] == ["moe_cbfocal_s42"]


def test_seed_override_and_cli_overrides():
    runs = load_ablation(ROOT / "configs" / "ablation.yaml", {"epochs": 1, "train_fraction": 0.01}, seeds=[7])
    assert len(runs) == 8 and all(r.seed == 7 and r.name.endswith("_s7") for r in runs)
    assert all(r.epochs == 1 and r.train_fraction == 0.01 for r in runs)


def test_file_without_seeds_keeps_plain_names(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("base: {epochs: 1}\nruns:\n  - {name: x, arch: moe}\n")
    (run,) = load_ablation(p)
    assert run.name == "x" and run.group_name == "x" and run.seed == 42


def test_encoder_study_parses():
    runs = load_ablation(ROOT / "configs" / "encoders.yaml")
    assert len(runs) == 6
    assert {r.encoder for r in runs} == {"microsoft/deberta-v3-base"}


def test_follow_up_studies_parse():
    heads = load_ablation(ROOT / "configs" / "heads.yaml")
    assert len(heads) == 12 and {r.study for r in heads} == {"heads"} and {r.loss for r in heads} == {"bce"}
    assert {r.arch for r in heads} == {"mmoe", "label_attn", "mlp", "moe"}
    assert not any(r.freeze_encoder for r in heads) and {r.split_seed for r in heads} == {42}
    assert {r.aux_loss_weight for r in heads if r.group == "moe_bce_noaux"} == {0.0}
    assert sorted(r.name for r in heads if r.save_checkpoint) == ["labelattn_bce_s42", "mmoe_bce_s42"]
    assert all(r.title for r in heads)
    probes = load_ablation(ROOT / "configs" / "probes.yaml")
    assert len(probes) == 6 and all(r.freeze_encoder and r.study == "probes" for r in probes)
    assert [r.pooling for r in probes if r.group == "probe_linear_mean"] == ["mean"]
    assert {r.arch for r in probes} == {"bert", "mlp", "moe", "mmoe", "label_attn"}


def test_common_block_applies_to_every_run_but_runs_win(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text(
        "base: {epochs: 1}\ncommon: {epochs: 2, loss: focal}\nruns:\n  - {name: x}\n  - {name: y, epochs: 3}\n"
    )
    x, y = load_ablation(p)
    assert (x.epochs, x.loss, y.epochs, y.loss) == (2, "focal", 3, "focal")


def test_load_config_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: demo\narch: moe\nloss: cb_focal\n")
    cfg = load_config(p, {"epochs": 3})
    assert (cfg.name, cfg.arch, cfg.loss, cfg.epochs) == ("demo", "moe", "cb_focal", 3)
