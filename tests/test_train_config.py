"""Config system: defaults, JSON round-trip, unknown-key rejection, ship configs."""

import json
from pathlib import Path

import pytest

from paper1_echofusenet.training.config import TrainConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_are_sane():
    cfg = TrainConfig()
    assert cfg.model.n_classes == 5
    assert cfg.data.oversample is True         # protocol: balance train fold
    assert cfg.train.checkpoint_metric == "macro_f1"
    assert cfg.optim.scheduler == "cosine"


def test_partial_dict_fills_defaults():
    cfg = TrainConfig.from_dict({"optim": {"lr": 0.01}})
    assert cfg.optim.lr == 0.01
    assert cfg.optim.name == "adamw"           # untouched default
    assert cfg.train.epochs == 30


def test_json_round_trip(tmp_path):
    cfg = TrainConfig.from_dict(
        {"train": {"epochs": 7, "seed": 42}, "model": {"dropout": 0.5}}
    )
    path = tmp_path / "cfg.json"
    cfg.to_file(path)
    reloaded = TrainConfig.from_file(path)
    assert reloaded.to_dict() == cfg.to_dict()
    assert reloaded.train.epochs == 7
    assert reloaded.model.dropout == 0.5


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        TrainConfig.from_dict({"optim": {"learnign_rate": 0.01}})  # typo


def test_unknown_section_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        TrainConfig.from_dict({"optimiser": {}})  # wrong section name


@pytest.mark.parametrize("name", ["echofusenet_default", "echofusenet_smoke"])
def test_shipped_configs_load(name):
    path = CONFIG_DIR / f"{name}.json"
    cfg = TrainConfig.from_file(path)
    # Re-dumping matches the loaded object (no silently-dropped keys).
    assert TrainConfig.from_dict(json.loads(path.read_text())).to_dict() == cfg.to_dict()
    assert cfg.model.n_classes == 5
