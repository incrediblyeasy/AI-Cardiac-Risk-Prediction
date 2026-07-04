"""CardioCausalConfig round-trip + validation; dataset stubs fail loudly."""

import pytest

from paper3_cardiocausal.training import CardioCausalConfig
from paper3_cardiocausal.datasets import build_linked_cohort, load_external


def test_config_json_roundtrip(tmp_path):
    cfg = CardioCausalConfig()
    cfg.fusion.latent_dim = 96
    cfg.scm.max_horizon = 6
    path = tmp_path / "cfg.json"
    cfg.to_file(path)
    reloaded = CardioCausalConfig.from_file(path)
    assert reloaded.fusion.latent_dim == 96
    assert reloaded.scm.max_horizon == 6


def test_config_rejects_unknown_key():
    with pytest.raises(ValueError):
        CardioCausalConfig.from_dict({"scm": {"hidden_dim": 8, "nope": 1}})


def test_smoke_config_loads():
    cfg = CardioCausalConfig.from_file("configs/cardiocausal_smoke.json")
    assert cfg.fusion.tabular_dim == 16
    assert cfg.train.device == "cpu"


def test_mimic_iv_stub_is_gated():
    with pytest.raises(NotImplementedError):
        build_linked_cohort()


def test_external_unknown_dataset_rejected():
    with pytest.raises(ValueError):
        load_external("not_a_dataset")


def test_external_known_dataset_is_gated_stub():
    with pytest.raises(NotImplementedError):
        load_external("code15")
