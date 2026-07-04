"""CVAETrainConfig round-trip + validation, and the training-step scaffold."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from paper2_causalechonet.training import CVAETrainConfig
from paper2_causalechonet.training.train import (
    beta_at,
    build_cvae,
    build_optimizer,
    run_from_config,
    train_one_epoch,
)


def test_config_json_roundtrip(tmp_path):
    cfg = CVAETrainConfig()
    cfg.cvae.latent_dim = 24
    cfg.train.beta = 4.0
    path = tmp_path / "cfg.json"
    cfg.to_file(path)
    reloaded = CVAETrainConfig.from_file(path)
    assert reloaded.cvae.latent_dim == 24
    assert reloaded.train.beta == 4.0


def test_config_rejects_unknown_key():
    with pytest.raises(ValueError):
        CVAETrainConfig.from_dict({"cvae": {"latent_dim": 8, "bogus": 1}})


def test_smoke_config_loads():
    cfg = CVAETrainConfig.from_file("configs/causalechonet_cvae_smoke.json")
    assert cfg.cvae.latent_dim == 16
    assert cfg.train.device == "cpu"


def test_beta_warmup_schedule():
    cfg = CVAETrainConfig()
    cfg.train.beta = 2.0
    cfg.train.beta_warmup_epochs = 4
    assert beta_at(cfg, 1) == pytest.approx(0.5)
    assert beta_at(cfg, 4) == pytest.approx(2.0)
    assert beta_at(cfg, 10) == pytest.approx(2.0)  # clamped after warm-up


def test_train_one_epoch_runs_and_reduces_loss():
    cfg = CVAETrainConfig()
    cfg.cvae.latent_dim = 8
    cfg.cvae.hidden_dim = 32
    model = build_cvae(cfg, representation_dim=48)
    opt = build_optimizer(model, cfg)
    x = torch.randn(64, 48)
    cond = torch.randint(0, 5, (64,))
    loader = DataLoader(TensorDataset(x, cond), batch_size=16)
    device = torch.device("cpu")

    first = train_one_epoch(model, loader, opt, device, beta=1.0)
    for _ in range(5):
        last = train_one_epoch(model, loader, opt, device, beta=1.0)
    assert set(first) == {"total", "recon", "kl"}
    assert last["recon"] < first["recon"]  # it actually learns to reconstruct


def test_run_from_config_requires_checkpoint():
    # Runtime gate (not a code stub): without a frozen Paper-1 checkpoint it must
    # fail loudly and early, before touching any data.
    cfg = CVAETrainConfig.from_file("configs/causalechonet_cvae_smoke.json")
    assert cfg.encoder.checkpoint is None
    with pytest.raises(ValueError):
        run_from_config(cfg)
