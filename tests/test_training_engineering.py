"""§7 training engineering: EMA, checkpoint averaging, early stopping, ckpt-avg."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper1_echofusenet.data.beats import BeatSegment
from paper1_echofusenet.data.dataset import MultimodalBeatDataset
from paper1_echofusenet.training.config import TrainConfig
from paper1_echofusenet.training.train import (
    ModelEma,
    average_state_dicts,
    build_model,
    train,
)


def _fake_loader(n=12, n_classes=5, batch_size=4, seed=0):
    rng = np.random.default_rng(seed)

    def beat(label, s):
        r = np.random.default_rng(s)
        sig = (
            np.sin(np.linspace(0, (label + 1) * np.pi, 64))
            + 0.05 * r.standard_normal(64)
        ).astype(np.float32)
        return BeatSegment(sig, "NSVFQ"[label], label, 1, 30, "DS1")

    beats = [beat(i % n_classes, int(rng.integers(0, 1 << 30))) for i in range(n)]
    return DataLoader(MultimodalBeatDataset(beats), batch_size=batch_size)


def _tiny_cfg(out_dir: Path) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "model": {"widths": [8, 16, 16], "fusion_hidden": 16},
            "optim": {"lr": 0.01},
            "train": {"epochs": 2, "device": "cpu", "out_dir": str(out_dir),
                      "log_interval": 0},
        }
    )


def test_model_ema_tracks_and_restores():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    ema = ModelEma(model, decay=0.9)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)
    ema.update(model)
    # Shadow moved toward the new weights but is not equal to them.
    assert not torch.equal(ema.shadow["weight"], original["weight"])
    assert not torch.equal(ema.shadow["weight"], model.weight)
    # store/copy_to/restore round-trips the live weights.
    ema.store(model)
    ema.copy_to(model)
    assert torch.equal(model.weight, ema.shadow["weight"])
    ema.restore(model)
    with torch.no_grad():
        model.weight.sub_(1.0)
    assert torch.allclose(model.weight, original["weight"])


def test_average_state_dicts_is_mean():
    a = {"w": torch.zeros(3), "n": torch.tensor(1)}
    b = {"w": torch.ones(3) * 4, "n": torch.tensor(2)}
    avg = average_state_dicts([a, b])
    assert torch.allclose(avg["w"], torch.ones(3) * 2)   # float averaged
    assert int(avg["n"]) == 2                            # int buffer -> last


def test_train_with_ema_and_checkpoint_averaging(tmp_path):
    cfg = _tiny_cfg(tmp_path / "run")
    cfg.train.ema = True
    cfg.train.ema_decay = 0.9
    cfg.train.checkpoint_avg_last = 2
    model = build_model(cfg)
    loader = _fake_loader()
    summary = train(model, loader, loader, cfg, device=torch.device("cpu"))
    assert (Path(cfg.train.out_dir) / "averaged.pt").exists()
    assert "averaged" in summary


def test_early_stopping_halts_early(tmp_path):
    # Constant-label data -> metric plateaus immediately -> early stop fires.
    cfg = _tiny_cfg(tmp_path / "run_es")
    cfg.train.epochs = 20
    cfg.train.early_stopping_patience = 2
    model = build_model(cfg)
    loader = _fake_loader()
    train(model, loader, loader, cfg, device=torch.device("cpu"))
    lines = (Path(cfg.train.out_dir) / "history.jsonl").read_text().strip().splitlines()
    # Should stop well before the full 20 epochs.
    assert len(lines) < 20


def test_focal_loss_config_runs(tmp_path):
    cfg = _tiny_cfg(tmp_path / "run_focal")
    cfg.loss.name = "focal"
    model = build_model(cfg)
    loader = _fake_loader()
    summary = train(model, loader, loader, cfg, device=torch.device("cpu"))
    assert "macro_f1" in summary
