"""Training loop: builders, one-epoch step, checkpointing/logging — no real data.

Uses synthetic beats (same trick as test_echofusenet) so the loop is exercised
end-to-end without the MIT-BIH download.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper1_echofusenet.data.beats import BeatSegment
from paper1_echofusenet.data.dataset import MultimodalBeatDataset
from paper1_echofusenet.training.config import TrainConfig
from paper1_echofusenet.training.train import (
    build_model,
    build_optimizer,
    build_scheduler,
    evaluate,
    train,
    train_one_epoch,
)


def _fake_loader(n=12, n_classes=5, batch_size=4, seed=0):
    rng = np.random.default_rng(seed)

    def beat(label, s):
        r = np.random.default_rng(s)
        # Give each class a distinct frequency so the model has signal to learn.
        sig = (
            np.sin(np.linspace(0, (label + 1) * np.pi, 64))
            + 0.05 * r.standard_normal(64)
        ).astype(np.float32)
        return BeatSegment(sig, "NSVFQ"[label], label, 1, 30, "DS1")

    beats = [beat(i % n_classes, int(rng.integers(0, 1 << 30))) for i in range(n)]
    ds = MultimodalBeatDataset(beats)
    return DataLoader(ds, batch_size=batch_size)


def _tiny_cfg(out_dir: Path) -> TrainConfig:
    return TrainConfig.from_dict(
        {
            "model": {"widths": [8, 16, 16], "fusion_hidden": 16},
            "optim": {"lr": 0.01, "scheduler": "cosine"},
            "train": {
                "epochs": 2,
                "device": "cpu",
                "out_dir": str(out_dir),
                "log_interval": 0,
            },
        }
    )


def test_build_optimizer_and_scheduler():
    cfg = TrainConfig.from_dict({"optim": {"name": "sgd", "scheduler": "step"}})
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.SGD)
    sched = build_scheduler(opt, cfg)
    assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    cfg_none = TrainConfig.from_dict({"optim": {"scheduler": "none"}})
    assert build_scheduler(opt, cfg_none) is None


def test_evaluate_returns_report_shape():
    cfg = TrainConfig.from_dict({"model": {"widths": [8, 16, 16], "fusion_hidden": 16}})
    model = build_model(cfg)
    report = evaluate(model, _fake_loader(), torch.device("cpu"), 5)
    assert report.confusion.shape == (5, 5)
    assert 0.0 <= report.accuracy <= 1.0


def test_train_writes_checkpoints_and_history(tmp_path):
    cfg = _tiny_cfg(tmp_path / "run")
    model = build_model(cfg)
    loader = _fake_loader()

    summary = train(model, loader, loader, cfg, device=torch.device("cpu"))

    out = Path(cfg.train.out_dir)
    assert (out / "best.pt").exists()
    assert (out / "last.pt").exists()
    assert (out / "config.json").exists()

    # history.jsonl has one record per epoch with the expected keys.
    lines = (out / "history.jsonl").read_text().strip().splitlines()
    assert len(lines) == cfg.train.epochs
    rec = json.loads(lines[0])
    for key in ("epoch", "train_loss", "lr", "accuracy", "macro_f1", "per_class_f1"):
        assert key in rec

    # Saved checkpoint reloads and carries the config snapshot.
    ckpt = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["config"]["model"]["fusion_hidden"] == 16
    assert "epoch" in summary and "macro_f1" in summary


def test_training_reduces_loss(tmp_path):
    # Two epochs of overfitting a tiny separable dataset must lower train loss.
    cfg = _tiny_cfg(tmp_path / "run2")
    cfg.train.epochs = 5
    model = build_model(cfg)
    loader = _fake_loader(n=20)
    device = torch.device("cpu")
    opt = build_optimizer(model, cfg)
    crit = torch.nn.CrossEntropyLoss()

    first = train_one_epoch(model, loader, opt, crit, device)
    for _ in range(4):
        last = train_one_epoch(model, loader, opt, crit, device)
    assert last < first
