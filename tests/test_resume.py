"""Tests for train.py's --resume / resume_from support.

The core scenario this guards against: a Kaggle/Colab session disconnects
mid-training. Before this fix, restarting meant losing all progress (no way
to pick back up with the optimizer/scheduler/EMA/early-stopping state
intact). These tests train for real (tiny synthetic data, CPU, few epochs)
across two separate `train()` calls simulating the disconnect, rather than
mocking the checkpoint mechanism.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from paper1_echofusenet.models import EchoFuseNet
from paper1_echofusenet.training.config import TrainConfig
from paper1_echofusenet.training.train import set_seed, train


def _make_loader(n=32, n_classes=5, size=32, batch_size=8):
    rp = torch.randn(n, 1, size, size)
    gaf = torch.randn(n, 1, size, size)
    mtf = torch.randn(n, 1, size, size)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(rp, gaf, mtf, y), batch_size=batch_size)


def _tiny_config(out_dir, epochs=2, **overrides):
    cfg_dict = {
        "data": {"batch_size": 8, "oversample": False, "normalize": False,
                  "num_workers": 0, "data_dir": None, "train_records": None,
                  "test_records": None},
        "model": {"n_classes": 5, "widths": [8, 8, 8, 8, 8], "fusion_hidden": 16,
                   "dropout": 0.0},
        "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4,
                    "momentum": 0.9, "scheduler": "cosine", "step_size": 10,
                    "gamma": 0.1, "min_lr": 1e-6},
        "train": {"epochs": epochs, "device": "cpu", "seed": 0, "amp": False,
                    "ema": False, "ema_decay": 0.999, "grad_clip": 1.0,
                    "log_interval": 1000, "checkpoint_metric": "macro_f1",
                    "checkpoint_avg_last": 0, "early_stopping_patience": 0,
                    "early_stopping_min_delta": 0.0, "out_dir": str(out_dir),
                    "tensorboard": False},
    }
    cfg_dict["train"].update(overrides)
    return TrainConfig.from_dict(cfg_dict)


def _model():
    return EchoFuseNet(n_classes=5, widths=(8, 8, 8, 8, 8), fusion_hidden=16, dropout=0.0)


def test_checkpoint_contains_full_resumable_state(tmp_path):
    """last.pt / best.pt must carry scheduler + best-metric + early-stopping
    state, not just model/optimizer — that's the actual fix here."""
    out_dir = tmp_path / "run"
    cfg = _tiny_config(out_dir, epochs=2)
    train_loader, test_loader = _make_loader(), _make_loader(n=16)

    set_seed(0)
    train(_model(), train_loader, test_loader, cfg, device=torch.device("cpu"))

    ckpt = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
    for key in ("epoch", "model_state", "optimizer_state", "scheduler_state",
                "best_metric", "epochs_since_improve"):
        assert key in ckpt, f"{key} missing from checkpoint — resume would be incomplete"


def test_resume_continues_epoch_numbering_not_restart(tmp_path):
    """The actual disconnect scenario: train 2 epochs, 'crash', resume for 3 more."""
    out_dir = tmp_path / "run"
    train_loader, test_loader = _make_loader(), _make_loader(n=16)

    cfg = _tiny_config(out_dir, epochs=2)
    set_seed(0)
    train(_model(), train_loader, test_loader, cfg, device=torch.device("cpu"))

    last_ckpt = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
    assert last_ckpt["epoch"] == 2

    # Simulate resuming in a fresh process: new model instance, same out_dir.
    cfg2 = _tiny_config(out_dir, epochs=5)
    summary = train(
        _model(), train_loader, test_loader, cfg2, device=torch.device("cpu"),
        resume_from=out_dir / "last.pt",
    )

    history_lines = (out_dir / "history.jsonl").read_text().strip().split("\n")
    assert len(history_lines) == 5, "history should be continuous (2 + 3), not reset"
    epochs_logged = [json.loads(l)["epoch"] for l in history_lines]
    assert epochs_logged == [1, 2, 3, 4, 5], f"epoch numbering broke on resume: {epochs_logged}"


def test_resume_preserves_prior_history_lines_unchanged(tmp_path):
    """Resuming must APPEND to history.jsonl, never overwrite what's already there."""
    out_dir = tmp_path / "run"
    train_loader, test_loader = _make_loader(), _make_loader(n=16)

    cfg = _tiny_config(out_dir, epochs=2)
    set_seed(0)
    train(_model(), train_loader, test_loader, cfg, device=torch.device("cpu"))
    history_before = (out_dir / "history.jsonl").read_text().strip().split("\n")

    cfg2 = _tiny_config(out_dir, epochs=4)
    train(
        _model(), train_loader, test_loader, cfg2, device=torch.device("cpu"),
        resume_from=out_dir / "last.pt",
    )
    history_after = (out_dir / "history.jsonl").read_text().strip().split("\n")

    assert history_after[: len(history_before)] == history_before, \
        "resume altered/overwrote pre-crash history lines"


def test_resume_beyond_configured_epochs_is_a_noop_not_a_crash(tmp_path):
    """Resuming from a checkpoint that's already at or past cfg.train.epochs
    should report and return cleanly, not error or silently retrain epoch 1."""
    out_dir = tmp_path / "run"
    train_loader, test_loader = _make_loader(), _make_loader(n=16)

    cfg = _tiny_config(out_dir, epochs=3)
    set_seed(0)
    train(_model(), train_loader, test_loader, cfg, device=torch.device("cpu"))

    # Resume with the SAME epoch budget (3) — checkpoint is already at epoch 3.
    cfg_same = _tiny_config(out_dir, epochs=3)
    summary = train(
        _model(), train_loader, test_loader, cfg_same, device=torch.device("cpu"),
        resume_from=out_dir / "last.pt",
    )
    history_lines = (out_dir / "history.jsonl").read_text().strip().split("\n")
    assert len(history_lines) == 3, "should not have trained/logged extra epochs"


def test_resume_restores_early_stopping_bookkeeping(tmp_path):
    """epochs_since_improve must carry over — otherwise a resumed run could
    immediately re-trigger (or wrongly delay) early stopping."""
    out_dir = tmp_path / "run"
    train_loader, test_loader = _make_loader(), _make_loader(n=16)

    cfg = _tiny_config(out_dir, epochs=2)
    set_seed(0)
    train(_model(), train_loader, test_loader, cfg, device=torch.device("cpu"))
    last_ckpt = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
    saved_esi = last_ckpt["epochs_since_improve"]

    # Load via the resume path and confirm the restored value matches what
    # was actually saved (not silently reset to 0).
    from paper1_echofusenet.training.train import load_checkpoint_for_resume
    from paper1_echofusenet.training.train import build_optimizer, build_scheduler

    model = _model()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    state = load_checkpoint_for_resume(
        out_dir / "last.pt", model, optimizer, torch.device("cpu"), scheduler=scheduler
    )
    assert state["epochs_since_improve"] == saved_esi


def test_crossval_resumes_at_fold_level_not_just_epoch_level(tmp_path):
    """The CV-specific resume gap: if the whole script dies between folds
    (e.g. a Kaggle quota cutoff), a completed fold's *trained checkpoint*
    already survives (train()'s own resume), but its *evaluation* result
    didn't -- until this fix. Simulates exactly that: run CV, delete one
    fold's result to simulate 'never finished', rerun, confirm the still-
    present fold is skipped (reloaded identically) and only the missing
    one retrains.
    """
    import torch
    from paper1_echofusenet.data.beats import BeatSegment
    from paper1_echofusenet.training.config import TrainConfig
    from paper1_echofusenet.training.crossval import cross_validate

    rng_seed = 0
    import numpy as np
    rng = np.random.default_rng(rng_seed)

    def make_patient_beats(record_id, n=30):
        return [
            BeatSegment(
                signal=rng.normal(size=256).astype(np.float32),
                aami="N", label=0, record_id=record_id,
                r_peak=1000 + i, fold="DS1",
            )
            for i in range(n)
        ]

    beats = (
        make_patient_beats(101) + make_patient_beats(106)
        + make_patient_beats(108) + make_patient_beats(109)
    )

    out_root = tmp_path / "cv_run"
    cfg_dict = {
        "data": {"batch_size": 16, "oversample": False, "normalize": False,
                   "num_workers": 0, "data_dir": None, "train_records": None,
                   "test_records": None},
        "model": {"n_classes": 5, "widths": [8, 8, 8, 8, 8], "fusion_hidden": 16,
                    "dropout": 0.0},
        "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4,
                    "momentum": 0.9, "scheduler": "none", "step_size": 10,
                    "gamma": 0.1, "min_lr": 1e-6},
        "train": {"epochs": 1, "device": "cpu", "seed": 0, "amp": False,
                    "ema": False, "ema_decay": 0.999, "grad_clip": 1.0,
                    "log_interval": 1000, "checkpoint_metric": "macro_f1",
                    "checkpoint_avg_last": 0, "early_stopping_patience": 0,
                    "early_stopping_min_delta": 0.0, "out_dir": str(out_root),
                    "tensorboard": False},
    }
    cfg = TrainConfig.from_dict(cfg_dict)

    report1 = cross_validate(cfg, k=2, beats=beats, n_boot=50)
    assert (out_root / "fold_0" / "fold_result.npz").exists()
    assert (out_root / "fold_1" / "fold_result.npz").exists()

    # Simulate: fold 1 never finished (e.g. crashed mid-fold).
    (out_root / "fold_1" / "fold_result.npz").unlink()

    report2 = cross_validate(cfg, k=2, beats=beats, n_boot=50)
    assert len(report2.per_fold) == 2
    # Fold 0 must be the exact reloaded result, not a re-trained (likely
    # different, since training isn't perfectly deterministic across
    # unrelated calls) one.
    assert report2.per_fold[0].accuracy == report1.per_fold[0].accuracy
    assert report2.per_fold[0].macro_f1 == report1.per_fold[0].macro_f1
    np.testing.assert_array_equal(report2.per_fold[0].y_true, report1.per_fold[0].y_true)
    np.testing.assert_array_equal(report2.per_fold[0].y_pred, report1.per_fold[0].y_pred)


def test_crossval_no_resume_flag_forces_full_retrain(tmp_path):
    """--no-resume (resume=False) must retrain every fold regardless of
    what's already saved on disk -- an explicit escape hatch."""
    import numpy as np
    from paper1_echofusenet.data.beats import BeatSegment
    from paper1_echofusenet.training.config import TrainConfig
    from paper1_echofusenet.training.crossval import cross_validate

    rng = np.random.default_rng(0)

    def make_patient_beats(record_id, n=30):
        return [
            BeatSegment(
                signal=rng.normal(size=256).astype(np.float32),
                aami="N", label=0, record_id=record_id,
                r_peak=1000 + i, fold="DS1",
            )
            for i in range(n)
        ]

    beats = make_patient_beats(101) + make_patient_beats(106) + make_patient_beats(108) + make_patient_beats(109)

    out_root = tmp_path / "cv_run_no_resume"
    cfg_dict = {
        "data": {"batch_size": 16, "oversample": False, "normalize": False,
                   "num_workers": 0, "data_dir": None, "train_records": None,
                   "test_records": None},
        "model": {"n_classes": 5, "widths": [8, 8, 8, 8, 8], "fusion_hidden": 16,
                    "dropout": 0.0},
        "optim": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4,
                    "momentum": 0.9, "scheduler": "none", "step_size": 10,
                    "gamma": 0.1, "min_lr": 1e-6},
        "train": {"epochs": 1, "device": "cpu", "seed": 0, "amp": False,
                    "ema": False, "ema_decay": 0.999, "grad_clip": 1.0,
                    "log_interval": 1000, "checkpoint_metric": "macro_f1",
                    "checkpoint_avg_last": 0, "early_stopping_patience": 0,
                    "early_stopping_min_delta": 0.0, "out_dir": str(out_root),
                    "tensorboard": False},
    }
    cfg = TrainConfig.from_dict(cfg_dict)

    cross_validate(cfg, k=2, beats=beats, n_boot=50)
    assert (out_root / "fold_0" / "fold_result.npz").exists()

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cross_validate(cfg, k=2, beats=beats, n_boot=50, resume=False)
    assert "already completed" not in buf.getvalue()
