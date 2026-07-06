"""§4 uncertainty: MC Dropout aggregation + temperature-scaling calibration."""

import numpy as np
import torch

from paper1_echofusenet.models import EchoFuseNet
from paper1_echofusenet.uncertainty import (
    TemperatureScaler,
    expected_calibration_error,
    mc_dropout_predict,
)
from paper1_echofusenet.uncertainty.mc_dropout import enable_mc_dropout


def _tiny_model():
    return EchoFuseNet(widths=(8, 16, 16), fusion_hidden=16, dropout=0.5)


def test_enable_mc_dropout_keeps_bn_eval():
    model = _tiny_model()
    enable_mc_dropout(model)
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert dropouts and all(m.training for m in dropouts)   # dropout stochastic
    assert all(not m.training for m in bns)                 # BN deterministic


def test_mc_dropout_shapes_and_uncertainty_bounds():
    torch.manual_seed(0)
    model = _tiny_model()
    x = torch.rand(6, 1, 32, 32)
    res = mc_dropout_predict(model, (x, x.clone(), x.clone()), n_passes=10)
    assert res.mean_probs.shape == (6, 5)
    assert np.allclose(res.mean_probs.sum(axis=1), 1.0, atol=1e-5)
    assert (res.predictive_entropy >= 0).all()
    assert (res.mutual_information >= 0).all()
    # Mutual information (epistemic) never exceeds predictive entropy (total).
    assert (res.mutual_information <= res.predictive_entropy + 1e-6).all()
    assert res.std.sum() > 0                                 # passes differ


def test_temperature_scaling_reduces_ece_and_keeps_argmax():
    rng = np.random.default_rng(0)
    n, c = 400, 5
    labels = rng.integers(0, c, n)
    base = rng.normal(0, 1, (n, c))
    base[np.arange(n), labels] += 1.2
    logits = base * 4.0                                      # over-confident
    probs_before = torch.softmax(torch.tensor(logits), dim=1).numpy()

    scaler = TemperatureScaler().fit(logits, labels)
    probs_after = scaler.predict_proba(logits)

    assert scaler.temperature > 1.0                          # was over-confident
    ece_before = expected_calibration_error(probs_before, labels)
    ece_after = expected_calibration_error(probs_after, labels)
    assert ece_after < ece_before
    # Temperature scaling never changes the predicted class.
    assert np.array_equal(probs_before.argmax(1), probs_after.argmax(1))


def test_ece_perfect_calibration_zero():
    # Confidence exactly matches accuracy -> ECE 0. Two-class, all confidence 1.0,
    # all correct.
    labels = np.array([0, 1, 0, 1])
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(probs, labels) == 0.0
