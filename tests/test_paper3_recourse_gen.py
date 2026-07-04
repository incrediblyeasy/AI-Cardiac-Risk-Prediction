"""Gradient recourse: lowers risk via modifiable features only, pins immutables."""

import torch

from paper3_cardiocausal.recourse import ModifiabilityMask, generate_recourse


def _risk_fn(x):
    # Risk driven by systolic BP (feature 1); age (feature 0) is irrelevant here.
    return torch.sigmoid((x[:, 1] - 120.0) / 10.0)


def test_recourse_lowers_risk_and_pins_age():
    names = ["age", "sbp"]
    mask = ModifiabilityMask(names, modifiable=["sbp"])
    x = torch.tensor([[60.0, 170.0]])  # high SBP -> high risk
    before = _risk_fn(x)
    # proximity_weight=0 -> pure target-seeking, so the target risk is reachable.
    out = generate_recourse(
        x, _risk_fn, mask, target_risk=0.1, proximity_weight=0.0, steps=500, lr=1.0
    )

    assert out["risk"] < before                      # risk actually dropped
    assert out["recourse"][0, 0] == 60.0             # age unchanged (immutable)
    assert out["recourse"][0, 1] < x[0, 1]           # SBP was lowered
    assert bool(out["success"][0])                   # target achieved


def test_recourse_delta_only_on_modifiable():
    names = ["age", "sex", "sbp", "ldl"]
    mask = ModifiabilityMask(names, modifiable=["sbp", "ldl"])
    x = torch.tensor([[55.0, 1.0, 160.0, 4.0]])
    out = generate_recourse(x, lambda z: torch.sigmoid((z[:, 2] - 120) / 10), mask, steps=100)
    delta = out["delta"][0]
    assert delta[0] == 0.0 and delta[1] == 0.0       # age, sex never move
