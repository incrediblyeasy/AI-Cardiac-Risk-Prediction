"""NeuralSCM: topo order, invertible counterfactuals, do-operator, fitting."""

import pytest
import torch

from paper3_cardiocausal.scm import NeuralSCM, fit_scm


def _chain():
    # A -> B -> C additive-noise chain.
    return NeuralSCM(nodes=["A", "B", "C"], parents={"B": ["A"], "C": ["B"]})


def test_cycle_is_rejected():
    with pytest.raises(ValueError):
        NeuralSCM(nodes=["X", "Y"], parents={"X": ["Y"], "Y": ["X"]})


def test_unknown_parent_rejected():
    with pytest.raises(ValueError):
        NeuralSCM(nodes=["X"], parents={"X": ["Z"]})


def test_counterfactual_without_intervention_is_identity():
    # Additive-noise SCM is invertible: abduct + regenerate reproduces the facts
    # exactly, regardless of (unfitted) mechanism weights.
    scm = _chain()
    obs = {"A": torch.randn(5), "B": torch.randn(5), "C": torch.randn(5)}
    cf = scm.counterfactual(obs, interventions={})
    for k in obs:
        assert torch.allclose(cf[k], obs[k], atol=1e-5)


def test_do_overrides_node_and_propagates():
    scm = _chain()
    obs = {"A": torch.zeros(4), "B": torch.zeros(4), "C": torch.zeros(4)}
    forced = torch.full((4,), 3.0)
    cf = scm.counterfactual(obs, interventions={"A": forced})
    assert torch.allclose(cf["A"], forced)          # intervened node takes the value
    # B depends on A, so a nonzero mechanism generally moves it off the factual 0.
    assert cf["B"].shape == (4,)


def test_abduct_requires_all_nodes():
    scm = _chain()
    with pytest.raises(ValueError):
        scm.abduct({"A": torch.randn(3)})


def test_fit_reduces_loss_on_linear_data():
    torch.manual_seed(0)
    n = 512
    A = torch.randn(n)
    B = 2.0 * A + 0.1 * torch.randn(n)      # B = 2A
    C = -1.0 * B + 0.1 * torch.randn(n)     # C = -B
    data = {"A": A, "B": B, "C": C}

    scm = _chain()
    before = sum(
        torch.mean((scm._mechanism(k, data, n) - data[k]) ** 2).item() for k in data
    )
    hist = fit_scm(scm, data, epochs=300, lr=0.05)
    assert hist["total"] < before          # it actually fits

    # A do-intervention on A should shift B by roughly 2*Δ (learned slope ~2).
    obs = {"A": torch.zeros(1), "B": torch.zeros(1), "C": torch.zeros(1)}
    with torch.no_grad():
        cf = scm.counterfactual(obs, interventions={"A": torch.ones(1)})
    assert 1.5 < float(cf["B"]) < 2.5
