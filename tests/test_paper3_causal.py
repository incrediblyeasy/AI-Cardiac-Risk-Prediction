"""Paper 3 causal-validation: E-values + target-trial protocol object."""

import math

import pytest

from paper3_cardiocausal.causal_validation import (
    TargetTrialProtocol,
    e_value,
    e_value_ci,
)


def test_e_value_null_is_one():
    assert e_value(1.0) == 1.0


def test_e_value_known():
    # RR=2 -> 2 + sqrt(2*1) = 3.4142... (VanderWeele & Ding).
    assert abs(e_value(2.0) - (2.0 + math.sqrt(2.0))) < 1e-9


def test_e_value_inverts_protective_effect():
    # RR below 1 is inverted, so 0.5 and 2.0 give the same E-value.
    assert abs(e_value(0.5) - e_value(2.0)) < 1e-9


def test_e_value_rejects_nonpositive():
    with pytest.raises(ValueError):
        e_value(0.0)


def test_e_value_ci_uses_limit_near_null():
    # estimate>1, lower limit 1.5 -> E-value of 1.5.
    assert abs(e_value_ci(2.0, 1.5, 2.7) - e_value(1.5)) < 1e-9


def test_e_value_ci_crossing_null_is_one():
    assert e_value_ci(1.2, 0.9, 1.6) == 1.0          # CI crosses 1
    assert e_value_ci(0.8, 0.6, 1.1) == 1.0          # protective CI crosses 1


def test_protocol_completeness_and_roundtrip(tmp_path):
    proto = TargetTrialProtocol()
    assert not proto.is_complete()

    proto = TargetTrialProtocol(
        population="adults with a MIMIC-IV ECG",
        intervention="start beta-blocker",
        comparator="no beta-blocker",
        outcome="30-day mortality",
        time_zero_rule="first eligible ECG during admission",
        follow_up="30 days",
        confounders=["age", "sex", "egfr"],
        dag_edges=[("age", "outcome"), ("treatment", "outcome")],
        negative_controls=["injury_hospitalization"],
    )
    assert proto.is_complete()

    path = tmp_path / "protocol.json"
    proto.to_file(path)
    reloaded = TargetTrialProtocol.from_file(path)
    assert reloaded.confounders == ["age", "sex", "egfr"]
    # JSON has no tuples; from_file normalises dag_edges back to pairs.
    assert reloaded.dag_edges == [("age", "outcome"), ("treatment", "outcome")]
    assert reloaded.is_complete()
