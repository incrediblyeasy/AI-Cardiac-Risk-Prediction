"""§5 significance convenience layer over training.stats."""

import numpy as np

from paper1_echofusenet.evaluation import (
    ComparisonReport,
    compare_models,
    mcnemar_test,
    wilcoxon_test,
)


def test_reexports_are_the_same_callables():
    # The convenience package must re-export the stats implementations, not fork.
    from paper1_echofusenet.training import stats

    assert wilcoxon_test is stats.wilcoxon_test
    assert mcnemar_test is stats.mcnemar_test


def test_compare_models_full_battery():
    rng = np.random.default_rng(0)
    # Model A consistently above B, but with a varying (non-constant) gap so the
    # paired t-test has non-degenerate difference variance.
    fold_a = [0.80, 0.82, 0.79, 0.81, 0.83]
    fold_b = [0.74, 0.71, 0.76, 0.70, 0.75]
    yt = rng.integers(0, 5, 300)
    pa = yt.copy(); ma = rng.random(300) < 0.15; pa[ma] = rng.integers(0, 5, int(ma.sum()))
    pb = yt.copy(); mb = rng.random(300) < 0.30; pb[mb] = rng.integers(0, 5, int(mb.sum()))

    rep = compare_models(
        "A", "B", fold_a, fold_b, (yt, pa), (yt, pb), n_boot=200
    )
    assert isinstance(rep, ComparisonReport)
    assert rep.wilcoxon is not None and rep.paired_t is not None
    assert rep.mcnemar is not None
    assert rep.ci_a is not None and rep.ci_b is not None
    # Model A (fewer errors) should have the higher point estimate.
    assert rep.ci_a.point > rep.ci_b.point
    d = rep.summary_dict()
    assert set(d) == {"models", "wilcoxon", "paired_t", "mcnemar", "bootstrap_ci"}


def test_compare_models_partial_inputs():
    # Only per-fold scores -> only paired tests populated; the rest stay None.
    rep = compare_models("A", "B", [0.80, 0.90, 0.85], [0.70, 0.83, 0.74])
    assert rep.wilcoxon is not None
    assert rep.mcnemar is None
    assert rep.ci_a is None
