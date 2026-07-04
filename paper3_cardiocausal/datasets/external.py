"""External-validation datasets: Chapman-Shaoxing + CODE-15% (stub).

External validation (roadmap §4) runs the frozen CardioCausal risk model — no
refitting — on out-of-distribution cohorts:
* **Chapman-Shaoxing** — 12-lead ECGs with rhythm labels.
* **CODE-15%** — 345,779 exams with mortality follow-up (the risk-outcome target).

Loaders are stubs until the downloads are scripted (mirroring Paper 1's
``data/download.py`` pattern) and the field mapping to the target-trial outcome is
fixed. Left as a named entry point so the evaluation harness has somewhere to plug
in once the data is in hand.
"""

from __future__ import annotations

from typing import Any

SUPPORTED = ("chapman_shaoxing", "code15")


def load_external(name: str, *args: Any, **kwargs: Any):
    """Not yet available — external cohorts need scripted download + field mapping."""
    if name not in SUPPORTED:
        raise ValueError(f"unknown external dataset {name!r}; choose from {SUPPORTED}")
    raise NotImplementedError(
        f"External dataset {name!r} loader is a scaffold. Script the download "
        "(mirror paper1_echofusenet/data/download.py), map fields to the "
        "target-trial outcome, then run the frozen risk model with no refitting "
        "(roadmap §4 external validation)."
    )
