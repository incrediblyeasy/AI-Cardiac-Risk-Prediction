"""Target-trial-emulation protocol — pre-registration object.

Ordering matters for validity, not just tidiness (roadmap §4.3): the target-trial
protocol must be specified *before* any causal estimation is run. This dataclass
is that pre-registration record — PICOT elements, the confounder set, the DAG
edges, and the time-zero rule — serialisable to JSON so it can be committed and
frozen ahead of the analysis, then referenced by every causal estimate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TargetTrialProtocol:
    """A pre-registered target-trial-emulation specification.

    Fields mirror the PICOT framework plus the causal-identification essentials.
    Populate and freeze (``to_file``) before estimation begins.
    """

    population: str = ""              # eligibility criteria (the "P")
    intervention: str = ""           # treatment strategy (the "I")
    comparator: str = ""             # comparison strategy (the "C")
    outcome: str = ""                # outcome + ascertainment (the "O")
    time_zero_rule: str = ""         # when follow-up starts (alignment of elig/treat/eval)
    follow_up: str = ""              # follow-up window (the "T")
    confounders: list[str] = field(default_factory=list)   # adjustment set
    dag_edges: list[tuple[str, str]] = field(default_factory=list)  # (cause, effect)
    negative_controls: list[str] = field(default_factory=list)  # NCO / NCE variables
    notes: str = ""

    def is_complete(self) -> bool:
        """True once every field required to run estimation is filled in."""
        return all(
            [
                self.population,
                self.intervention,
                self.comparator,
                self.outcome,
                self.time_zero_rule,
                self.follow_up,
                self.confounders,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_file(self, path: str | Path) -> None:
        """Freeze the protocol to JSON (create parent dirs)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")

    @classmethod
    def from_file(cls, path: str | Path) -> "TargetTrialProtocol":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # JSON has no tuples; normalise dag_edges back to (cause, effect) pairs.
        data["dag_edges"] = [tuple(e) for e in data.get("dag_edges", [])]
        return cls(**data)
