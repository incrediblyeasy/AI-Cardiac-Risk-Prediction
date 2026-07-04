"""Recourse engine: causally-consistent intervention recommendations.

Extends Paper 2's CVAE into a recourse generator that proposes changes over
**modifiable** variables only (e.g. blood pressure, medication) and never
immutable ones (age, sex) — a hard constraint for the recommendations to be
actionable and causally valid. The ``ModifiabilityMask`` that enforces this is
implemented and tested; the full generator (which composes the SCM's do-operator
with the CVAE) is gated on the SCM + cohort.
"""

from .engine import ModifiabilityMask, generate_recourse

__all__ = ["ModifiabilityMask", "generate_recourse"]
