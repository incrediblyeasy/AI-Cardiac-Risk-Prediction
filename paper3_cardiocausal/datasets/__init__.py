"""Dataset access for CardioCausal.

Three sources (roadmap §4):
* ``mimic_iv`` — MIMIC-IV-ECG + MIMIC-IV EHR linked cohort; the backbone for
  causal estimation. **Gated on PhysioNet credentialing** (identity verification +
  a training course) — a lead-time access-control step, not a technical one.
* ``external`` — Chapman-Shaoxing + CODE-15% for external validation.
* PTB-XL is reused from Paper 1 for representation development / benchmarking.

All loaders here are stubs that fail loudly with the exact access/credentialing
prerequisite, so the blocking step is explicit rather than a silent gap.
"""

from .mimic_iv import build_linked_cohort, link_ecg_ehr, subject_level_split
from .external import load_external

__all__ = [
    "build_linked_cohort",
    "link_ecg_ehr",
    "subject_level_split",
    "load_external",
]
