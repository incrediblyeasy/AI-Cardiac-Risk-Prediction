"""de Chazal inter-patient DS1 / DS2 split for the MIT-BIH Arrhythmia DB.

The inter-patient protocol trains on one set of patients (DS1) and tests on a
disjoint set (DS2) so that no heartbeat from a test patient is ever seen during
training. This is the honest evaluation protocol for MIT-BIH; the older
"intra-patient" protocol (random beat shuffle) leaks patient identity and
massively over-states accuracy.

The four paced records (102, 104, 107, 217) are excluded per the AAMI
recommendation because paced beats are not clinically meaningful for this task.
44 records remain: 22 in DS1, 22 in DS2.

Reference:
    de Chazal, O'Dwyer, Reilly, "Automatic classification of heartbeats using
    ECG morphology and heartbeat interval features," IEEE TBME 51(7), 2004,
    Table II.
"""

from __future__ import annotations

# Training patients (DS1) — 22 records.
DS1_PATIENTS: tuple[int, ...] = (
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
    124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
)

# Test patients (DS2) — 22 records.
DS2_PATIENTS: tuple[int, ...] = (
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
    212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
)

# Paced records excluded from the 44-record inter-patient benchmark.
EXCLUDED_PACED: tuple[int, ...] = (102, 104, 107, 217)

# Full 48-record set for cross-checking.
ALL_MITDB_RECORDS: tuple[int, ...] = tuple(
    sorted(DS1_PATIENTS + DS2_PATIENTS + EXCLUDED_PACED)
)


def split_for(record: int) -> str:
    """Return ``"DS1"``, ``"DS2"``, or ``"excluded"`` for a MIT-BIH record id."""
    if record in DS1_PATIENTS:
        return "DS1"
    if record in DS2_PATIENTS:
        return "DS2"
    if record in EXCLUDED_PACED:
        return "excluded"
    raise ValueError(f"{record} is not a MIT-BIH Arrhythmia DB record id")


def assert_no_leakage() -> None:
    """Fail loudly if DS1 and DS2 share any patient id. Called by the Day-2
    leakage test, but cheap enough to keep as a self-check here."""
    overlap = set(DS1_PATIENTS) & set(DS2_PATIENTS)
    if overlap:
        raise AssertionError(f"DS1/DS2 patient overlap detected: {sorted(overlap)}")
