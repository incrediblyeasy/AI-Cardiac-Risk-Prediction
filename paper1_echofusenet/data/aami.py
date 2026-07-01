"""AAMI beat-class mapping for the MIT-BIH Arrhythmia Database.

The ANSI/AAMI EC57 recommendation groups the raw MIT-BIH beat annotation
symbols into five classes: N, S, V, F, Q. This is the mapping used by de Chazal
et al. (2004) and by essentially every inter-patient MIT-BIH paper since.

Reference:
    de Chazal, O'Dwyer, Reilly, "Automatic classification of heartbeats using
    ECG morphology and heartbeat interval features," IEEE TBME 51(7), 2004.

Only *beat* annotations are mapped. Non-beat annotations (rhythm markers,
signal-quality markers, etc.) are not beats and are filtered out upstream; see
``NON_BEAT_SYMBOLS``.
"""

from __future__ import annotations

# Canonical class order (index == integer label used by the model).
AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")
CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(AAMI_CLASSES)}

# MIT-BIH beat symbol -> AAMI super-class.
#
#   N  Normal / bundle-branch-block / escape beats
#   S  Supraventricular ectopic beats (SVEB)
#   V  Ventricular ectopic beats (VEB)
#   F  Fusion of ventricular and normal
#   Q  Paced / fusion-of-paced / unclassifiable (the "unknown" bucket)
SYMBOL_TO_AAMI: dict[str, str] = {
    # --- N: any non-ectopic beat ---
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat
    # --- S: supraventricular ectopic ---
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Supraventricular premature beat
    # --- V: ventricular ectopic ---
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat
    # --- F: fusion ---
    "F": "F",  # Fusion of ventricular and normal beat
    # --- Q: unknown ---
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
}

# WFDB beat-annotation symbols that carry no AAMI class here. These appear in
# MIT-BIH but are excluded from the 5-class problem (and 'P' paced context is
# handled via '/' above). Kept explicit so filtering is auditable.
NON_BEAT_SYMBOLS: frozenset[str] = frozenset(
    {"+", "~", "|", "x", "[", "]", "!", '"', "=", "@", "p", "t", "u", "?"}
)


def symbol_to_aami(symbol: str) -> str | None:
    """Return the AAMI class for a MIT-BIH beat symbol, or ``None`` if it is
    not a mappable beat symbol (non-beat annotation)."""
    return SYMBOL_TO_AAMI.get(symbol)


def class_index(aami_class: str) -> int:
    """Integer label (0..4) for an AAMI class string."""
    return CLASS_TO_INDEX[aami_class]
