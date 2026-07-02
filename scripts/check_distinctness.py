"""Distinctness verification report: RP vs GAF vs MTF are physically different.

For one representative beat per AAMI class (from DS1), computes all three
signal-to-image transforms, reports the pairwise correlation between them (after
min-max normalization so range differences don't inflate distinctness), and
saves a side-by-side comparison figure to
``docs/figures/transforms_comparison.png``.

This is the human-readable companion to the automated guard in
`tests/test_distinctness.py`. A correlation near 1.0 for any pair would flag the
channel-duplication defect from the original draft.

Usage:
    python -m scripts.check_distinctness
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from paper1_echofusenet.data import AAMI_CLASSES
from paper1_echofusenet.data.beats import extract_beats
from paper1_echofusenet.data.mitbih import load_record
from paper1_echofusenet.data.splits import DS1_PATIENTS
from paper1_echofusenet.transforms import (
    gramian_angular_field,
    markov_transition_field,
    recurrence_plot,
)

OUT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "figures"
    / "transforms_comparison.png"
)
MODALITIES = ("RP", "GAF", "MTF")
CMAPS = {"RP": "viridis", "GAF": "rainbow", "MTF": "magma"}


def _transforms(signal) -> dict:
    return {
        "RP": recurrence_plot(signal),
        "GAF": gramian_angular_field(signal),
        "MTF": markov_transition_field(signal),
    }


def _minmax(a):
    a = a.astype(np.float64)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _corr(a, b) -> float:
    return float(np.corrcoef(_minmax(a).ravel(), _minmax(b).ravel())[0, 1])


def _one_beat_per_class() -> dict:
    found: dict = {}
    for record_id in DS1_PATIENTS:
        record = load_record(record_id)
        for beat in extract_beats(record, fold="DS1"):
            if beat.aami not in found:
                found[beat.aami] = beat
        if len(found) == len(AAMI_CLASSES):
            break
    return found


def main() -> None:
    beats = _one_beat_per_class()
    classes = [c for c in AAMI_CLASSES if c in beats]

    # --- correlation report ---
    pairs = list(combinations(MODALITIES, 2))
    header = "class  " + "  ".join(f"{a}-{b}" for a, b in pairs)
    print("Pairwise |correlation| between transforms (min-max normalized):")
    print(header)
    worst = 0.0
    for cls in classes:
        fields = _transforms(beats[cls].signal)
        cells = []
        for a, b in pairs:
            c = abs(_corr(fields[a], fields[b]))
            worst = max(worst, c)
            cells.append(f"{c:5.3f}")
        print(f"  {cls}    " + "   ".join(cells))
    verdict = "DISTINCT" if worst < 0.95 else "POSSIBLE DUPLICATION"
    print(f"\nMax pairwise correlation across all classes: {worst:.3f}  -> {verdict}")

    # --- comparison figure: rows = modalities, cols = classes ---
    n = len(classes)
    fig, axes = plt.subplots(len(MODALITIES), n, figsize=(3 * n, 3 * len(MODALITIES)))
    for col, cls in enumerate(classes):
        fields = _transforms(beats[cls].signal)
        for row, mod in enumerate(MODALITIES):
            ax = axes[row, col]
            ax.imshow(fields[mod], cmap=CMAPS[mod], origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{cls}  (rec {beats[cls].record_id})")
            if col == 0:
                ax.set_ylabel(mod, fontsize=12)
    fig.suptitle("RP vs GAF vs MTF per AAMI class (DS1) — distinct modalities", fontsize=13)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
