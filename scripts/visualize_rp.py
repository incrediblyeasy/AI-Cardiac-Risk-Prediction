"""Save a grid of Recurrence Plot samples, one column per AAMI class.

Pulls one representative beat of each class (N, S, V, F, Q) from the DS1
(training) fold, renders its 1-D trace and its recurrence plot, and writes the
figure to ``docs/figures/rp_samples.png``.

Usage:
    python -m scripts.visualize_rp
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from paper1_echofusenet.data import AAMI_CLASSES
from paper1_echofusenet.data.beats import extract_beats
from paper1_echofusenet.data.mitbih import load_record
from paper1_echofusenet.data.splits import DS1_PATIENTS
from paper1_echofusenet.transforms import recurrence_plot

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "rp_samples.png"


def _one_beat_per_class() -> dict:
    """Find one BeatSegment per AAMI class by scanning DS1 records."""
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
    missing = [c for c in AAMI_CLASSES if c not in beats]
    if missing:
        print(f"warning: no beats found for classes {missing} in DS1")

    classes = [c for c in AAMI_CLASSES if c in beats]
    n = len(classes)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, cls in enumerate(classes):
        beat = beats[cls]
        rp = recurrence_plot(beat.signal)

        ax_sig = axes[0, col]
        ax_sig.plot(beat.signal, color="crimson", linewidth=0.9)
        ax_sig.set_title(f"{cls}  (rec {beat.record_id})")
        ax_sig.set_xticks([])
        ax_sig.set_yticks([])

        ax_rp = axes[1, col]
        ax_rp.imshow(rp, cmap="viridis", origin="lower")
        ax_rp.set_xticks([])
        ax_rp.set_yticks([])

    axes[0, 0].set_ylabel("beat signal", fontsize=11)
    axes[1, 0].set_ylabel("recurrence plot", fontsize=11)
    fig.suptitle("Recurrence Plots by AAMI class (DS1)", fontsize=13)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"Saved {OUT}  ({n} classes: {', '.join(classes)})")


if __name__ == "__main__":
    main()
