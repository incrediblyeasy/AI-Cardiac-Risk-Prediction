"""Save a grid of Gramian Angular Field samples, one column per AAMI class.

Pulls one representative beat of each class (N, S, V, F, Q) from the DS1
(training) fold, renders its 1-D trace and its GASF image, and writes the figure
to ``docs/figures/gaf_samples.png``. Pass ``--method difference`` for GADF.

Usage:
    python -m scripts.visualize_gaf [--method summation|difference]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from paper1_echofusenet.data import AAMI_CLASSES
from paper1_echofusenet.data.beats import extract_beats
from paper1_echofusenet.data.mitbih import load_record
from paper1_echofusenet.data.splits import DS1_PATIENTS
from paper1_echofusenet.transforms import gramian_angular_field

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "gaf_samples.png"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default="summation",
        choices=["summation", "difference"],
        help="GASF (summation, default) or GADF (difference).",
    )
    args = parser.parse_args()
    label = "GASF" if args.method == "summation" else "GADF"

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
        gaf = gramian_angular_field(beat.signal, method=args.method)

        ax_sig = axes[0, col]
        ax_sig.plot(beat.signal, color="crimson", linewidth=0.9)
        ax_sig.set_title(f"{cls}  (rec {beat.record_id})")
        ax_sig.set_xticks([])
        ax_sig.set_yticks([])

        ax_gaf = axes[1, col]
        ax_gaf.imshow(gaf, cmap="rainbow", origin="lower", vmin=-1.0, vmax=1.0)
        ax_gaf.set_xticks([])
        ax_gaf.set_yticks([])

    axes[0, 0].set_ylabel("beat signal", fontsize=11)
    axes[1, 0].set_ylabel(label.lower(), fontsize=11)
    fig.suptitle(f"Gramian Angular Fields ({label}) by AAMI class (DS1)", fontsize=13)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"Saved {OUT}  ({n} classes: {', '.join(classes)}; method={args.method})")


if __name__ == "__main__":
    main()
