"""Sanity-check the multimodal DataLoader end to end.

Builds the full inter-patient split, applies oversampling to the training fold
only, and reports:

* batch shapes for one train and one test batch (three (B, 1, L, L) channels +
  labels),
* the DS1 (train) class distribution before vs after oversampling — it should
  become balanced,
* the DS2 (test) class distribution — it must equal the natural extracted
  distribution (untouched),
* a patient-disjointness confirmation between the two folds.

Use ``--subset`` for a quick 2-records-per-fold smoke run.

Usage:
    python -m scripts.check_dataloader [--subset] [--batch-size 32]
"""

from __future__ import annotations

import argparse

from paper1_echofusenet.data import AAMI_CLASSES
from paper1_echofusenet.data.beats import WINDOW_AFTER, WINDOW_BEFORE, build_split, class_counts
from paper1_echofusenet.data.dataset import build_dataloaders


def _fmt(counts: dict) -> str:
    return "  ".join(f"{c}:{counts.get(c, 0)}" for c in AAMI_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", action="store_true", help="2 records per fold")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_records = (101, 106) if args.subset else None
    test_records = (100, 103) if args.subset else None

    # Natural distributions (pre-oversampling) for reference.
    raw_train, raw_test = build_split(
        train_records=train_records, test_records=test_records
    )
    raw_train_counts = class_counts(raw_train)
    raw_test_counts = class_counts(raw_test)

    train_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        train_records=train_records,
        test_records=test_records,
        oversample=True,
    )
    train_beats = train_loader.dataset.beats
    test_beats = test_loader.dataset.beats

    L = WINDOW_BEFORE + WINDOW_AFTER
    print(f"Beat window: {L} samples  ->  transforms produce {L}x{L} images\n")

    # --- batch shapes ---
    rp, gaf, mtf, labels = next(iter(train_loader))
    print("Train batch shapes:")
    print(f"  RP  {tuple(rp.shape)}   GAF {tuple(gaf.shape)}   MTF {tuple(mtf.shape)}")
    print(f"  labels {tuple(labels.shape)}  range [{int(labels.min())}, {int(labels.max())}]")
    print(f"  channel value range: [{float(rp.min()):.2f}, {float(rp.max()):.2f}] (normalized)\n")

    # --- train balance before/after oversampling ---
    print("DS1 (train) class distribution:")
    print(f"  before oversampling: {_fmt(raw_train_counts)}  (total {len(raw_train)})")
    print(f"  after  oversampling: {_fmt(class_counts(train_beats))}  (total {len(train_beats)})")
    after = set(class_counts(train_beats).values())
    print(f"  -> {'BALANCED' if len(after) == 1 else 'NOT balanced'}\n")

    # --- test untouched ---
    print("DS2 (test) class distribution:")
    print(f"  natural extracted:   {_fmt(raw_test_counts)}  (total {len(raw_test)})")
    print(f"  in test loader:      {_fmt(class_counts(test_beats))}  (total {len(test_beats)})")
    untouched = class_counts(test_beats) == raw_test_counts
    print(f"  -> {'UNTOUCHED (natural distribution)' if untouched else 'MODIFIED — BUG'}\n")

    # --- leakage ---
    train_ids = {b.record_id for b in train_beats}
    test_ids = {b.record_id for b in test_beats}
    disjoint = train_ids.isdisjoint(test_ids)
    print(f"Patient folds disjoint: {disjoint}  "
          f"(train {len(train_ids)} patients, test {len(test_ids)} patients)")


if __name__ == "__main__":
    main()
