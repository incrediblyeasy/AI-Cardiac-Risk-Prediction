"""Data acquisition, splitting, and labeling for the MIT-BIH Arrhythmia DB."""

from .aami import AAMI_CLASSES, SYMBOL_TO_AAMI, class_index, symbol_to_aami
from .beats import (
    WINDOW_AFTER,
    WINDOW_BEFORE,
    BeatSegment,
    build_split,
    class_counts,
    extract_beats,
    load_fold,
)
from .dataset import (
    CHANNEL_NAMES,
    MultimodalBeatDataset,
    beat_to_channels,
    build_dataloaders,
    oversample_beats,
    oversample_indices,
)
from .mitbih import Beat, Record, class_distribution, load_record
from .splits import (
    DS1_PATIENTS,
    DS2_PATIENTS,
    EXCLUDED_PACED,
    assert_no_leakage,
    assert_patient_disjoint,
    records_for_fold,
    split_for,
)

__all__ = [
    # AAMI
    "AAMI_CLASSES",
    "SYMBOL_TO_AAMI",
    "class_index",
    "symbol_to_aami",
    # splits
    "DS1_PATIENTS",
    "DS2_PATIENTS",
    "EXCLUDED_PACED",
    "assert_no_leakage",
    "assert_patient_disjoint",
    "records_for_fold",
    "split_for",
    # record loading
    "Beat",
    "Record",
    "load_record",
    "class_distribution",
    # beat extraction
    "BeatSegment",
    "WINDOW_BEFORE",
    "WINDOW_AFTER",
    "extract_beats",
    "load_fold",
    "build_split",
    "class_counts",
    # multimodal dataset / dataloader
    "CHANNEL_NAMES",
    "MultimodalBeatDataset",
    "beat_to_channels",
    "build_dataloaders",
    "oversample_beats",
    "oversample_indices",
]
