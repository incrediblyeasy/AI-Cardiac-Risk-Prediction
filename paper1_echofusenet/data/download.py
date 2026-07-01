"""Download the MIT-BIH Arrhythmia Database via wfdb (PhysioNet).

Usage:
    python -m paper1_echofusenet.data.download                # default dest
    python -m paper1_echofusenet.data.download --dest some/dir

Data goes to ``data/raw/mitdb`` by default and is git-ignored. The download is
idempotent: records already present are skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import wfdb

from .splits import ALL_MITDB_RECORDS

# PhysioNet slug for the MIT-BIH Arrhythmia Database.
PHYSIONET_DB = "mitdb"

# Repo-root-relative default destination.
DEFAULT_DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "mitdb"

# Files that make up one WFDB record.
_RECORD_EXTS = (".dat", ".hea", ".atr")


def _record_present(dest: Path, record: int) -> bool:
    return all((dest / f"{record}{ext}").exists() for ext in _RECORD_EXTS)


def download(dest: Path = DEFAULT_DEST, records: tuple[int, ...] = ALL_MITDB_RECORDS) -> Path:
    """Download the requested MIT-BIH records into ``dest``. Returns ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    to_fetch = [str(r) for r in records if not _record_present(dest, r)]

    if not to_fetch:
        print(f"All {len(records)} records already present in {dest}")
        return dest

    print(f"Downloading {len(to_fetch)} record(s) to {dest} ...")
    wfdb.dl_database(PHYSIONET_DB, dl_dir=str(dest), records=to_fetch)
    print("Done.")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MIT-BIH Arrhythmia DB")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    download(args.dest)


if __name__ == "__main__":
    main()
