"""Disk-backed, size-capped cache for RP/GAF/MTF transforms.

Why disk, not an in-memory dict: with ``num_workers > 0`` (needed for real
speedup — see ``echofusenet_ds1ds2_baseline.json``'s ``num_workers: 0``,
the actual bottleneck), PyTorch's DataLoader runs separate worker
*processes*. An in-memory cache attribute on the Dataset object is not
shared across those processes, so it silently wouldn't help in the
multi-worker case this is meant to speed up. A disk cache is naturally
shared across worker processes, at the cost of disk I/O instead of a
dict lookup — still far cheaper than recomputing RP/GAF/MTF from scratch.

Why size-capped, not "cache everything": DS1 alone is 51,000 beats;
RP+GAF+MTF at 256x256 float16 for all of them is ~18.7 GB — too close to
Kaggle's typical ~20GB /kaggle/working budget to safely promise as
"just cache it all". This cache instead enforces a hard byte budget,
checked before every write, and simply stops caching (falls back to
recompute-on-the-fly) once the budget is reached. It can never overflow
the disk budget you give it, by construction — not by hoping the data
happens to fit.

Cache key: ``(record_id, r_peak)`` — the R-peak position within its source
record is a stable identity for "this exact heartbeat", so oversampled
duplicates of the same beat (common in the training fold) share one cache
entry instead of each being computed and stored separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import numpy as np


class DiskTransformCache:
    """Size-capped, disk-backed cache for (rp, gaf, mtf) beat transforms.

    Safe by construction: tracks cumulative bytes written and refuses new
    writes once ``max_bytes`` is reached, rather than writing until the disk
    itself fills up. Existing entries are always served from cache even
    after the cap is hit — only *new* entries stop being cached.
    """

    def __init__(self, cache_dir: str | Path, max_bytes: int = 4 * 1024**3):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_bytes)
        self._index_path = self.cache_dir / "_index.json"
        self._lock = Lock()  # guards the in-process byte counter only
        self._bytes_written = self._load_existing_size()
        self._budget_exceeded_warned = False

    def _load_existing_size(self) -> int:
        """Sum of bytes already on disk from a previous run (cache survives
        across script invocations, e.g. across CV folds run in sequence)."""
        total = 0
        for f in self.cache_dir.glob("*.npz"):
            total += f.stat().st_size
        return total

    @staticmethod
    def _key_to_filename(record_id: int, r_peak: int) -> str:
        return f"{record_id}_{r_peak}.npz"

    def get(self, record_id: int, r_peak: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return cached ``(rp, gaf, mtf)`` as float32, or None on a miss."""
        path = self.cache_dir / self._key_to_filename(record_id, r_peak)
        if not path.exists():
            return None
        try:
            with np.load(path) as data:
                return (
                    data["rp"].astype(np.float32),
                    data["gaf"].astype(np.float32),
                    data["mtf"].astype(np.float32),
                )
        except (OSError, ValueError, KeyError):
            # Corrupt/partial file (e.g. a crash mid-write) — treat as a
            # miss and let the caller recompute, rather than raising.
            return None

    def put(self, record_id: int, r_peak: int, rp: np.ndarray, gaf: np.ndarray, mtf: np.ndarray) -> bool:
        """Cache ``(rp, gaf, mtf)`` as float16 if budget allows. Returns
        whether it was actually written (False once the budget is hit).

        Writes to a temp file first and checks its *actual* on-disk size
        before committing — this is what makes the budget a real hard cap
        rather than an estimate that could quietly overshoot by a file or
        two (npz's zip-container overhead makes exact pre-write sizing
        unreliable; checking the real written size avoids that entirely).
        """
        with self._lock:
            if self._bytes_written >= self.max_bytes:
                if not self._budget_exceeded_warned:
                    print(
                        f"  [transform cache: {self.max_bytes / 1024**3:.2f} GB budget "
                        f"reached — further beats will be recomputed on the fly, "
                        f"not cached. Raise max_bytes if you have more disk headroom.]"
                    )
                    self._budget_exceeded_warned = True
                return False

        path = self.cache_dir / self._key_to_filename(record_id, r_peak)
        if path.exists():
            return True  # already cached (e.g. by another worker process)

        tmp_path = self.cache_dir / f"{record_id}_{r_peak}.tmp.npz"
        np.savez(
            tmp_path,
            rp=rp.astype(np.float16),
            gaf=gaf.astype(np.float16),
            mtf=mtf.astype(np.float16),
        )
        size = tmp_path.stat().st_size

        with self._lock:
            if self._bytes_written + size > self.max_bytes:
                tmp_path.unlink(missing_ok=True)
                if not self._budget_exceeded_warned:
                    print(
                        f"  [transform cache: {self.max_bytes / 1024**3:.2f} GB budget "
                        f"reached — further beats will be recomputed on the fly, "
                        f"not cached. Raise max_bytes if you have more disk headroom.]"
                    )
                    self._budget_exceeded_warned = True
                return False
            tmp_path.rename(path)
            self._bytes_written += size
        return True

    def stats(self) -> dict:
        """Current cache occupancy — call after a run to see how much of the
        budget got used and whether raising it would help."""
        n_files = sum(1 for _ in self.cache_dir.glob("*.npz"))
        return {
            "bytes_written": self._bytes_written,
            "gb_written": self._bytes_written / 1024**3,
            "max_bytes": self.max_bytes,
            "max_gb": self.max_bytes / 1024**3,
            "n_cached_beats": n_files,
            "budget_exhausted": self._bytes_written >= self.max_bytes,
        }
