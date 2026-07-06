"""Per-run experiment-metadata logging (§8 code-quality enhancement).

Reproducibility needs more than a saved config: to reconstruct *exactly* what
produced a number in the paper you also need the **code version**, the **seed**,
the **environment**, and the **final metrics**, captured together in one place.
This utility writes that record once per run so every training entrypoint
(``train``, ``crossval``, ``compare_imbalance``, the Optuna objective, …) can call
one function instead of each re-implementing ad-hoc logging.

The checklist explicitly folds the §3 "log every trial's config + result" item
into this single utility — build it once, call it everywhere — rather than
duplicating metadata logging per script.

A record captures:

* ``git_commit`` + ``git_dirty`` — the exact code state (dirty flag warns that
  uncommitted changes were present, so the commit alone may not reproduce it);
* ``seed`` and the full resolved ``config`` dict;
* ``metrics`` — the run's final/headline numbers;
* ``python`` / ``torch`` / ``platform`` and a UTC ``timestamp`` for provenance;
* free-form ``extra`` (e.g. dataset records, notes).

Records are appended as one JSON object per line to a shared ``experiments.jsonl``
ledger *and* written as a standalone ``experiment.json`` in the run directory, so
you get both a per-run artifact and a greppable cross-run history.

No hard dependency on torch: the torch version is captured opportunistically and
omitted if torch is not importable, keeping this reusable from any context.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_commit_hash(short: bool = False, cwd: str | Path | None = None) -> str | None:
    """Current HEAD commit hash, or ``None`` outside a git repo / on error."""
    args = ["git", "rev-parse", *(["--short", "HEAD"] if short else ["HEAD"])]
    try:
        out = subprocess.run(
            args, cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def git_is_dirty(cwd: str | Path | None = None) -> bool | None:
    """True if the working tree has uncommitted changes; None if git unavailable."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except Exception:  # torch optional — provenance is best-effort
        return None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ExperimentRecord:
    """A single reproducibility record for one training/eval run."""

    name: str
    seed: int
    config: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    git_commit: str | None = None
    git_dirty: bool | None = None
    python: str = field(default_factory=lambda: sys.version.split()[0])
    torch: str | None = field(default_factory=_torch_version)
    platform: str = field(default_factory=platform.platform)
    timestamp: str = field(default_factory=_utc_timestamp)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_experiment(
    name: str,
    seed: int,
    config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
    ledger: str | Path | None = None,
    extra: dict[str, Any] | None = None,
    repo_dir: str | Path | None = None,
) -> ExperimentRecord:
    """Build and persist an experiment record.

    Writes ``experiment.json`` into ``out_dir`` (when given) and appends the same
    record as a line to ``ledger`` (defaults to ``experiments.jsonl`` in the
    current directory, or in ``out_dir``'s parent when ``out_dir`` is set).
    Returns the record so callers can inspect / augment it.

    ``config`` should be the *resolved* config dict (e.g. ``TrainConfig.to_dict``)
    so the record is self-contained. ``repo_dir`` locates the git repo for the
    commit hash (defaults to ``out_dir`` or the cwd).
    """
    probe_dir = repo_dir or out_dir
    record = ExperimentRecord(
        name=name,
        seed=seed,
        config=config,
        metrics=metrics or {},
        git_commit=git_commit_hash(cwd=probe_dir),
        git_dirty=git_is_dirty(cwd=probe_dir),
        extra=extra or {},
    )
    payload = record.to_dict()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "experiment.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")

    if ledger is None:
        ledger = (Path(out_dir).parent / "experiments.jsonl") if out_dir else Path("experiments.jsonl")
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")

    return record
