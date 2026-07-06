"""§8 experiment logging: record fields, artifacts, git-hash arg handling."""

import json

from shared.utils import git_commit_hash, log_experiment
from shared.utils.experiment_log import ExperimentRecord


def test_log_experiment_writes_run_file_and_ledger(tmp_path):
    run_dir = tmp_path / "run"
    rec = log_experiment(
        name="unit",
        seed=7,
        config={"optim": {"lr": 1e-3}},
        metrics={"macro_f1": 0.8},
        out_dir=run_dir,
        extra={"note": "smoke"},
    )
    assert isinstance(rec, ExperimentRecord)
    assert (run_dir / "experiment.json").exists()
    ledger = tmp_path / "experiments.jsonl"          # written to out_dir's parent
    assert ledger.exists()

    payload = json.loads((run_dir / "experiment.json").read_text())
    for key in ("name", "seed", "config", "metrics", "git_commit", "git_dirty",
                "python", "timestamp", "extra"):
        assert key in payload
    assert payload["seed"] == 7
    assert payload["extra"]["note"] == "smoke"

    # Ledger line matches the standalone record.
    line = json.loads(ledger.read_text().strip().splitlines()[-1])
    assert line["name"] == "unit"


def test_git_commit_hash_short_and_full():
    # Run from the repo root: both forms should resolve (or both be None off-repo).
    full = git_commit_hash()
    short = git_commit_hash(short=True)
    if full is not None:
        assert len(short) < len(full)
        assert full.startswith(short)
