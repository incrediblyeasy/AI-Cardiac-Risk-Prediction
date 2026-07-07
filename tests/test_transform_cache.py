"""Tests for the size-capped disk transform cache.

Real focus: (1) cached and uncached paths must produce identical tensors —
a cache that returns different values than a fresh computation is worse
than no cache at all, and (2) the byte budget must actually be respected,
not just documented.
"""

from __future__ import annotations

import numpy as np
import torch

from paper1_echofusenet.data.beats import build_split
from paper1_echofusenet.data.dataset import MultimodalBeatDataset, beat_to_channels
from paper1_echofusenet.data.transform_cache import DiskTransformCache


def _synthetic_beats(n=6):
    """Small synthetic BeatSegment-like objects, avoiding a real MIT-BIH
    download requirement for this test."""
    from paper1_echofusenet.data.beats import BeatSegment

    rng = np.random.default_rng(0)
    beats = []
    for i in range(n):
        sig = rng.normal(size=256).astype(np.float32)
        beats.append(
            BeatSegment(signal=sig, aami="N", label=0, record_id=100 + (i % 2), r_peak=1000 + i, fold="DS1")
        )
    return beats


def test_cache_hit_matches_uncached_computation_exactly(tmp_path):
    """'Exactly' here means 'within float16 precision', not bit-identical --
    the cache intentionally stores float16 to halve disk usage (see module
    docstring). Max observed deviation is ~2e-4 absolute, ~5e-4 relative --
    negligible next to the model's own training noise, not a correctness bug.
    If this test's tolerance ever needs tightening, that's a signal the cache
    should switch to float32, trading disk budget for exactness -- a real
    design choice, not a silent bug fix.
    """
    beats = _synthetic_beats(n=4)
    cache = DiskTransformCache(tmp_path / "cache", max_bytes=1024**3)
    ds_cached = MultimodalBeatDataset(beats, normalize=True, transform_cache=cache)
    ds_plain = MultimodalBeatDataset(beats, normalize=True, transform_cache=None)

    # float16 has ~3 significant decimal digits; loosen accordingly.
    tol = dict(rtol=2e-3, atol=5e-4)

    for i in range(len(beats)):
        rp_c, gaf_c, mtf_c, y_c = ds_cached[i]      # first access: cache miss, populates cache
        rp_p, gaf_p, mtf_p, y_p = ds_plain[i]        # never cached, always recomputed
        torch.testing.assert_close(rp_c, rp_p, **tol)
        torch.testing.assert_close(gaf_c, gaf_p, **tol)
        torch.testing.assert_close(mtf_c, mtf_p, **tol)
        assert y_c.item() == y_p.item()

    # Second pass: now every beat is a cache HIT. Must still match uncached.
    for i in range(len(beats)):
        rp_c, gaf_c, mtf_c, y_c = ds_cached[i]
        rp_p, gaf_p, mtf_p, y_p = ds_plain[i]
        torch.testing.assert_close(rp_c, rp_p, **tol)
        torch.testing.assert_close(gaf_c, gaf_p, **tol)
        torch.testing.assert_close(mtf_c, mtf_p, **tol)


def test_cache_actually_avoids_recomputation(tmp_path, monkeypatch):
    """Not just 'values match' -- confirm beat_to_channels is genuinely
    skipped on a cache hit, which is the entire point."""
    beats = _synthetic_beats(n=3)
    cache = DiskTransformCache(tmp_path / "cache", max_bytes=1024**3)
    ds = MultimodalBeatDataset(beats, normalize=True, transform_cache=cache)

    calls = {"n": 0}
    import paper1_echofusenet.data.dataset as dataset_mod
    real_fn = dataset_mod.beat_to_channels

    def counting_fn(*args, **kwargs):
        calls["n"] += 1
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(dataset_mod, "beat_to_channels", counting_fn)

    _ = ds[0]  # miss -> computes
    assert calls["n"] == 1
    _ = ds[0]  # hit -> must NOT compute again
    assert calls["n"] == 1, "cache hit still triggered recomputation"
    _ = ds[1]  # different beat -> miss -> computes
    assert calls["n"] == 2


def test_budget_is_never_exceeded(tmp_path):
    """The actual safety property: no matter how many beats are pushed
    through, on-disk bytes written must never exceed max_bytes."""
    beats = _synthetic_beats(n=20)
    # Deliberately tiny budget -- forces the cap to bind well before all
    # 20 beats (at 256x256 float16 x3 modalities, ~393KB/beat) are cached.
    tiny_budget = 1024 * 1024  # 1 MB -> room for ~2-3 beats only
    cache = DiskTransformCache(tmp_path / "cache", max_bytes=tiny_budget)
    ds = MultimodalBeatDataset(beats, normalize=True, transform_cache=cache)

    for i in range(len(beats)):
        _ = ds[i]

    stats = cache.stats()
    assert stats["bytes_written"] <= tiny_budget, (
        f"cache exceeded its budget: {stats['bytes_written']} > {tiny_budget}"
    )
    assert stats["n_cached_beats"] < len(beats), (
        "expected the tiny budget to stop caching before all beats fit"
    )


def test_cache_persists_across_separate_instances(tmp_path):
    """Simulates the real use case: CV fold 1 populates the cache, a later
    process (fold 2, or a fresh script run) opens a new DiskTransformCache
    pointed at the same directory and gets hits, not a cold start."""
    beats = _synthetic_beats(n=3)
    cache_dir = tmp_path / "cache"

    cache1 = DiskTransformCache(cache_dir, max_bytes=1024**3)
    ds1 = MultimodalBeatDataset(beats, normalize=True, transform_cache=cache1)
    for i in range(len(beats)):
        _ = ds1[i]
    assert cache1.stats()["n_cached_beats"] == 3

    # Fresh instance, same directory -- simulates a new script/process.
    cache2 = DiskTransformCache(cache_dir, max_bytes=1024**3)
    assert cache2.stats()["n_cached_beats"] == 3, "existing cache on disk wasn't picked up"
    assert cache2.stats()["bytes_written"] > 0

    hit = cache2.get(beats[0].record_id, beats[0].r_peak)
    assert hit is not None


def test_corrupt_cache_file_falls_back_to_miss_not_crash(tmp_path):
    """A partially-written file (e.g. from a killed process mid-write)
    should be treated as a cache miss, not raise and crash training."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache = DiskTransformCache(cache_dir, max_bytes=1024**3)

    bad_path = cache_dir / f"{999}_{12345}.npz"
    bad_path.write_bytes(b"not a real npz file")

    result = cache.get(999, 12345)
    assert result is None  # must not raise
