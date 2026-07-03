"""Deployability benchmark: size threshold (deterministic) + latency harness."""

import torch

from paper1_echofusenet.benchmark import (
    SIZE_THRESHOLD_MB,
    LatencyStats,
    benchmark_model,
    measure_latency_ms,
    measure_model_size_mb,
)
from paper1_echofusenet.models import EchoFuseNet


def test_full_model_size_under_threshold():
    # Deterministic (CPU-load-independent): the assembled model must serialise
    # to under the 3 MB deployment budget.
    size = measure_model_size_mb(EchoFuseNet())
    assert size < SIZE_THRESHOLD_MB, f"{size:.3f} MB exceeds {SIZE_THRESHOLD_MB} MB"
    assert size > 0.5  # sanity floor — it's not an empty file


def test_size_scales_with_modalities():
    full = measure_model_size_mb(EchoFuseNet())
    single = measure_model_size_mb(EchoFuseNet(modalities=("rp",)))
    assert single < full


def test_latency_harness_returns_stats():
    # Tiny model + few iters so the test is fast and load-independent; we assert
    # structure/orderings, not the 15 ms threshold (that's for an idle CPU run).
    model = EchoFuseNet(widths=(8, 16, 16), fusion_hidden=16)
    stats = measure_latency_ms(model, batch=1, n_warmup=2, n_iter=5, num_threads=1)
    assert isinstance(stats, LatencyStats)
    assert stats.n_iter == 5
    assert stats.min > 0
    assert stats.min <= stats.median <= stats.p95
    assert stats.mean > 0


def test_measure_latency_restores_thread_count():
    before = torch.get_num_threads()
    model = EchoFuseNet(widths=(8, 16, 16), fusion_hidden=16)
    measure_latency_ms(model, n_warmup=1, n_iter=2, num_threads=1)
    assert torch.get_num_threads() == before  # no global side effect


def test_benchmark_model_result_fields():
    model = EchoFuseNet(widths=(8, 16, 16), fusion_hidden=16)
    result = benchmark_model(model, n_warmup=1, n_iter=3, num_threads=1)
    assert result.size_ok  # tiny model is well under budget
    assert result.n_params > 0
    d = result.summary_dict()
    assert "latency_ms" in d and "size_mb" in d and "passed" in d
