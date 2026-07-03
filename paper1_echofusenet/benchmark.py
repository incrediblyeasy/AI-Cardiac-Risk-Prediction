"""Edge-deployability benchmark for EchoFuseNet (Day 13).

Confirms the model is genuinely deployable, not just parameter-light on paper, by
measuring the two things that matter on an edge device:

* **CPU inference latency** — single-beat (batch-1) forward time, since edge
  inference is one heartbeat at a time, not batched. Reported as mean / median /
  p95 over many timed iterations after warm-up.
* **Exported model size** — bytes of the serialised ``state_dict`` (deployment
  weights only, no optimizer state), in MiB.

Thresholds (from the Day-13 spec):

    latency  < 15 ms  (CPU, batch 1)
    size     <  3 MB

``benchmark_model`` returns a structured result with pass/fail per threshold so
both a human report (``scripts/benchmark.py``) and a test can consume it.

Note on measurement hygiene: latency is only meaningful on an **idle** CPU — run
it when nothing else is competing for cores, or the numbers inflate.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .models import EchoFuseNet, count_parameters

# Day-13 deployability thresholds.
LATENCY_THRESHOLD_MS: float = 15.0
SIZE_THRESHOLD_MB: float = 3.0

# Default single-beat input: one (1, L, L) image per branch (L = 256 window).
DEFAULT_IMAGE_SIZE: int = 256


def measure_model_size_mb(model: nn.Module) -> float:
    """Serialised ``state_dict`` size in MiB (deployment weights only)."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024 * 1024)


def _make_inputs(batch: int, image_size: int, device: torch.device):
    x = torch.rand(batch, 1, image_size, image_size, device=device)
    return x, x.clone(), x.clone()


@dataclass
class LatencyStats:
    """Per-inference latency distribution in milliseconds."""

    mean: float
    median: float
    p95: float
    std: float
    min: float
    n_iter: int
    batch: int
    num_threads: int

    def __str__(self) -> str:
        return (
            f"mean {self.mean:.2f} ms | median {self.median:.2f} | "
            f"p95 {self.p95:.2f} | min {self.min:.2f} "
            f"(batch {self.batch}, {self.n_iter} iters, {self.num_threads} threads)"
        )


def measure_latency_ms(
    model: nn.Module,
    batch: int = 1,
    image_size: int = DEFAULT_IMAGE_SIZE,
    n_warmup: int = 10,
    n_iter: int = 50,
    device: torch.device | None = None,
    num_threads: int | None = None,
) -> LatencyStats:
    """Time ``model.forward`` over ``n_iter`` runs after ``n_warmup`` warm-ups.

    Uses fresh random inputs and ``torch.no_grad`` in eval mode. ``num_threads``
    pins the intra-op thread count for a reproducible measurement (e.g. 1 or 4
    to mimic an edge CPU); ``None`` leaves Torch's default.
    """
    device = device or torch.device("cpu")
    prev_threads = torch.get_num_threads()
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    try:
        model = model.to(device).eval()
        rp, gaf, mtf = _make_inputs(batch, image_size, device)

        with torch.no_grad():
            for _ in range(n_warmup):
                model(rp, gaf, mtf)

            times_ms: list[float] = []
            for _ in range(n_iter):
                start = time.perf_counter()
                model(rp, gaf, mtf)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        torch.set_num_threads(prev_threads)

    arr = np.asarray(times_ms, dtype=np.float64)
    return LatencyStats(
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
        std=float(arr.std()),
        min=float(arr.min()),
        n_iter=n_iter,
        batch=batch,
        num_threads=torch.get_num_threads() if num_threads is None else num_threads,
    )


@dataclass
class BenchmarkResult:
    """Latency + size measurement with pass/fail against the Day-13 thresholds."""

    n_params: int
    size_mb: float
    latency: LatencyStats
    latency_threshold_ms: float = LATENCY_THRESHOLD_MS
    size_threshold_mb: float = SIZE_THRESHOLD_MB

    @property
    def latency_ok(self) -> bool:
        return self.latency.median <= self.latency_threshold_ms

    @property
    def size_ok(self) -> bool:
        return self.size_mb <= self.size_threshold_mb

    @property
    def passed(self) -> bool:
        return self.latency_ok and self.size_ok

    def format(self) -> str:
        lat_v = "PASS" if self.latency_ok else "FAIL"
        size_v = "PASS" if self.size_ok else "FAIL"
        return "\n".join(
            [
                f"Parameters : {self.n_params:,}",
                f"Model size : {self.size_mb:.3f} MB   "
                f"(threshold < {self.size_threshold_mb} MB)  -> {size_v}",
                f"Latency    : {self.latency}",
                f"           : median {self.latency.median:.2f} ms   "
                f"(threshold < {self.latency_threshold_ms} ms)  -> {lat_v}",
                f"Overall    : {'PASS' if self.passed else 'FAIL'}",
            ]
        )

    def summary_dict(self) -> dict:
        return {
            "n_params": self.n_params,
            "size_mb": self.size_mb,
            "size_ok": self.size_ok,
            "latency_ms": {
                "mean": self.latency.mean,
                "median": self.latency.median,
                "p95": self.latency.p95,
                "min": self.latency.min,
            },
            "latency_ok": self.latency_ok,
            "thresholds": {
                "latency_ms": self.latency_threshold_ms,
                "size_mb": self.size_threshold_mb,
            },
            "passed": self.passed,
        }


def benchmark_model(
    model: nn.Module | None = None,
    batch: int = 1,
    image_size: int = DEFAULT_IMAGE_SIZE,
    n_warmup: int = 10,
    n_iter: int = 50,
    num_threads: int | None = None,
) -> BenchmarkResult:
    """Measure size + latency of a model (default: a fresh full EchoFuseNet)."""
    if model is None:
        model = EchoFuseNet()
    size_mb = measure_model_size_mb(model)
    latency = measure_latency_ms(
        model,
        batch=batch,
        image_size=image_size,
        n_warmup=n_warmup,
        n_iter=n_iter,
        num_threads=num_threads,
    )
    return BenchmarkResult(
        n_params=count_parameters(model),
        size_mb=size_mb,
        latency=latency,
    )
