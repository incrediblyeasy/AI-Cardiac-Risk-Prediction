"""Latency & size benchmark for EchoFuseNet (Day 13).

Measures single-beat CPU inference latency and exported model size, then checks
them against the edge-deployability thresholds (< 15 ms, < 3 MB). Exits non-zero
if either threshold fails so it can gate CI.

Run on an **idle** machine — a busy CPU inflates latency.

Usage:
    python -m scripts.benchmark
    python -m scripts.benchmark --threads 4 --iters 100
    python -m scripts.benchmark --checkpoint runs/ds1ds2_baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from paper1_echofusenet.benchmark import benchmark_model
from paper1_echofusenet.models import EchoFuseNet


def main() -> None:
    parser = argparse.ArgumentParser(description="EchoFuseNet latency & size benchmark.")
    parser.add_argument("--threads", type=int, default=None, help="Pin intra-op threads.")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=20, help="Warm-up iterations.")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size.")
    parser.add_argument(
        "--checkpoint", default=None, help="Optional best.pt to load real weights."
    )
    parser.add_argument("--json", default=None, help="Write the summary JSON here.")
    args = parser.parse_args()

    model = EchoFuseNet()
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded weights from {args.checkpoint}")

    result = benchmark_model(
        model,
        batch=args.batch,
        n_warmup=args.warmup,
        n_iter=args.iters,
        num_threads=args.threads,
    )

    print("EchoFuseNet edge-deployability benchmark")
    print("=" * 44)
    print(result.format())

    if args.json:
        Path(args.json).write_text(json.dumps(result.summary_dict(), indent=2) + "\n")
        print(f"\nSaved {args.json}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
