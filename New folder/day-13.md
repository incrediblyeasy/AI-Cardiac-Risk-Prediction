# Day 13 — Latency & Size Benchmark

## Goal
Confirm the model is actually edge-deployable, not just parameter-light on paper.

## Tasks
- [x] Build a CPU inference latency benchmark script
- [x] Measure exported model size on disk — **2.596 MB** (real)
- [x] Assert against thresholds: <15ms CPU latency, <3MB model size — **both PASS** (10.44 ms 4-thread / 9.48 ms 1-thread; 2.596 MB)
- [x] If failing either threshold, profile and optimize before moving on — *both thresholds pass; no optimization needed*

## Deliverable / Definition of Done
- Benchmark report confirming (or fixing to confirm) both thresholds

## Dependencies
Day 8 (final assembled model).

---

## Daily Update (fill in when done)

**Date completed:** 2026-07-03

**Status:** ✅ Done — both thresholds PASS

**Measured (idle CPU, best.pt, batch-1, 200 iters):**

| metric | threshold | 4-thread | 1-thread (edge) | verdict |
|---|---|---|---|---|
| exported size | < 3 MB | 2.596 MB | 2.596 MB | ✅ PASS |
| median latency | < 15 ms | 10.44 ms | 9.48 ms | ✅ PASS |
| p95 latency | — | 13.05 ms | 14.06 ms | (under 15) |

Artifacts: `runs/ds1ds2_baseline/benchmark_{4,1}thread.json`.

**What I completed:**
- `paper1_echofusenet/benchmark.py` — deployability harness:
  - `measure_model_size_mb()` — serialised `state_dict` size (deployment weights
    only, no optimizer). **Full model = 2.596 MB < 3 MB → PASS** (deterministic,
    CPU-load-independent).
  - `measure_latency_ms()` — single-beat (batch-1) forward latency with warm-up,
    `torch.no_grad`, pinnable `num_threads`; reports mean/median/p95/min and
    restores the global thread count afterwards (no side effects).
  - `benchmark_model()` → `BenchmarkResult` with pass/fail per threshold and a
    `format()` report + `summary_dict()`.
- `scripts/benchmark.py` — CLI (`python -m scripts.benchmark [--threads N --iters M
  --checkpoint best.pt --json out.json]`); exits non-zero on any threshold fail so
  it can gate CI.
- `tests/test_benchmark.py` — 5 tests: real size < 3 MB, size scales with
  modalities, latency-stats structure/orderings, thread-count restoration,
  result fields. All green.

**Blockers / issues:**
- **Latency must be measured on an idle CPU.** The Day-10 training run is
  currently saturating all cores, and repeated test runs contend too, so any
  latency number taken now would be inflated. Plan: run
  `python -m scripts.benchmark --iters 200 --threads 4` once the CPU is free
  (after Day-10), record median vs the 15 ms threshold, and profile/optimize only
  if it fails. Size already passes independent of this.

**Notes for next day:**
- If latency fails on this CPU: try `--threads 1` and `--threads 4` (edge-realistic),
  then TorchScript (`torch.jit.script`) and/or dynamic int8 quantization
  (`torch.ao.quantization.quantize_dynamic`) — the harness accepts any nn.Module,
  so quantized/scripted variants drop straight in.
- Batch-1 on CPU has poor utilization; median (not mean) is the honest edge number.

**Time spent:** ~1.5h (harness; latency run pending)
