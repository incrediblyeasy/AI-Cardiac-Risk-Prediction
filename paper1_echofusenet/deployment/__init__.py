"""§8 deployment — ONNX export, runtime-parity check, and quantization.

The publishable-fixes checklist (§8, "Add deployment realism") asks the edge
story to go beyond a parameter count: export to a portable runtime (ONNX),
prove the export is numerically faithful to the PyTorch model, and report the
size win from int8 dynamic quantization.

``onnx`` / ``onnxruntime`` are **optional** dependencies (the ``[deployment]``
extra). They are imported lazily inside the functions that need them, so the rest
of ``paper1_echofusenet`` imports fine without them; the functions raise a clear
install hint if they are missing.
"""

from __future__ import annotations

from .onnx_export import (
    DeploymentReport,
    ParityResult,
    export_and_report,
    export_onnx,
    onnx_file_size_mb,
    quantize_dynamic_onnx,
    verify_parity,
)

__all__ = [
    "export_onnx",
    "onnx_file_size_mb",
    "verify_parity",
    "quantize_dynamic_onnx",
    "export_and_report",
    "ParityResult",
    "DeploymentReport",
]
