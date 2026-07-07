"""Export EchoFuseNet to ONNX, check runtime parity, and quantize (§8).

EchoFuseNet's ``forward`` takes the three images in ``(rp, gaf, mtf)`` order, so
the ONNX graph is exported with three named inputs and a dynamic batch axis on
each (edge inference is one beat at a time, but a batch axis keeps the graph
reusable). ``verify_parity`` then runs the exported graph under ONNX Runtime and
compares logits against the PyTorch model — an export that silently diverges is
worse than none, so this guard is the point of the module.

``onnx`` / ``onnxruntime`` are optional; they are imported lazily so importing
this module never fails on a machine without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..benchmark import measure_model_size_mb
from ..models import EchoFuseNet, count_parameters

DEFAULT_IMAGE_SIZE: int = 256
DEFAULT_OPSET: int = 17
INPUT_NAMES = ["rp", "gaf", "mtf"]
OUTPUT_NAMES = ["logits"]


def _require(module: str):
    """Import an optional dependency or raise an actionable install hint."""
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            f"'{module}' is required for ONNX deployment. Install it with "
            f"`pip install onnx onnxruntime` or `pip install -e '.[deployment]'`."
        ) from exc


def _dummy_inputs(batch: int, image_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.rand(batch, 1, image_size, image_size)
    return x, x.clone(), x.clone()


def export_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    opset: int = DEFAULT_OPSET,
    dynamic_batch: bool = True,
) -> Path:
    """Export ``model`` to an ONNX file at ``path`` (eval mode, batch axis dynamic).

    Returns the written path. Uses a batch-2 dummy input so BatchNorm layers trace
    with running statistics (eval mode), then marks the batch axis dynamic so any
    batch size runs at inference.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    dummy = _dummy_inputs(2, image_size)
    dynamic_axes = (
        {name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES}
        if dynamic_batch
        else None
    )
    kwargs = dict(
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    # Use the legacy TorchScript exporter: torch>=2.5 defaults to the dynamo
    # exporter, whose graphs onnxruntime's dynamic quantizer cannot version-convert
    # (it trips over Squeeze/Unsqueeze axes). ``dynamo`` is unknown to torch<2.5,
    # so fall back to the plain call there (legacy is already the default).
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # pragma: no cover - only on torch<2.5
        torch.onnx.export(model, dummy, str(path), **kwargs)
    return path


def onnx_file_size_mb(path: str | Path) -> float:
    """On-disk size of an ONNX file in MiB."""
    return Path(path).stat().st_size / (1024 * 1024)


@dataclass
class ParityResult:
    """Numerical agreement between the PyTorch model and its ONNX export."""

    max_abs_diff: float
    atol: float
    rtol: float

    @property
    def ok(self) -> bool:
        return self.max_abs_diff <= self.atol

    def __str__(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        return f"parity max|diff|={self.max_abs_diff:.2e} (atol={self.atol:.0e}) -> {verdict}"


def verify_parity(
    model: nn.Module,
    onnx_path: str | Path,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch: int = 4,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    seed: int = 0,
) -> ParityResult:
    """Compare PyTorch logits to ONNX Runtime logits on shared random inputs."""
    ort = _require("onnxruntime")
    torch.manual_seed(seed)
    rp, gaf, mtf = _dummy_inputs(batch, image_size)

    model = model.eval()
    with torch.no_grad():
        torch_out = model(rp, gaf, mtf).cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(
        OUTPUT_NAMES,
        {"rp": rp.numpy(), "gaf": gaf.numpy(), "mtf": mtf.numpy()},
    )[0]

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    return ParityResult(max_abs_diff=max_abs_diff, atol=atol, rtol=rtol)


def quantize_dynamic_onnx(src: str | Path, dst: str | Path) -> Path:
    """Int8 dynamic-quantize an ONNX model (weights only) — the edge size win."""
    _require("onnx")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


@dataclass
class DeploymentReport:
    """Sizes + parity for the full export→quantize pipeline."""

    n_params: int
    torch_size_mb: float
    onnx_size_mb: float
    quantized_size_mb: float | None
    parity: ParityResult

    @property
    def quantization_ratio(self) -> float | None:
        """ONNX-to-quantized size ratio (higher = bigger win); None if not run."""
        if self.quantized_size_mb is None or self.quantized_size_mb == 0:
            return None
        return self.onnx_size_mb / self.quantized_size_mb

    def format(self) -> str:
        lines = [
            f"Parameters      : {self.n_params:,}",
            f"PyTorch size    : {self.torch_size_mb:.3f} MB",
            f"ONNX size       : {self.onnx_size_mb:.3f} MB",
        ]
        if self.quantized_size_mb is not None:
            ratio = self.quantization_ratio
            lines.append(
                f"Quantized (int8): {self.quantized_size_mb:.3f} MB"
                + (f"   ({ratio:.2f}x smaller)" if ratio else "")
            )
        lines.append(f"ONNX parity     : {self.parity}")
        return "\n".join(lines)

    def summary_dict(self) -> dict:
        return {
            "n_params": self.n_params,
            "torch_size_mb": self.torch_size_mb,
            "onnx_size_mb": self.onnx_size_mb,
            "quantized_size_mb": self.quantized_size_mb,
            "quantization_ratio": self.quantization_ratio,
            "parity": {"max_abs_diff": self.parity.max_abs_diff, "ok": self.parity.ok},
        }


def export_and_report(
    model: nn.Module | None = None,
    out_dir: str | Path = "runs/deployment",
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    quantize: bool = True,
    opset: int = DEFAULT_OPSET,
    atol: float = 1e-4,
) -> DeploymentReport:
    """Export → verify parity → (optionally) quantize, and return a report.

    With ``model=None`` a fresh full EchoFuseNet is exported (useful for a smoke
    check); in a real run pass the trained checkpoint's model.
    """
    if model is None:
        model = EchoFuseNet()
    out_dir = Path(out_dir)
    onnx_path = export_onnx(model, out_dir / "echofusenet.onnx", image_size=image_size, opset=opset)
    parity = verify_parity(model, onnx_path, image_size=image_size, atol=atol)

    quantized_mb: float | None = None
    if quantize:
        q_path = quantize_dynamic_onnx(onnx_path, out_dir / "echofusenet.int8.onnx")
        quantized_mb = onnx_file_size_mb(q_path)

    return DeploymentReport(
        n_params=count_parameters(model),
        torch_size_mb=measure_model_size_mb(model),
        onnx_size_mb=onnx_file_size_mb(onnx_path),
        quantized_size_mb=quantized_mb,
        parity=parity,
    )
