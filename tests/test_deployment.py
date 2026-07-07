"""§8 deployment: ONNX export, runtime parity, and int8 quantization.

Skips cleanly when the optional onnx/onnxruntime deps are absent (like the
torchvision-gated baseline tests), so the core suite never hard-requires them.
"""

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from paper1_echofusenet.deployment import (  # noqa: E402
    DeploymentReport,
    export_and_report,
    export_onnx,
    onnx_file_size_mb,
    quantize_dynamic_onnx,
    verify_parity,
)
from paper1_echofusenet.models import EchoFuseNet  # noqa: E402

IMG = 32  # small images keep export/parity fast


def _tiny_model():
    return EchoFuseNet(widths=(8, 16, 16), fusion_hidden=16, dropout=0.0)


def test_export_writes_a_loadable_onnx_graph(tmp_path):
    path = export_onnx(_tiny_model(), tmp_path / "m.onnx", image_size=IMG)
    assert path.exists()
    model = onnx.load(str(path))
    onnx.checker.check_model(model)          # structurally valid
    names = {i.name for i in model.graph.input}
    assert {"rp", "gaf", "mtf"} <= names     # three explicit modality inputs
    assert onnx_file_size_mb(path) > 0.0


def test_onnx_runtime_matches_pytorch(tmp_path):
    model = _tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", image_size=IMG)
    parity = verify_parity(model, path, image_size=IMG, batch=4, atol=1e-4)
    assert parity.ok, f"ONNX diverged from PyTorch: {parity}"


def test_export_supports_dynamic_batch(tmp_path):
    model = _tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", image_size=IMG, dynamic_batch=True)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 8):
        x = np.random.default_rng(batch).random((batch, 1, IMG, IMG)).astype(np.float32)
        out = sess.run(["logits"], {"rp": x, "gaf": x.copy(), "mtf": x.copy()})[0]
        assert out.shape == (batch, 5)


def test_dynamic_quantization_shrinks_the_model(tmp_path):
    # Use the full-size EchoFuseNet: int8 dynamic quantization genuinely shrinks a
    # real model (~4x), whereas on a toy model the per-tensor quant-node overhead
    # can outweigh the weight savings. Image size is irrelevant to weight bytes.
    model = EchoFuseNet()
    path = export_onnx(model, tmp_path / "m.onnx", image_size=IMG)
    q_path = quantize_dynamic_onnx(path, tmp_path / "m.int8.onnx")
    assert q_path.exists()
    assert onnx_file_size_mb(q_path) < onnx_file_size_mb(path)


def test_export_and_report_end_to_end(tmp_path):
    report = export_and_report(_tiny_model(), out_dir=tmp_path, image_size=IMG, quantize=True)
    assert isinstance(report, DeploymentReport)
    assert report.n_params > 0
    assert report.onnx_size_mb > 0.0
    assert report.quantized_size_mb is not None
    assert report.parity.ok
    # summary_dict is JSON-friendly and complete.
    d = report.summary_dict()
    assert d["parity"]["ok"] is True
    assert "quantization_ratio" in d
