"""Torchvision baseline backbones adapted to EchoFuseNet's inputs (§6).

Each baseline is wrapped in :class:`BaselineClassifier`, which:

1. stacks the three ``(B, 1, L, L)`` RP/GAF/MTF images into one ``(B, 3, L, L)``
   tensor — the natural 3-channel input a standard ImageNet backbone expects; and
2. exposes ``forward(rp, gaf, mtf)`` so the wrapped model is a drop-in for the
   EchoFuseNet training / evaluation / benchmark code (same protocol, same split).

Backbones are built **from scratch** (``weights=None``) by default: ImageNet
pretraining on natural images is a different, arguably unfair advantage over a
from-scratch EchoFuseNet, and the paper's comparison is about architecture size
vs. accuracy on ECG images, not transfer learning. Pretraining can be toggled on
for a separate reported row.
"""

from __future__ import annotations

import torch
from torch import nn

# Registry of supported architectures -> the torchvision constructor name and the
# attribute path of its final classifier layer (so we can re-init it to n_classes).
BASELINE_ARCHITECTURES: dict[str, dict] = {
    "resnet18": {"ctor": "resnet18", "head": "fc"},
    "resnet50": {"ctor": "resnet50", "head": "fc"},
    "densenet121": {"ctor": "densenet121", "head": "classifier"},
    "efficientnet_b0": {"ctor": "efficientnet_b0", "head": "classifier.-1"},
    "convnext_tiny": {"ctor": "convnext_tiny", "head": "classifier.-1"},
    "vit_b_16": {"ctor": "vit_b_16", "head": "heads.head"},
}


def _require_torchvision():
    try:
        import torchvision

        return torchvision
    except ImportError as exc:  # pragma: no cover - only without torchvision
        raise ImportError(
            "torchvision is required for the §6 baselines. Install it with "
            "`pip install torchvision` or `pip install -e '.[baselines]'`."
        ) from exc


def _replace_head(backbone: nn.Module, head_path: str, n_classes: int) -> None:
    """Re-initialise a backbone's classifier layer to output ``n_classes``.

    ``head_path`` is a dotted path (supporting negative Sequential indices, e.g.
    ``classifier.-1``) to the final ``nn.Linear``; it is swapped for a fresh
    ``Linear(in_features, n_classes)``.
    """
    parts = head_path.split(".")
    parent = backbone
    for p in parts[:-1]:
        parent = parent[int(p)] if p.lstrip("-").isdigit() else getattr(parent, p)
    last = parts[-1]
    module = parent[int(last)] if last.lstrip("-").isdigit() else getattr(parent, last)
    new = nn.Linear(module.in_features, n_classes)
    if last.lstrip("-").isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


class BaselineClassifier(nn.Module):
    """Adapter: (rp, gaf, mtf) 3x single-channel -> 3-channel backbone -> logits.

    ``vit_b_16`` requires a fixed 224x224 input, so this adapter resizes the
    stacked image to ``input_size`` when set (bilinear); other CNN backbones
    accept the native beat-image size and leave it untouched.
    """

    def __init__(
        self, backbone: nn.Module, input_size: int | None = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size

    def forward(
        self, rp: torch.Tensor, gaf: torch.Tensor, mtf: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([rp, gaf, mtf], dim=1)  # (B, 3, L, L)
        if self.input_size is not None and x.shape[-1] != self.input_size:
            x = nn.functional.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )
        return self.backbone(x)


def build_baseline(
    name: str,
    n_classes: int = 5,
    pretrained: bool = False,
) -> BaselineClassifier:
    """Build a baseline classifier by name (see ``BASELINE_ARCHITECTURES``)."""
    if name not in BASELINE_ARCHITECTURES:
        raise ValueError(
            f"unknown baseline '{name}'; choose from {sorted(BASELINE_ARCHITECTURES)}"
        )
    tv = _require_torchvision()
    spec = BASELINE_ARCHITECTURES[name]
    ctor = getattr(tv.models, spec["ctor"])
    backbone = ctor(weights="DEFAULT" if pretrained else None)
    _replace_head(backbone, spec["head"], n_classes)
    # ViT needs 224x224; CNNs run at native resolution.
    input_size = 224 if name == "vit_b_16" else None
    return BaselineClassifier(backbone, input_size=input_size)
