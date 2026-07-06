"""Standard-CNN baselines for the EchoFuseNet comparison table (§6 enhancement).

Framed exactly as the checklist requires: these architectures (ResNet, DenseNet,
EfficientNet, ConvNeXt, ViT) are **comparison-table entries, not candidates to
replace EchoFuseNet**. They are 10-100x larger than the 0.7M-parameter budget, so
the paper's claim is *accuracy-per-parameter and edge-deployability* — "competitive
accuracy at a fraction of the size/latency" — not "we beat ResNet".

The adapter (``BaselineClassifier``) stacks the three RP/GAF/MTF single-channel
images into one 3-channel input and feeds a torchvision backbone. Because it
keeps the ``forward(rp, gaf, mtf)`` signature EchoFuseNet uses, every baseline is
a drop-in for the *same* ``training.train`` / ``training.evaluate`` /
``benchmark`` code — guaranteeing an apples-to-apples comparison on the identical
DS1/DS2 inter-patient split and leakage guard.

torchvision is an **optional** dependency (``[baselines]`` extra); the registry
raises a clear install hint only when a backbone is actually built.
"""

from .models import BASELINE_ARCHITECTURES, BaselineClassifier, build_baseline
from .compare import BaselineRow, compare_baselines

__all__ = [
    "BASELINE_ARCHITECTURES",
    "BaselineClassifier",
    "build_baseline",
    "BaselineRow",
    "compare_baselines",
]
