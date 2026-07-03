"""EchoFuseNet model components.

Day 7: a single lightweight depthwise-separable CNN branch (`CNNBranch`) that
encodes one modality (RP/GAF/MTF) into an embedding. Day 8: `EchoFuseNet` — the
three-branch late-fusion classifier assembled from those branches.
"""

from .branch import CNNBranch, DepthwiseSeparableConv, count_parameters
from .echofusenet import CANONICAL_MODALITIES, EchoFuseNet

__all__ = [
    "CNNBranch",
    "DepthwiseSeparableConv",
    "count_parameters",
    "EchoFuseNet",
    "CANONICAL_MODALITIES",
]
