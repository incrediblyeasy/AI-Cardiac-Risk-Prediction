"""EchoFuseNet model components.

Day 7: a single lightweight depthwise-separable CNN branch (`CNNBranch`) that
encodes one modality (RP/GAF/MTF) into an embedding. Day 8 adds the other two
branches and the late-fusion classifier.
"""

from .branch import CNNBranch, DepthwiseSeparableConv, count_parameters

__all__ = ["CNNBranch", "DepthwiseSeparableConv", "count_parameters"]
