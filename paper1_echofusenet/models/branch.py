"""Lightweight CNN branch for one EchoFuseNet modality (RP / GAF / MTF).

Each modality (Day 3-5) is a single-channel ``(1, L, L)`` image; one of these
branches encodes it into a fixed-length embedding, and Day 8 fuses the three
embeddings (late fusion) into the classifier.

Design — depthwise separable convolutions
-----------------------------------------
To stay inside the ~0.7M-parameter budget for the *whole* three-branch model,
every spatial convolution is **depthwise separable** (MobileNet-style): a
3x3 depthwise conv (one filter per channel) followed by a 1x1 pointwise conv
that mixes channels. This replaces a standard ``k*k*Cin*Cout`` conv with
``k*k*Cin + Cin*Cout`` parameters — roughly a ``k*k`` reduction — which is what
makes the branch lightweight.

Stem stride-2 conv, then a chain of stride-2 depthwise-separable blocks halves
the spatial size each step (256 -> 128 -> 64 -> 32 -> 16 -> 8) while growing the
channel width, ending in global average pooling to a ``widths[-1]``-D embedding.
With the default widths a single branch is ~0.19M params, so three branches plus
a small fusion head (Day 8) come to ≈0.7M as specified.

The branch returns an embedding (no classifier head) so Day 8 can concatenate the
three modalities before the final linear layer. Pass ``n_classes`` to attach a
standalone softmax head (useful for smoke-testing or per-branch pretraining).
"""

from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """3x3 depthwise conv + 1x1 pointwise conv, each with BN + ReLU.

    ``stride`` applies to the depthwise conv, so the block both downsamples and
    changes channel count in one lightweight step.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,  # one filter per channel -> "depthwise"
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class CNNBranch(nn.Module):
    """Depthwise-separable CNN encoder for one modality.

    Parameters
    ----------
    in_channels:
        Input channels (1 for a single RP/GAF/MTF image).
    widths:
        Channel width after the stem and after each depthwise-separable block.
        ``widths[-1]`` is the embedding dimension.
    n_classes:
        If given, attach a linear classifier head and ``forward`` returns logits;
        otherwise ``forward`` returns the ``widths[-1]``-D embedding (for fusion).
    """

    def __init__(
        self,
        in_channels: int = 1,
        widths: tuple[int, ...] = (32, 64, 128, 256, 256),
        n_classes: int | None = None,
    ) -> None:
        super().__init__()
        if len(widths) < 2:
            raise ValueError("widths must have at least 2 entries")

        self.embedding_dim = widths[-1]

        # Stem: standard stride-2 conv to lift 1 -> widths[0] and halve H, W.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        # Stride-2 depthwise-separable blocks across the width schedule, plus one
        # final block at constant width for extra depth.
        blocks: list[nn.Module] = []
        for cin, cout in zip(widths[:-1], widths[1:]):
            blocks.append(DepthwiseSeparableConv(cin, cout, stride=2))
        blocks.append(DepthwiseSeparableConv(widths[-1], widths[-1], stride=2))
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B, widths[-1], 1, 1)
        self.head = nn.Linear(self.embedding_dim, n_classes) if n_classes else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)  # (B, embedding_dim)
        if self.head is not None:
            return self.head(x)
        return x


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Total number of (trainable) parameters in a module."""
    return sum(
        p.numel()
        for p in module.parameters()
        if p.requires_grad or not trainable_only
    )
