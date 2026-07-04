"""Frozen Paper-1 encoder access for CausalEchoNet.

The single hard dependency gate between Paper 1 and Paper 2: Paper 2 loads Paper
1's trained EchoFuseNet checkpoint, strips the classifier head, freezes every
weight, and exposes a clean ``encode(rp, gaf, mtf) -> representation`` interface.
Nothing in Paper 2 ever updates these weights (a unit test enforces it).
"""

from .frozen import FrozenEncoder, load_frozen_encoder

__all__ = ["FrozenEncoder", "load_frozen_encoder"]
