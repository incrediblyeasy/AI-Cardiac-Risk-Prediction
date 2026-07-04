"""CVAE training config + loop for CausalEchoNet.

Config-driven and reproducible, mirroring Paper 1's `training` package: a run is
fully described by a JSON config + seed. The loop itself is a scaffold — see
`train.py` — because it must not start until Paper 1's frozen encoder checkpoint
is exported (roadmap §2/§3).
"""

from .config import CVAETrainConfig

__all__ = ["CVAETrainConfig"]
