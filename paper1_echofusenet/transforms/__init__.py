"""Signal-to-image transforms for EchoFuseNet.

Each transform maps a 1-D ECG beat (e.g. ``BeatSegment.signal``) to a 2-D image
that a CNN branch consumes. Day 3: Recurrence Plot (RP). Days 4-5 add GAF and
MTF, giving the three modalities fused in the model.
"""

from .rp import recurrence_plot

__all__ = ["recurrence_plot"]
