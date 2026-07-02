"""Signal-to-image transforms for EchoFuseNet.

Each transform maps a 1-D ECG beat (e.g. ``BeatSegment.signal``) to a 2-D image
that a CNN branch consumes. Day 3: Recurrence Plot (RP). Day 4: Gramian Angular
Field (GAF). Day 5: Markov Transition Field (MTF). Together these are the three
modalities fused in the model.
"""

from .gaf import gramian_angular_field
from .mtf import markov_transition_field
from .rp import recurrence_plot

__all__ = ["recurrence_plot", "gramian_angular_field", "markov_transition_field"]
