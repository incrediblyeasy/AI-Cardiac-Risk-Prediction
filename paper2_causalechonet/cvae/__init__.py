"""Feature-space Conditional VAE for CausalEchoNet counterfactuals.

The CVAE lives in the frozen encoder's representation space (Paper 1), *not* in
raw signal/image space — so a "counterfactual edit" is a small move in the fused
RP/GAF/MTF embedding that, when pushed through the frozen decision head, flips the
predicted AAMI class. Conditioning on the target class lets us ask the core
question: **what minimal change shifts class A -> class B?**
"""

from .model import FeatureCVAE, cvae_loss
from .metrics import proximity, sparsity, validity, counterfactual_report

__all__ = [
    "FeatureCVAE",
    "cvae_loss",
    "validity",
    "proximity",
    "sparsity",
    "counterfactual_report",
]
