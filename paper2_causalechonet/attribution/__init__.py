"""Modality-level causal attribution for CausalEchoNet.

The paper's core novelty (roadmap §3.4): instead of a saliency heat-map, quantify
each modality's *causal* effect on the decision by **intervening** on its block of
the frozen representation (RP / GAF / MTF) and measuring how the predicted class
probability moves. This is an interventional (do-operator) quantity over the three
image encodings, not an associational saliency score — which is exactly what makes
the Grad-CAM/SHAP comparison in ``baselines`` a "causal vs. associational" study.
"""

from .ite import intervene, modality_ite, attribution_table

__all__ = ["intervene", "modality_ite", "attribution_table"]
