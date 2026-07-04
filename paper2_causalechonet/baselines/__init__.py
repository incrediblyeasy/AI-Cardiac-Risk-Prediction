"""Associational attribution baselines (Grad-CAM, SHAP) — comparison suite.

These exist so Paper 2's causal, intervention-based modality attribution
(``attribution``) can be compared *on the same frozen encoder* against the two
most common associational explainers. The "causal vs. associational divergence"
table (roadmap §3.5) is only apples-to-apples if all three read the identical
model.

Both are implemented and unit-tested against the frozen encoder, with no extra
dependency: ``branch_gradcam`` hooks each branch's final conv for a Grad-CAM
saliency reduced to a per-modality scalar; ``shap_modality_values`` computes
*exact* Shapley values over the 2^3 modality coalitions (only three players, so no
sampling / no ``shap`` package needed).
"""

from .gradcam import branch_gradcam
from .shap_baseline import shap_modality_values

__all__ = ["branch_gradcam", "shap_modality_values"]
