"""CausalEchoNet (Paper 2) — modality-specific counterfactual explanations.

Paper 2 sits directly on top of Paper 1: it imports EchoFuseNet's **frozen**
encoder and never retrains it. On that fixed representation space it builds

* a feature-space Conditional VAE that answers "what minimal change flips the
  predicted class A -> B?" (`cvae`),
* modality-level causal attribution via intervention / ITE over the RP/GAF/MTF
  branches (`attribution`), and
* associational baselines (Grad-CAM, SHAP) for an apples-to-apples "causal vs.
  associational" comparison (`baselines`).

**Full title:** *CausalEchoNet: Modality-Specific Counterfactual Explanations
for Multimodal ECG Arrhythmia Classification via Conditional VAE and Causal
Attribution*. Target journal: *IEEE Journal of Biomedical and Health
Informatics*.

Status: **scaffold.** The frozen-encoder loader, CVAE architecture, attribution
mechanics, and counterfactual-quality metrics are implemented and unit-tested;
the end-to-end *training* loop and the Grad-CAM/SHAP baselines are stubs, because
per the project roadmap (`PROJECT_STATUS_AND_ROADMAP.md` §2/§3) Paper 2 must not
train against a *moving* Paper 1 target — training starts only once Paper 1's GPU
headline run is locked and its encoder checkpoint is exported and frozen.
"""

__version__ = "0.0.1"
