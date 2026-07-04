"""CardioCausal (Paper 3, Capstone) — structural-causal multimodal risk engine.

Paper 3 fuses the ECG representation from Papers 1-2 with tabular EHR context into
a shared latent space, places a structural causal model over it for calibrated
risk, extends Paper 2's CVAE into a **recourse** engine (causally-consistent
intervention recommendations over *modifiable* variables only), and adds a bounded
in-silico longitudinal layer — all validated with a target-trial-emulation causal
stack (IPW, E-values, negative controls, positivity/balance diagnostics).

**Full title:** *CardioCausal: A Structural-Causal Multimodal Engine for
Personalized Cardiac Risk, Counterfactual Intervention Recommendation, and
Pathway-Level Explanation*. Target journal: *Artificial Intelligence in Medicine*.

Status: **scaffold — largest remaining scope.** The self-contained, testable
pieces are implemented (fusion module, risk head, recourse *modifiability mask*,
bounded longitudinal propagation, evaluation metrics, E-values, target-trial
protocol object). Everything gated on **MIMIC-IV access** — cohort construction,
the full SCM fit, external validation on Chapman-Shaoxing / CODE-15% — is a
documented stub, because per ``PROJECT_STATUS_AND_ROADMAP.md`` §4 PhysioNet
credentialing is a lead-time bottleneck that no code can route around.
"""

__version__ = "0.0.1"
