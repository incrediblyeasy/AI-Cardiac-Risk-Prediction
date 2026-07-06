"""Uncertainty estimation & calibration for EchoFuseNet (§4 enhancement).

Scoped deliberately to the two edge-friendly methods the checklist greenlit:

* **Monte Carlo Dropout** (``mc_dropout``) — predictive uncertainty from a single
  trained model by keeping dropout active at inference and averaging several
  stochastic forward passes. Costs latency (N passes) but *no* extra parameters,
  so it respects the edge-deployment premise; measure the N-pass latency against
  the <15 ms budget before shipping.
* **Temperature scaling** (``temperature``) — a one-parameter post-hoc
  calibration that rescales logits to fix over/under-confidence. Near-zero cost.

**Deep ensembles are intentionally excluded** (see the checklist §4 flag): N×
model size and N× latency conflicts with the edge premise. If explored, an
ensemble is a *research-only* calibration comparison reported in the paper, never
"the model" — it does not belong in this shipping-path module.
"""

from .mc_dropout import MCDropoutResult, enable_mc_dropout, mc_dropout_predict
from .temperature import TemperatureScaler, expected_calibration_error

__all__ = [
    "MCDropoutResult",
    "enable_mc_dropout",
    "mc_dropout_predict",
    "TemperatureScaler",
    "expected_calibration_error",
]
