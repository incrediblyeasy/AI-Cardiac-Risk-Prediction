"""Temperature scaling for EchoFuseNet calibration (§4 enhancement).

Modern CNNs are typically *over-confident*: the softmax probability of the
predicted class overstates its true accuracy. Temperature scaling (Guo et al.,
2017) is the standard fix — divide the logits by a single learned scalar ``T``
before softmax:

    p = softmax(logits / T)

``T`` is fit **post-hoc** on a held-out validation set by minimising negative
log-likelihood; it does not change which class is predicted (argmax is invariant
to positive scaling), only the confidence, so accuracy is untouched while
calibration improves. One parameter, one extra division at inference — the
cheapest calibration there is, fully compatible with the edge budget.

Fitting operates on cached ``(logits, labels)`` (run the val fold once, keep the
logits) rather than re-running the model, so it is fast and pure. Calibration
quality is measured with **Expected Calibration Error** (ECE) before/after.

Validation-fold discipline: fit ``T`` on a DS1-internal validation split, never
on DS2 — calibrating on the test fold would leak it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error (ECE) with equal-width confidence bins.

    Bins beats by their predicted-class confidence, then sums the gap between
    average confidence and empirical accuracy in each bin, weighted by bin size.
    0 = perfectly calibrated. ``probs`` is ``(N, C)`` softmax; ``labels`` ``(N,)``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    if probs.size == 0:
        return 0.0
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        # Half-open bins (lo, hi]; the top bin includes confidence == 1.0.
        in_bin = (confidences > lo) & (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        avg_conf = float(confidences[in_bin].mean())
        avg_acc = float(accuracies[in_bin].mean())
        ece += (count / n) * abs(avg_conf - avg_acc)
    return ece


class TemperatureScaler(nn.Module):
    """A single learned temperature that recalibrates logits post-hoc.

    Usage::

        scaler = TemperatureScaler().fit(val_logits, val_labels)
        calibrated_probs = scaler.predict_proba(test_logits)

    ``fit`` optimises ``T`` by L-BFGS on validation NLL; ``temperature`` exposes
    the learned value (``> 1`` means the model was over-confident).
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        # Parameterised in raw form and softplus-mapped to keep T strictly > 0.
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(init_temperature))))

    @property
    def temperature(self) -> float:
        return float(self.log_temperature.exp().item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits (``logits / T``)."""
        return logits / self.log_temperature.exp()

    def fit(
        self,
        logits: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> "TemperatureScaler":
        """Fit ``T`` to minimise NLL on validation ``(logits, labels)``."""
        logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        labels = torch.as_tensor(np.asarray(labels).ravel(), dtype=torch.long)
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nll(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_closure)
        return self

    @torch.no_grad()
    def predict_proba(
        self, logits: torch.Tensor | np.ndarray
    ) -> np.ndarray:
        """Calibrated softmax probabilities ``(N, C)`` for the given logits."""
        logits = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
        return torch.softmax(self.forward(logits), dim=1).cpu().numpy()
