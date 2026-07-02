"""EchoFuseNet — the assembled three-branch late-fusion classifier.

Each heartbeat is encoded as three signal-to-image modalities (Day 3-5):
Recurrence Plot, Gramian Angular Field, Markov Transition Field. Each modality
gets its **own** lightweight depthwise-separable CNN branch (Day 7) — weights are
*not* shared, because the three images have very different texture statistics.

Late fusion
-----------
The three branch embeddings are concatenated and passed through a small MLP head
that produces the 5-class (AAMI N/S/V/F/Q) logits:

    rp  -> branch_rp  --\
    gaf -> branch_gaf ---> concat(3 x emb) -> Linear -> BN -> ReLU -> Dropout
    mtf -> branch_mtf --/                                        -> Linear -> logits

``forward`` takes the three images in the fixed ``(rp, gaf, mtf)`` order produced
by the Day-6 DataLoader (whose batches are ``(rp, gaf, mtf, label)``).

With the default branch widths this assembles to ≈0.65M parameters — within the
~0.7M budget; see `scripts/model_summary.py` / `tests/test_echofusenet.py`.
"""

from __future__ import annotations

import torch
from torch import nn

from .branch import CNNBranch


class EchoFuseNet(nn.Module):
    """Three-branch late-fusion ECG-beat classifier.

    Parameters
    ----------
    n_classes:
        Number of output classes (5 for AAMI N/S/V/F/Q).
    widths:
        Channel schedule for each branch (see ``CNNBranch``).
    fusion_hidden:
        Hidden width of the fusion MLP.
    dropout:
        Dropout probability in the fusion head.
    """

    def __init__(
        self,
        n_classes: int = 5,
        widths: tuple[int, ...] = (32, 64, 128, 256, 256),
        fusion_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # One independent branch per modality (no weight sharing).
        self.branch_rp = CNNBranch(in_channels=1, widths=widths)
        self.branch_gaf = CNNBranch(in_channels=1, widths=widths)
        self.branch_mtf = CNNBranch(in_channels=1, widths=widths)

        emb = self.branch_rp.embedding_dim
        fused_dim = 3 * emb

        # Late-fusion classifier head over the concatenated embeddings.
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, n_classes),
        )

    def forward(
        self, rp: torch.Tensor, gaf: torch.Tensor, mtf: torch.Tensor
    ) -> torch.Tensor:
        """Return class logits ``(B, n_classes)`` for a batch of the three images.

        Each input is ``(B, 1, L, L)`` (as yielded by the Day-6 DataLoader).
        """
        e_rp = self.branch_rp(rp)
        e_gaf = self.branch_gaf(gaf)
        e_mtf = self.branch_mtf(mtf)
        fused = torch.cat([e_rp, e_gaf, e_mtf], dim=1)
        return self.fusion(fused)
