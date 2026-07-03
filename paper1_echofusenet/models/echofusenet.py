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

# Fixed modality order — matches the (rp, gaf, mtf) tuple the Day-6 DataLoader
# yields and the ``forward`` argument order.
CANONICAL_MODALITIES: tuple[str, ...] = ("rp", "gaf", "mtf")


class EchoFuseNet(nn.Module):
    """Late-fusion ECG-beat classifier over one or more signal-to-image modalities.

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
    modalities:
        Which modalities to fuse — any non-empty subset of ``("rp", "gaf",
        "mtf")``. Defaults to all three (the full EchoFuseNet). Single- and
        two-modality subsets are what the Day-12 ablation study sweeps; only the
        selected branches are built, and only their embeddings are concatenated
        before the fusion head. ``forward`` still takes the full ``(rp, gaf,
        mtf)`` triple (the DataLoader always yields all three) and simply ignores
        the images whose branch is absent.
    """

    def __init__(
        self,
        n_classes: int = 5,
        widths: tuple[int, ...] = (32, 64, 128, 256, 256),
        fusion_hidden: int = 128,
        dropout: float = 0.3,
        modalities: tuple[str, ...] = CANONICAL_MODALITIES,
    ) -> None:
        super().__init__()

        requested = {m.lower() for m in modalities}
        unknown = requested - set(CANONICAL_MODALITIES)
        if unknown:
            raise ValueError(
                f"unknown modalities {sorted(unknown)}; "
                f"choose from {list(CANONICAL_MODALITIES)}"
            )
        if not requested:
            raise ValueError("at least one modality is required")
        # Store in canonical order regardless of input order.
        self.modalities = tuple(m for m in CANONICAL_MODALITIES if m in requested)

        # One independent branch per *selected* modality (no weight sharing);
        # inactive branches are left as None so their input is skipped.
        self.branch_rp = CNNBranch(in_channels=1, widths=widths) if "rp" in requested else None
        self.branch_gaf = CNNBranch(in_channels=1, widths=widths) if "gaf" in requested else None
        self.branch_mtf = CNNBranch(in_channels=1, widths=widths) if "mtf" in requested else None

        emb = widths[-1]
        fused_dim = len(self.modalities) * emb

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
        Images whose branch was not built (ablation subsets) are ignored.
        """
        embeddings: list[torch.Tensor] = []
        if self.branch_rp is not None:
            embeddings.append(self.branch_rp(rp))
        if self.branch_gaf is not None:
            embeddings.append(self.branch_gaf(gaf))
        if self.branch_mtf is not None:
            embeddings.append(self.branch_mtf(mtf))
        fused = torch.cat(embeddings, dim=1)
        return self.fusion(fused)
