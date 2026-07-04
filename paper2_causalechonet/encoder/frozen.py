"""Load and freeze Paper 1's EchoFuseNet as a fixed feature extractor.

Why a dedicated wrapper
-----------------------
Paper 2 operates in the **representation space** produced by Paper 1's encoder,
not in raw signal or image space. Two things must be guaranteed for the causal
claims downstream to be well-posed:

1. **The encoder never trains.** Every parameter is frozen (``requires_grad =
   False``) and the module is kept in ``eval`` mode so BatchNorm/Dropout are
   deterministic. If any gradient ever reached these weights, "the CVAE edits the
   representation" and "the representation itself moved" would be confounded.
   ``tests/test_paper2_encoder.py`` asserts the freeze holds even after a
   backward pass through a downstream module.

2. **The representation is well-defined and decomposable by modality.** We expose
   the *pre-fusion* embedding — the concatenation of the per-branch (RP/GAF/MTF)
   embeddings, in EchoFuseNet's canonical order — because the whole point of
   Paper 2 is per-modality attribution, which needs to address each modality's
   sub-vector independently (see ``modality_slices``). The frozen fusion head is
   also exposed (``decision``) so Paper 2 can ask "does this edited representation
   flip the predicted class?" without ever touching the raw images again.

Checkpoint format
-----------------
Reads the dict written by ``paper1_echofusenet.training.train.save_checkpoint``:
``{"model_state", "optimizer_state", "epoch", "metric", "config"}``. The encoder
architecture is rebuilt from the embedded resolved ``config`` (so the wrapper
never has to guess widths/modalities), then ``model_state`` is loaded strictly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from paper1_echofusenet.models import EchoFuseNet
from paper1_echofusenet.models.echofusenet import CANONICAL_MODALITIES
from paper1_echofusenet.training.config import TrainConfig


class FrozenEncoder(nn.Module):
    """A frozen EchoFuseNet exposed as ``encode`` + ``decision``.

    Parameters
    ----------
    model:
        A constructed :class:`EchoFuseNet`. Its weights are frozen in place.

    Notes
    -----
    Construct from a Paper-1 checkpoint with :meth:`from_checkpoint` /
    :func:`load_frozen_encoder`; the bare constructor is handy for tests that
    build a fresh (untrained) EchoFuseNet to exercise the plumbing.
    """

    def __init__(self, model: EchoFuseNet) -> None:
        super().__init__()
        self.model = model
        self.modalities = model.modalities
        self.embedding_dim = model.fusion[0].in_features // len(self.modalities)
        self.freeze()

    # -- freezing ----------------------------------------------------------
    def freeze(self) -> None:
        """Disable grads on every parameter and pin eval mode."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "FrozenEncoder":
        """Override: the wrapped encoder must never leave eval mode.

        ``nn.Module.train()`` is called implicitly by many training loops (e.g.
        when a downstream CVAE calls ``self.train()``). We keep the frozen
        encoder in eval regardless so its BN statistics never update.
        """
        super().train(mode)
        self.model.eval()
        return self

    # -- representation geometry ------------------------------------------
    @property
    def representation_dim(self) -> int:
        """Width of the fused pre-classifier representation."""
        return len(self.modalities) * self.embedding_dim

    def modality_slices(self) -> dict[str, slice]:
        """Map each active modality to its contiguous block in the representation.

        The representation is ``concat([emb_rp, emb_gaf, emb_mtf])`` restricted to
        the *active* modalities, in canonical order. Attribution intervenes on one
        modality by addressing its slice.
        """
        slices: dict[str, slice] = {}
        for i, m in enumerate(self.modalities):
            slices[m] = slice(i * self.embedding_dim, (i + 1) * self.embedding_dim)
        return slices

    # -- forward paths -----------------------------------------------------
    @torch.no_grad()
    def encode(self, rp: torch.Tensor, gaf: torch.Tensor, mtf: torch.Tensor) -> torch.Tensor:
        """Return the fused pre-classifier representation ``(B, representation_dim)``.

        Mirrors ``EchoFuseNet.forward`` up to (but not through) the fusion head:
        each active branch encodes its image, and the embeddings are concatenated
        in canonical order. Runs under ``no_grad`` in eval mode.
        """
        m = self.model
        embeddings: list[torch.Tensor] = []
        if m.branch_rp is not None:
            embeddings.append(m.branch_rp(rp))
        if m.branch_gaf is not None:
            embeddings.append(m.branch_gaf(gaf))
        if m.branch_mtf is not None:
            embeddings.append(m.branch_mtf(mtf))
        return torch.cat(embeddings, dim=1)

    def decision(self, representation: torch.Tensor) -> torch.Tensor:
        """Class logits ``(B, n_classes)`` from a (possibly edited) representation.

        Applies the frozen fusion head. This is the fixed decision function the
        CVAE's counterfactual *validity* metric and the attribution ITE both query
        — kept differentiable (no ``no_grad``) so callers may backprop *into a
        representation* (never into the encoder weights, which are frozen).
        """
        return self.model.fusion(representation)

    @torch.no_grad()
    def classify(self, rp: torch.Tensor, gaf: torch.Tensor, mtf: torch.Tensor) -> torch.Tensor:
        """Convenience: ``decision(encode(...))`` — full frozen forward to logits."""
        return self.decision(self.encode(rp, gaf, mtf))

    # -- construction ------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> "FrozenEncoder":
        """Build a frozen encoder from a Paper-1 training checkpoint."""
        ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
        if "model_state" not in ckpt or "config" not in ckpt:
            raise ValueError(
                "checkpoint is missing 'model_state'/'config'; expected the dict "
                "written by paper1_echofusenet.training.train.save_checkpoint"
            )
        cfg = TrainConfig.from_dict(ckpt["config"])
        model = EchoFuseNet(
            n_classes=cfg.model.n_classes,
            widths=tuple(cfg.model.widths),
            fusion_hidden=cfg.model.fusion_hidden,
            dropout=cfg.model.dropout,
            modalities=tuple(cfg.model.modalities),
        )
        model.load_state_dict(ckpt["model_state"])
        return cls(model)


def load_frozen_encoder(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> FrozenEncoder:
    """Functional alias for :meth:`FrozenEncoder.from_checkpoint`."""
    return FrozenEncoder.from_checkpoint(path, map_location=map_location)


__all__ = ["FrozenEncoder", "load_frozen_encoder", "CANONICAL_MODALITIES"]
