"""Precompute and cache frozen-encoder representations for CVAE training.

Because the encoder is **frozen**, every beat maps to a fixed representation — so
we encode each fold exactly once and train the CVAE on the cached vectors instead
of re-running three CNN branches every epoch. This is both a large speed-up and
the thing that makes ``training.train`` a small MLP loop over ``(representation,
label)`` tensors (see ``training/train.py``).

``build_representation_dataset`` consumes any loader yielding the Day-6
``(rp, gaf, mtf, label)`` tuples (e.g. Paper 1's ``build_dataloaders``) and returns
an in-memory ``TensorDataset``. Pass ``cache_path`` to persist it; a second call
with the same path loads instantly (the cache records the encoder's representation
dim so a stale cache is caught).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .encoder import FrozenEncoder


@torch.no_grad()
def encode_loader(encoder: FrozenEncoder, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every ``(rp, gaf, mtf, label)`` batch into ``(representations, labels)``.

    Returns two stacked tensors: representations ``(N, D)`` and labels ``(N,)``.
    """
    reps: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for rp, gaf, mtf, label in loader:
        reps.append(encoder.encode(rp, gaf, mtf))
        labels.append(label)
    if not reps:
        return (
            torch.empty(0, encoder.representation_dim),
            torch.empty(0, dtype=torch.long),
        )
    return torch.cat(reps, dim=0), torch.cat(labels, dim=0)


def build_representation_dataset(
    encoder: FrozenEncoder,
    loader: DataLoader,
    cache_path: str | Path | None = None,
) -> TensorDataset:
    """Build (or load) a cached ``TensorDataset`` of ``(representation, label)``.

    If ``cache_path`` exists and matches the encoder's representation dim, it is
    loaded. Otherwise the loader is encoded once and (if a path was given) saved.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            if blob.get("representation_dim") != encoder.representation_dim:
                raise ValueError(
                    f"cache {cache_path} has representation_dim "
                    f"{blob.get('representation_dim')} but encoder produces "
                    f"{encoder.representation_dim}; delete the stale cache"
                )
            return TensorDataset(blob["representations"], blob["labels"])

    reps, labels = encode_loader(encoder, loader)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "representations": reps,
                "labels": labels,
                "representation_dim": encoder.representation_dim,
            },
            cache_path,
        )
    return TensorDataset(reps, labels)
