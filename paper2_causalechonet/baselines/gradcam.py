"""Grad-CAM over each EchoFuseNet branch — associational saliency baseline.

For each active modality we hook the **last** ``DepthwiseSeparableConv`` of its
branch, capture that layer's activations ``A`` and the gradient of the target
logit w.r.t. ``A``, and form the standard Grad-CAM map::

    α_k = GAP(∂y_c/∂A_k)              (channel importance weights)
    CAM = ReLU(Σ_k α_k A_k)           (spatial saliency, per branch)

Each branch's map is reduced to a scalar modality importance (mean over the map)
so it is directly comparable to the per-modality interventional ITE in
``attribution`` — the associational-vs-causal comparison (roadmap §3.5).

No-leakage note
---------------
The encoder stays frozen: gradients are taken w.r.t. the *input images* /
*activations*, never the weights. We enable grad locally (the caller may be under
``torch.no_grad``) and set ``requires_grad`` on cloned inputs so the graph builds
even though every parameter has ``requires_grad = False``. Parameter ``.grad`` is
never populated (a test asserts this).
"""

from __future__ import annotations

import torch


def _last_conv(branch):
    """The branch's final DepthwiseSeparableConv (its deepest feature map)."""
    return branch.blocks[-1]


def branch_gradcam(
    encoder,
    rp: torch.Tensor,
    gaf: torch.Tensor,
    mtf: torch.Tensor,
    target: torch.Tensor,
    return_maps: bool = False,
) -> dict[str, torch.Tensor]:
    """Per-modality Grad-CAM importance for a target class.

    Parameters
    ----------
    encoder:
        A :class:`FrozenEncoder` (its ``.model`` is the frozen EchoFuseNet).
    rp, gaf, mtf:
        Input image batches ``(B, 1, L, L)``.
    target:
        Target class indices ``(B,)`` (or one-hot, argmax-ed).
    return_maps:
        If True, also return the full ``(B, h, w)`` CAM per modality under the
        key ``f"{modality}_map"``.

    Returns ``{modality: importance (B,)}`` (+ maps if requested), for the active
    modalities only.
    """
    if target.dim() > 1:
        target = target.argmax(dim=1)

    model = encoder.model
    was_training = model.training
    model.eval()

    branches = {"rp": model.branch_rp, "gaf": model.branch_gaf, "mtf": model.branch_mtf}
    inputs = {"rp": rp, "gaf": gaf, "mtf": mtf}
    active = {m: b for m, b in branches.items() if b is not None}

    activations: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_module, _inp, out):
            out.retain_grad()          # keep grad on this non-leaf activation
            activations[name] = out
        return hook

    for name, branch in active.items():
        handles.append(_last_conv(branch).register_forward_hook(make_hook(name)))

    try:
        with torch.enable_grad():
            # Clone so we never flip requires_grad on the caller's tensors.
            x = {m: inputs[m].clone().requires_grad_(True) for m in inputs}
            logits = model(x["rp"], x["gaf"], x["mtf"])
            model.zero_grad(set_to_none=True)
            selected = logits.gather(1, target.view(-1, 1)).sum()
            selected.backward()

            out: dict[str, torch.Tensor] = {}
            for name in active:
                A = activations[name]                 # (B, C, h, w)
                grad = A.grad                          # (B, C, h, w)
                alpha = grad.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
                cam = torch.relu((alpha * A).sum(dim=1))     # (B, h, w)
                out[name] = cam.mean(dim=(1, 2)).detach()    # (B,) scalar importance
                if return_maps:
                    out[f"{name}_map"] = cam.detach()
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
        model.eval()  # frozen encoder: never leave eval

    return out
