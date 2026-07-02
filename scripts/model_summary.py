"""Assemble EchoFuseNet, forward-pass smoke test, and parameter-budget report.

Instantiates the full three-branch late-fusion model, runs a dummy forward pass,
and logs the parameter count per component against the ~0.7M budget.

Usage:
    python -m scripts.model_summary
"""

from __future__ import annotations

import torch

from paper1_echofusenet.models import EchoFuseNet, count_parameters

BUDGET = 0.7e6


def main() -> None:
    model = EchoFuseNet(n_classes=5).eval()

    # --- forward-pass smoke test with the (rp, gaf, mtf) input triple ---
    rp = torch.zeros(2, 1, 256, 256)
    gaf = torch.zeros(2, 1, 256, 256)
    mtf = torch.zeros(2, 1, 256, 256)
    with torch.no_grad():
        logits = model(rp, gaf, mtf)
    print("Forward-pass smoke test:")
    print(f"  inputs  3 x {tuple(rp.shape)}")
    print(f"  logits  {tuple(logits.shape)}")
    assert logits.shape == (2, 5)
    print("  OK\n")

    # --- parameter breakdown ---
    print("Parameter count by component:")
    total = count_parameters(model)
    for name, module in [
        ("branch_rp", model.branch_rp),
        ("branch_gaf", model.branch_gaf),
        ("branch_mtf", model.branch_mtf),
        ("fusion", model.fusion),
    ]:
        n = count_parameters(module)
        print(f"  {name:11s} {n:>9,}  ({100 * n / total:4.1f}%)")
    print(f"  {'TOTAL':11s} {total:>9,}  ({total / 1e6:.3f}M)\n")

    verdict = "WITHIN BUDGET" if total <= BUDGET else "OVER BUDGET"
    print(f"Budget verification: {total:,} <= {int(BUDGET):,} ?  ->  {verdict}")


if __name__ == "__main__":
    main()
