"""Forward-pass smoke test + parameter report for one CNN branch (Day 7).

Instantiates a `CNNBranch`, runs a dummy forward pass, and logs the parameter
count — per stage and total — against the ~0.7M budget for the full three-branch
EchoFuseNet model (Day 8).

Usage:
    python -m scripts.branch_summary
"""

from __future__ import annotations

import torch

from paper1_echofusenet.models import CNNBranch, count_parameters

BUDGET = 0.7e6  # total-model parameter budget (all three branches + fusion)


def main() -> None:
    branch = CNNBranch()
    branch.eval()

    # --- forward-pass smoke test with dummy input ---
    x = torch.zeros(2, 1, 256, 256)  # (B, 1, L, L) as produced by the DataLoader
    with torch.no_grad():
        emb = branch(x)
    print("Forward-pass smoke test:")
    print(f"  input   {tuple(x.shape)}")
    print(f"  output  {tuple(emb.shape)}   (embedding_dim={branch.embedding_dim})")
    assert emb.shape == (2, branch.embedding_dim)
    print("  OK\n")

    # --- parameter breakdown ---
    print("Parameter count by stage:")
    total = 0
    for name, module in [
        ("stem", branch.stem),
        ("blocks", branch.blocks),
    ]:
        n = count_parameters(module)
        total += n
        print(f"  {name:8s} {n:>9,}")
    branch_total = count_parameters(branch)
    print(f"  {'TOTAL':8s} {branch_total:>9,}  ({branch_total / 1e6:.3f}M)\n")

    # --- budget projection for the full model (Day 8) ---
    three = 3 * branch_total
    emb = branch.embedding_dim
    fusion_est = 3 * emb * 128 + 128 + 128 * 5 + 5  # concat(3*emb)->128->5
    projected = three + fusion_est
    print("Full-model budget projection (~0.7M target):")
    print(f"  3 x branch          {three:>9,}")
    print(f"  fusion head (est.)  {fusion_est:>9,}   (3*{emb} -> 128 -> 5)")
    print(f"  projected total     {projected:>9,}  ({projected / 1e6:.3f}M)")
    verdict = "WITHIN BUDGET" if projected <= BUDGET else "OVER BUDGET"
    print(f"  budget {BUDGET/1e6:.1f}M  ->  {verdict}")


if __name__ == "__main__":
    main()
