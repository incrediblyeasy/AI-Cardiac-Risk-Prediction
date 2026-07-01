# AAMI Beat-Class Mapping (MIT-BIH → N/S/V/F/Q)

The ANSI/AAMI EC57 recommendation collapses the raw MIT-BIH beat annotation
symbols into five classes. EchoFuseNet uses this 5-class problem.

Source of truth in code:
[`paper1_echofusenet/data/aami.py`](../paper1_echofusenet/data/aami.py).

| AAMI class | Meaning | MIT-BIH symbols |
|---|---|---|
| **N** | Normal / bundle-branch / escape | `N` (normal), `L` (LBBB), `R` (RBBB), `e` (atrial escape), `j` (junctional escape) |
| **S** | Supraventricular ectopic (SVEB) | `A` (APB), `a` (aberrated APB), `J` (junctional premature), `S` (supraventricular premature) |
| **V** | Ventricular ectopic (VEB) | `V` (PVC), `E` (ventricular escape) |
| **F** | Fusion | `F` (fusion of ventricular + normal) |
| **Q** | Unknown / paced | `/` (paced), `f` (fusion of paced + normal), `Q` (unclassifiable) |

## Non-beat annotations

Symbols such as `+` (rhythm change), `~` (signal quality), `|`, `!`, `[`, `]`
are **not** beats and are filtered out before labeling. See `NON_BEAT_SYMBOLS`
in `aami.py`.

## Notes

- Class order `(N, S, V, F, Q)` is canonical → integer labels `0..4`.
- The paced records (102, 104, 107, 217) are excluded entirely under the
  inter-patient protocol, so class **Q** is dominated by `Q`/`f` from the
  remaining records and is naturally rare — expected under de Chazal.

## Reference

de Chazal et al., *IEEE TBME* 51(7), 2004; ANSI/AAMI EC57 standard.
