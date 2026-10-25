# patched/SCRIPT

One optional change to `Robust_Steganography/SCRIPT/costGISS.m`, used only by
`experiments/exp40_script_fix.py` to test a repair. The file here shadows the
clone via `addpath(..., '-begin')` in `experiments/script_embed_shard.m`, and
only when that shard is called with its optional last argument `si_fix` set to
1, 2 or 3. Every SCRIPT result in the paper's main tables was produced with the
clone's own file. **The pristine clone is never modified.**

## Files here

| File | Origin | Change |
|---|---|---|
| `costGISS.m` | copied from `SCRIPT/` | new optional sixth argument `si_fix`; three marked lines act on it; with `si_fix = 0` the function is the original |

## What it addresses

`costGISS.m` discounts the cost of a change towards the precover's value by
`1 - 2|e|`, where `e` is the exact DCT of the precover minus the cover JPEG's
coefficient, and clips `|e|` at 0.5 with the comment *"image compressed with
imwrite(imread) is different form that with idct(dct)"*. Where the JPEG
compressor and the exact DCT round a coefficient differently, `|e| > 0.5`, the
clipped discount is exactly zero and the change is free whatever the message.

On BOSSbase at QF 65 that is about 320 coefficients per image; SCRIPT changes
about 147 of them at 0.05 bpnzac and the same number at 0.2, and those changes
alone are detected at P_E 0.43 (FINDINGS F28).

## The three variants

| `si_fix` | Where `|e| > 0.5` | Elsewhere |
|---|---|---|
| 1 | no discount (cost as without side information) | unchanged |
| 2 | no change allowed (wet cost) | unchanged |
| 3 | discount factor floored at 0.1 | discount factor floored at 0.1 |

Which of them, if any, lowers detectability without hurting delivery is what
`exp40_script_fix.py` measures.

2026-10-11: variant `si_fix = 4` added (no change wherever the released discount is
zero: |e| > 0.5 and |e| = 0.5). Used by `experiments/exp51_exact_dct_control.py`.
Variants 0-3 are unchanged.
