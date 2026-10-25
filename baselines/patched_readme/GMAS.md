# patched/GMAS

One repair to `Robust_Steganography/GMAS` so that GMAS, and the reconstructed
ROAST that reuses its embedder, run on images that are not 512x512. The file
here shadows the clone via `addpath(..., '-begin')` in
`experiments/gmas_embed_shard.m` and `experiments/roast_recon_shard.m`.
**The pristine clone is never modified.**

## Files here

| File | Origin | Change |
|---|---|---|
| `stc3_embed.m` | copied from `GMAS/` | line 4: `zeros(3, 86016, 'single')` -> `zeros(3, numel(rho_m), 'single')` |

## The defect

`stc3_embed.m:4` allocates the ternary cost array with a literal length:

```matlab
costs = zeros(3, 86016, 'single');
costs(1,:) = rho_m;
```

86016 = 21 mid-frequency coefficients x 4096 blocks, i.e. a 512x512 image. On a
256x256 image `rho_m` has 21504 entries and the assignment on the next line
throws *"Unable to perform assignment because the size of the left side is
1-by-86016 and the size of the right side is 1-by-21504"*. Every image fails.

Found 2026-10-05 by the 8-image pilot of `exp24_second_source.py`: GMAS and
ROAST embedded 0 of 8 on both BOSSbase-256 and BOWS2-256.

## Why the patch is safe

For a 512x512 image `numel(rho_m)` is 86016, so the patched file allocates the
same array as the original and nothing downstream changes. Checked by
re-embedding 512x512 images with the patched copy and comparing the stego files
byte for byte against `results/exp13_fullscale` (see FINDINGS).

Any GMAS or ROAST number at a size other than 512x512 comes from this patched
version and must say so.
