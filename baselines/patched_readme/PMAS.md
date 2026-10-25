# patched/PMAS

Minimal repairs to `Robust_Steganography/PMAS` so it can execute at all. These
files shadow the clone via `addpath(..., '-begin')` in
`experiments/pmas_embed_shard.m`. **The pristine clone is never modified.**

Any PMAS number in the paper must be labelled as coming from this patched
version, with the defects below stated.

## Files here

| File | Origin | Change |
|---|---|---|
| `preprocessUNICORE.m` | copied from `PMAS/` | line 27: `JPEGrecompress(...,attack_QF)` → `JPEGrecompress(...,attack_QF,1,0)` |
| `postprocessUNICORE.m` | copied from `PMAS/` | line 9: same fix |
| `J_UNIWARDcost.m` | copied **unmodified** from `MINICER/` | supplies a dependency missing from `PMAS/` |
| `UERDcost.m` | copied **unmodified** from `MINICER/` | as above, for the `distortion=0` path |

## Defect 1 — `JPEGrecompress` called with 3 of 5 arguments

`JPEGrecompress(input_path, output_path, QF, save, color)` takes five arguments.
**Every** call site in `PMAS/` passes three:

- `preprocessUNICORE.m:27`
- `postprocessUNICORE.m:9`
- `generateStegoUNICORE.m:32` (commented out)

`color` is therefore undefined and line 4's `if color` throws
*"Not enough input arguments"* on the first image and every one after it.

The values are unambiguous from context, so this is a safe patch:
- `color = 0` — BOSSbase is grayscale, and the `else` branch at line 13 is the
  grayscale path.
- `save = 1` — callers read the written file back (`preprocessUNICORE` line 28
  does `jpeg_read(afterchannel_cover_path)` when the file already exists), so it
  must be saved.

## Defect 2 — `J_UNIWARDcost` missing from the folder

`preprocessUNICORE` calls `J_UNIWARDcost`, which does not exist anywhere in
`PMAS/`. It ships in `MINICER/` and `ROAST/`, whose copies are byte-identical to
each other. Copied here unmodified. `UERDcost.m` (the `distortion=0` branch) is
likewise absent from `PMAS/` and taken from `MINICER/`.

## Defect 3 — undefined variable in the driver (not patched; not needed)

`testRobustUNICORE.m:52` reads

```matlab
imwrite(imread(stego_Path),afterchannel_stego_Path,'quality',attack_QF);
```

while the loop defines `stego_path` / `afterchannel_stego_path` (lines 28–29,
lowercase). MATLAB is case-sensitive, so this throws on every iteration; the
catch at line 60 then sets `bit_error_rate(i)=0` and those zeros are averaged
into `ave_error_rate: 0`.

**Combined with defect 1, the shipped driver cannot successfully process a
single image, yet reports flawless robustness.** Not patched here because
`experiments/pmas_embed_shard.m` replaces the driver entirely; the line is also
redundant, since `generateStegoUNICORE` already writes the after-channel file
via `postprocessUNICORE` (`generateStegoUNICORE.m:22`).

## Parameter search so far (2026-08-07)

The mechanism demonstrably engages — BER responds monotonically to the knobs in
the direction the README predicts — but extraction is not yet correct.

Payload 0.1, 6 BOSSbase covers, `distortion=1`:

**Threat model is not the issue.** Unlike SCRIPT, high quality factors do not
fix it:

| Cover QF → attack QF | Mean BER |
|---|---|
| 95 → 95 | 0.4449 |
| 95 → 90 | 0.5005 |
| 90 → 95 | 0.4963 |
| 85 → 85 | 0.4665 |

**Post-processing rounds `mode(3)` and cover mode `mode(1)` both matter**, at
QF 95 → 95, T=8:

| mode | Mean BER |
|---|---|
| [1 1 1] | 0.4905 |
| [2 1 1] | 0.4449 |
| [1 1 3] | 0.4526 |
| [2 1 3] | 0.3124 |
| [1 1 5] | 0.4276 |
| **[2 1 5]** | **0.2958** |

`mode(1)=2` (post-compressed cover) beats 1 throughout, matching the README's
"Postprocessing Mitigation", and more rounds monotonically reduce BER — the
post-processing loop is doing what it is supposed to do.

**`T` has an interior optimum near 8**, at `mode=[2 1 5]`:

| T | 2 | 8 | 16 | 32 |
|---|---|---|---|---|
| Mean BER | 0.4564 | **0.2958** | 0.3515 | 0.4702 |

Best configuration found: `T=8, mode=[2 1 5], distortion=1, QF 95→95` →
**BER 0.296, 0/6 errorless.**

### Diagnosis: post-processing does not converge

The decisive diagnostic (QF 95→95, T=8, mode=[2 1 5]), tracing one image through
every stage:

| Extraction source | BER | Coefficients ≠ STC target |
|---|---|---|
| in-memory `stego` (STC output) | **0** | — (by definition) |
| written stego file, pre-channel | 0.4696 | 2.378% |
| after-channel file | 0.4280 | **1.767%** |

**Our harness is not the problem.** Extracting from the in-memory STC output
gives BER 0, so `stc_pm1_pls_embed` is being called correctly, and
`generateStegoUNICORE` is invoked exactly as the published driver invokes it.

**The failure is localised precisely.** PMAS's design requires the after-channel
coefficients to equal the STC target *exactly*; extraction reads them straight
with `stc_ml_extract`. Post-processing is working in the right direction — it
narrows the gap from 2.378% to 1.767% — but 1.767% of 262,144 is **4,632 wrong
coefficients**, and syndrome-trellis decoding at H=10 cannot survive that. Hence
BER ≈ 0.43 regardless of how the remaining knobs are set.

**PMAS is a same-QF method.** Upward recompression is strictly worse, at full
post-processing (mode=[2 1 5], T=8):

| Cover QF → attack QF | Mean BER |
|---|---|
| **95 → 95** | **0.2958** |
| 85 → 90 | 0.4981 |
| 90 → 95 | 0.4984 |
| 85 → 95 | 0.4990 |

Note also that `mode(2)` selects among five distortion **scaling functions**
(`preprocessUNICORE.m:88-104` — linear, exponential, log, log10, squared). That
is the README's "Exploring Distortion Assignment"; it shapes *security*, not
robustness, so it is not a candidate for the convergence gap.

### Recommendation: ask the authors

Further blind sweeping is poor value. The question is now specific enough to be
worth an email, which is far more likely to get a useful answer than a vague
request would have been:

> With cover QF 95 and channel QF 95 at 0.1 bpnzac, our post-processing loop
> converges to ~1.8% of DCT coefficients differing from the STC target rather
> than to exact equality, so `stc_ml_extract` fails. Which `T` and `mode` values
> reproduce the paper's results, and is post-processing expected to reach exact
> coefficient equality? We also had to supply `J_UNIWARDcost`/`UERDcost` from
> the MINICER folder and add the missing `save`/`color` arguments to the
> `JPEGrecompress` calls in `preprocessUNICORE`/`postprocessUNICORE`.

Until then, **PMAS is excluded from the baseline set** and the exclusion is
documented here. That is a defensible position given the evidence: three fatal
defects in the shipped code, undocumented parameters, and a precisely
characterised convergence failure.

## Status: runs, but not yet correct

With the patches, the pipeline executes cleanly — no exceptions, ~2.9 s/image.
But at `T=8, mode=[2 1 1], distortion=1` the extraction returns **BER ≈ 0.50**,
i.e. pure chance: the decoder recovers nothing.

That is almost certainly wrong parameters rather than a broken method. The repo
README documents what `T` and `mode` *mean* but not the values used for the
published results:

- `T` — threshold controlling the robust region
- `mode(1)` — 1 = uncompressed cover, 2 = post-compressed
- `mode(2)` — distortion calculation method
- `mode(3)` — number of post-processing rounds
- `distortion` — 1 = J-UNIWARD, 0 = UERD

The README also notes that `preprocessUNICORE` can use GDM or PDM, with the GDM
code commented out — another unrecorded configuration choice.

**Next step:** read the TCSVT paper ("Robust Steganography for High Quality
Images") for the parameter values, or sweep `T` and `mode`. Do not quote a PMAS
number until the extraction works.
