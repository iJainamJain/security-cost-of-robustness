# Baseline status — measured 2026-08-07

Verified against `github.com/coriverchen/Robust_Steganography` at clone time,
MATLAB R2024b Update 5, PCWIN64.

## M1 gate: PASSED

`experiments/exp02_m1_gate.py` → `experiments/m1_verify_baselines.m`:

```
jpeg_read OK: 512x512 coefficients, 16628 non-zero AC
payload: 0.1  image_number: 1..10  error_rate: 0   (all ten)
payload: 0.1  ave_error_rate: 0
completed in 34.6 s
```

GMAS at 0.1 bpnzAC, cover QF 65 → "Facebook" QF 71, J-UNIWARD asymmetric cost +
generalized dither modulation + ternary STC3 + RS(31,15): **zero bit errors on
all ten bundled covers.** This is what the literature predicts for the upward-
recompression regime (channel QF > cover QF), so the toolchain is behaving
correctly end to end.

**All four mex binaries load under R2024b** — `jpeg_read`, `jpeg_write`,
`stc_pm1_pls_embed`, `stc_ml_extract`. This was the main risk and it is retired.

The single most important project risk — unreproducible baselines — is closed.

## Per-baseline readiness

Only GMAS is self-contained. The rest were written for an HPC cluster and need
scaffolding. None of these are blockers, but budget for them.

| Method | mex present | Covers bundled | Runs as-is | What it needs |
|---|---|---|---|---|
| **GMAS** | JPEG_Toolbox + STC3 | 10 @ QF65 | **yes** | nothing |
| **ROAST** | STC3 only | no | **runs but does not reproduce** | see the ROAST section below — three defects, needs reconstruction |
| **MINICER** | STC3 only | no | **WORKS** | raw BER 1.78% at `stego_step`=6 (shipped value 1 is a demo); no ECC by design — see below |
| **PMAS** | none | no | **EXCLUDED** | 3 fatal defects; post-processing converges to ~1.8% coefficient mismatch, fatal for STC. Diagnosed, documented, authors to be contacted. See `patched/PMAS/README.md` |
| **SCRIPT** | none | no | **WORKS** (same-QF channel) | needs no patch; errorless at Q_channel = Q_cover; `distortion=1` unrunnable — see below |

### Smoke-test results (`experiments/m1_smoketest_others.m`)

Run: `matlab -batch "m1_smoketest_others('<repo>','<out>')"`

- **MINICER** — the toolchain resolves (jpeg_read works, STC runs), so the
  caller-side path fix is correct. But extraction failed with *"stc extracted
  wrong msg"*, then crashed on `Unrecognized function or variable 'wet_rate'` at
  `Test_QMAS` line 75. That undefined variable is **a bug in the published
  code**, sitting in the failure-reporting branch — it only fires when extraction
  has already failed, so it masks the real error rather than causing it. The
  actual problem is a parameter mismatch: GMAS's QF65 covers are being fed to
  MINICER with settings it does not expect. Needs its own correctly configured
  covers, not a different code fix.
- **PMAS** — fails immediately at `parpool`, see the toolbox finding below.
- **SCRIPT** — hardcoded Linux path confirmed present.
- **ROAST** — deferred; needs a precover set, which needs BOSSbase.

### Toolbox finding: no Parallel Computing Toolbox

```
installed: Communications, DSP System, Fuzzy Logic, Image Processing,
           MATLAB, Signal Processing, Simulink
parpool available: 0        blockproc available: 1        numcores: 12
```

`license('test','distrib_computing_toolbox')` returns 1, but PCT is **not
installed** — licensed and installed are different things, and the license test
alone would have been misleading.

Consequences:

- **`parfor` still runs, but serially.** MATLAB degrades `parfor` to a plain
  loop without PCT. Any speedup assumed from `parfor` is imaginary.
- **PMAS's explicit `parpool(...)` call errors outright** and cannot be fixed
  from the caller; it needs a patched copy with that line removed.
- **Statistics and Machine Learning Toolbox is also absent** from `ver`. Check
  before relying on it for the M2 FLD ensemble — Kodovský's reference ensemble
  is usually self-contained, but confirm rather than assume.

**Parallelisation: measured, not assumed.** Shard the image list across N MATLAB
processes launched from Python (`experiments/exp03_grid_runner.py`). Measured on
**200 real BOSSbase covers** at QF65, GMAS payload 0.1:

| Workers | Wall (200 covers) | s/image | Speedup |
|--------:|------------------:|--------:|--------:|
| 1 | 420.7 s | 2.10 | 1.00× |
| **4** | **154.9 s** | **0.77** | **2.72×** |
| 6 | 151.4 s | 0.76 | 2.78× |

**Use 4 workers.** 6 is within noise of 4, and an earlier 48-cover run showed 8
and 12 workers are *slower* than 4. The limit is core contention, not memory:
MATLAB multithreads its own BLAS and image built-ins, so a few processes already
saturate 12 physical cores. `-singleCompThread` was tested and made no
measurable difference (2.18 vs 2.12 s/image solo), so the flag is not the lever.

Two earlier estimates in this project were wrong and are superseded: "~12–16 h
via `parfor`" (ignored that PCT is absent) and "~27 h via 6 processes" (assumed
linear scaling). Both were written before measuring.

**Grid budget:** 2,000 images/cell × 84 cells at 0.77 s/image ≈ **36 h** of
embedding with 4 workers (117 h serial).

## GMAS is not errorless on real covers — and the bundled demo hides it

`experiments/exp04_overflow_analysis.py`. The M1 gate reproduced
`ave_error_rate: 0` on the ten covers bundled with the repo. On 200 real
BOSSbase covers at the *same* operating point:

```
errorless : 152/200 (76%)      mean BER 0.00620   max 0.2086
```

This is **not a harness bug**. Dither modulation is errorless under upward
recompression only if no pixel is driven outside [0,255]; spatial overflow
breaks the guarantee. Failure tracks near-saturation monotonically:

| Cover saturation quartile | Errorless |
|---|---|
| Q1 (0.0000–0.0017) | **50/50 (100%)** |
| Q2 (0.0017–0.0100) | 44/50 (88%) |
| Q3 (0.0100–0.0393) | 32/50 (64%) |
| Q4 (0.0393–0.7137) | **26/50 (52%)** |

Mean saturation is 3× higher among failing images (0.0828 vs 0.0281). This is
exactly the premise ROAST and Cheng et al.'s boundary-preserving work were built
on, so reproducing it independently **validates** the pipeline.

And the demo set is unrepresentative: its *maximum* saturation (0.0680) is only
the **84th percentile** of BOSSbase, so all ten bundled covers avoid the
overflow-prone tail entirely.

## ROAST as shipped does not reproduce — three defects

`experiments/roast_embed_shard.m` reimplements `TestRobustnessURA_JUNIWARD.m`
faithfully, parameterised and manifest-driven. On 30 BOSSbase covers, payload
0.1, QF 65 → 71 — the same operating point where GMAS gets 76% errorless and
mean BER 0.0062:

| CostUR parameters | Errorless | Mean BER |
|---|---|---|
| α=0, β=1 (**as shipped in their driver**) | **0/30** | 0.1108 |
| α=0.2, β=0.7 (CostUR's own suggested values) | **0/30** | 0.1080 |

ROAST is supposed to *improve* on GMAS. It comes out ~18× worse. Three distinct
problems, in increasing order of severity:

**1. `CostUR(...,0,1)` is a mathematical no-op.** With α=0 line 32 adds
`0*rhoPMean`; with β=1 line 35 multiplies by 1. The costs pass through
untouched, so the shipped driver disables ROAST's cost-adjustment contribution
entirely. CostUR's own comments suggest 0.2 and 0.7. Measured above: switching
to them changes almost nothing, so this is not the main problem.

**2. Failures are scored as perfect successes.** `TestRobustnessURA_JUNIWARD.m`
lines 76–79 catch any embed/extract exception and set `bit_error_rate(i)=0`,
which is then averaged in. MINICER's `Test_QMAS` does the same — that is why its
smoke test printed `ave_error_rate: 0` immediately after "stc extracted wrong
msg". Any published average BER produced by these drivers is biased downward by
however many images failed outright.

**3. ROAST/ has no dither modulation at all.** `grep -ri "gmas\|dither"
ROAST/` returns nothing, while GMAS's `gmas.m` is precisely the generalized
dither modulation that makes coefficients quantisation-stable — the actual
source of GMAS's robustness. The shipped ROAST driver embeds with plain
J-UNIWARD costs into an overflow-suppressed cover, with no robustness mechanism
in the pipeline. That fully explains BER ≈ 0.11.

The literature states ROAST "operates within the framework of the GMAS
algorithm", so the intended pipeline is almost certainly
`PreprocessURA` (ROAST) → `gmas()` DM (GMAS) → `CostUR` → STC. **The shipped
driver omits the DM step.** Note also that the sibling driver
`TestRobustnessUR_JUNIWARD.m` is the more careful of the two: it preallocates
`bit_error_rate`, uses `for` rather than `parfor`, and seeds `rand('seed',0)`.

### RESOLVED — reconstruction works

`experiments/roast_recon_shard.m` composes the pipeline the literature implies:

```
PreprocessURA (ROAST)  ->  J_UNIWARD_Asy_cost  ->  CostUR (ROAST)
                       ->  gmas() DM (GMAS)    ->  stc3_embed (GMAS)
```

i.e. GMAS's dither modulation is fed the ROAST-adjusted costs. GMAS's and
ROAST's copies of `J_UNIWARD_Asy_cost.m` and `rs_encode_yxz.m` are byte-
identical, so path order between them is immaterial; the STC wrappers differ in
name (`stc3_embed` vs `stc3Embed`) so there is no clash.

Head to head on 30 identical BOSSbase covers, payload 0.1, QF 65 → 71
(`experiments/exp05_roast_reconstruction.py`), split at the median cover
saturation:

| Method | Errorless | Mean BER | Low-sat half | High-sat half |
|---|---|---|---|---|
| GMAS | 20/30 | 0.00443 | 14/15 | **6/15** |
| ROAST shipped (α=0, β=1) | 0/30 | 0.11082 | 0/15 | 0/15 |
| ROAST shipped (α=0.2, β=0.7) | 0/30 | 0.10801 | 0/15 | 0/15 |
| **ROAST reconstructed** | **30/30** | **0.00000** | **15/15** | **15/15** |

The reconstruction does not merely work — it reproduces **ROAST's actual
claim**. GMAS collapses to 6/15 on the high-saturation half, which is precisely
the overflow regime quantified in `exp04_overflow_analysis.py`; overflow
suppression repairs it completely. The method is sound; only the shipped driver
is broken.

### Confirmed across payloads and channels (`exp06_roast_sweep.py`)

20 BOSSbase covers, Q_cover 65, split at the median cover saturation (0.0088).

**Q_channel = 71** — the reconstruction holds, and the gap *widens* with payload:

| Payload | GMAS errorless | GMAS high-sat | ROAST errorless | ROAST high-sat |
|---|---|---|---|---|
| 0.05 | 14/20 | 6/10 | **20/20** | **10/10** |
| 0.10 | 14/20 | 5/10 | **20/20** | **10/10** |
| 0.20 | 13/20 | 4/10 | **20/20** | **10/10** |
| 0.30 | 13/20 | 3/10 | 19/20 | 9/10 |

GMAS's low-saturation column stays flat (8, 9, 9, 10 of 10) while its
high-saturation column degrades monotonically (6, 5, 4, 3). **All of GMAS's
payload-dependent failure is in the overflow-prone half of the cover source**,
and ROAST's overflow suppression removes it entirely up to payload 0.2.

**Q_channel = 95 — the advantage nearly vanishes:**

| Payload | GMAS errorless | GMAS high-sat | ROAST errorless | ROAST high-sat |
|---|---|---|---|---|
| 0.10 | **20/20** | 10/10 | 20/20 | 10/10 |
| 0.30 | 19/20 | 9/10 | **20/20** | **10/10** |

At QF 95 the quantisation step is small enough that recompression barely
perturbs coefficients and spatial overflow is rare, so plain GMAS is already
essentially errorless and there is little left for ROAST to fix.

**ROAST's benefit is channel-dependent, and that belongs in the paper.** It is
large where the channel is aggressive relative to the cover (QF 65→71) and
negligible where it is mild (QF 65→95). Reporting only one channel would
misrepresent the method in either direction — which is a concrete instance of
the threat-model point made further down.

**Usable as a baseline, with the reconstruction documented.** Every ROAST number
in the paper must be labelled as coming from the reconstructed pipeline, with
the three shipped-driver defects stated. Two caveats before scaling up:

- n = 30 at a single payload and a single QF pair. Confirm the result holds
  across the payload sweep and at Q_channel = 95 before relying on it.
- α = 0.2, β = 0.7 are CostUR's suggested values, not fitted. Since the shipped
  α = 0, β = 1 is a no-op, they were never validated by the authors' own driver
  either. Worth a small sweep.

## MINICER runs, but its BER is not comparable to GMAS/ROAST

`experiments/minicer_embed_shard.m`. Two fixes were needed to get it running at
all, both caller-side:

- `Test_QMAS` assigns `wet_rate` only inside its `try` block but reads it
  unconditionally at line 75. If the *first* image fails, the run dies with
  `Unrecognized function or variable 'wet_rate'` — which masks the real error
  and is exactly what the smoke test hit. Initialised.
- Same catch-sets-BER-to-0 defect as ROAST. Failures recorded as failures.

Verified first: MINICER *computes* its quantization table from the QF via
`quantizationTable()` instead of reading it from the file, unlike GMAS and
ROAST. `quantizationTable(65)` and `quantizationTable(71)` are **identical** to
the tables in our covers and in a QF71 recompression, so the conventions agree
and the same cover set is valid. And `generateRandMsg` uses
`ceil(payload*nzAC)` — the same payload definition as GMAS, so payloads are
directly comparable.

On the same 30 covers, payload 0.1, QF 65 → 71: **0/30 errorless, BER ≈
0.12–0.20**.

**But that number must not be put in a table next to GMAS's.** `grep -rl
"rs_encode|rs_decode|bch|comm\." MINICER/` returns nothing: **MINICER has no
error-correcting code**. Its BER is the *raw channel error rate*; GMAS's and
ROAST's are *post-RS(31,15)*. Different quantities.

For reference, RS(31,15) over GF(2⁵) corrects 8 symbol errors in 31, so it needs
a symbol error rate below ~0.26. A raw BER of 0.15 gives a symbol error rate of
1−0.85⁵ ≈ 0.56 — far beyond repair. So MINICER at this operating point would not
be rescued by adding RS either, which points at the configuration rather than
the method. The prime suspect is `stego_step = ones(8,8)` in the shipped driver:
that is the QIM step size, and a step of 1 is the least robust setting possible.
**Open question — sweep `stego_step` before quoting any MINICER number.**

### RESOLVED — `stego_step` is the robustness knob, and MINICER works

`stego_step` is the QIM step size per DCT position. The shipped driver hardcodes
`ones(8,8)` — step 1, the least robust setting available, and evidently a demo
value rather than an operating point. Swept on 6 BOSSbase covers, payload 0.1,
QF 65 → 71:

| `stego_step` | Mean raw BER | PSNR |
|---|---|---|
| 1 (**as shipped**) | 0.1310 | — |
| 2 | 0.0418 | — |
| 3 | 0.0333 | — |
| 4 | 0.0184 | — |
| **6** | **0.0178** | 39.77 dB |
| 8 | 0.0776 | 37.05 dB |
| 12 | 0.0216 | 33.34 dB |

Robustness improves sharply to an optimum around step 4–6, then degrades while
PSNR keeps falling — the robustness/imperceptibility trade-off made visible in
one sweep.

**MINICER never reaches errorless, and that is correct by design.** The name is
*MINImise ChannEl eRror*: it minimises the **raw** channel error and leaves
error correction to the application. At its optimum the raw BER is 1.78%, and an
RS(31,15) outer code — the one GMAS and ROAST use — corrects that comfortably:
5-bit symbols give a symbol error rate of 1 − (1 − 0.0178)⁵ ≈ 0.086, against
RS(31,15)'s correction capacity of 8/31 ≈ 0.258. A 3× margin.

This closes the comparability question below: MINICER's numbers are raw BER by
construction, and once an outer code is applied it lands in the same errorless
regime as GMAS and ROAST.

### Protocol consequence: report raw BER *and* post-ECC BER

The cleanest fix for the whole comparison, and it should go in the paper's
protocol: record the pre-ECC bit error rate alongside the post-ECC one for
every method. That separates two things the literature routinely conflates —
how well the embedding survives the channel, versus how much the outer code
repairs — and it is the only way MINICER (no ECC) and GMAS/ROAST (RS(31,15))
can appear in the same table honestly. It also directly serves the redundancy-
relocation argument in `RESEARCH_DOSSIER.md` Part V, which is precisely a claim
about where error-correction effort should live.

## PMAS cannot process a single image as shipped

Three independent defects, full detail in `patched/PMAS/README.md`:

1. **`JPEGrecompress` called with 3 of 5 arguments** at every call site
   (`preprocessUNICORE:27`, `postprocessUNICORE:9`). `color` is undefined, so
   line 4's `if color` throws on the first image and every one after.
2. **`J_UNIWARDcost` is missing from `PMAS/` entirely.** It ships in `MINICER/`
   and `ROAST/` (byte-identical copies). `UERDcost` likewise.
3. **`testRobustUNICORE:52` uses `stego_Path`/`afterchannel_stego_Path`** while
   the loop defines them lowercase (lines 28–29). MATLAB is case-sensitive.

Any one of these is fatal. Combined with the catch-scores-failures-as-zero
pattern, **the shipped driver processes zero images successfully while printing
`ave_error_rate: 0`** — flawless robustness, with no extraction ever performed.
This is the most severe defect found in the repo, and it is in code released
alongside an IEEE TCSVT paper.

Patched copies live in `baselines/patched/PMAS/` (tracked in git, since a paper
citing a modified implementation must ship the modification). With them the
pipeline executes cleanly at ~2.9 s/image.

**But it is not yet correct.** At `T=8, mode=[2 1 1], distortion=1` — parameters
we guessed — extraction returns **BER ≈ 0.50**, pure chance. The repo README
documents what `T` and `mode` mean but not the values behind the published
results, and notes that `preprocessUNICORE` can use GDM or PDM with the GDM path
commented out. Read the TCSVT paper for the parameters, or sweep. **Do not quote
a PMAS number until extraction works.**

## SCRIPT — runs, but the driver's extraction is provably wrong

`experiments/script_embed_shard.m`. No patched copy needed: the hardcoded Linux
path and the SLURM `parcluster`/`parpool` block live only in the *driver*, which
this replaces, and `afterchannel_cover_path` is assigned at
`testRobustGISS.m:54` and never read — the path is vestigial.

**Defect A — the README contradicts the code.** The README states
`distortion: =1 for JUNIWARD, =2 for UERD`. But `costGISS.m:15` is
`if distortion` — a *truthiness* test, not `== 2`. So `distortion=2` takes the
J-UNIWARD branch, and UERD requires `distortion = 0`. (`costGISS.m:3` even
carries a stale `% distortion = 2;` comment.) Line 22's `if distortion<=1`
additionally gates the side-information adjustment, so the parameter has three
behaviours, not the two documented: 0 = UERD + side-info, 1 = J-UNIWARD +
side-info, 2 = J-UNIWARD without it.

**Defect B — `distortion=1` (J-UNIWARD) cannot run from this repo at all.**
`costGISS.m:17` calls `J_UNIWARD_cost(spa, coeffs_real, QUANT)` — three
arguments, taking in-memory spatial data and real-valued coefficients. Every
J-UNIWARD implementation shipped anywhere in the repo takes a single file path
(`J_UNIWARDcost(coverPath)`, `J_UNIWARD_Asy_cost(coverpath)`), so none can
substitute. The README acknowledges this class of gap outright: *"If other
essential functions are missing, please look for them in my other
repositories."* Only the UERD path is reachable.

**Defect C — embed/extract index mismatch, confirmed empirically.**
`generateStegoGISS.m:9-13` selects `is_cover==1` when `R3==0` and the superset
`is_cover>0` otherwise. The driver's extraction (`testRobustGISS.m:~92`)
**always** uses `is_cover==1`. So for any `R3≠0` it extracts from a different
index set than it embedded into. Measured on a clean stego, no channel at all:

| Extraction index | BER on clean stego |
|---|---|
| matched to the embedder (`is_cover>0`, R3=2) | **0** |
| the driver's (`is_cover==1`) | **0.493** |

Chance-level before the channel is even applied — and the catch-scores-zero
pattern would then report it as `ave_error_rate: 0`.

### RESOLVED — SCRIPT works; two of our assumptions were wrong

Both mistakes were ours, not the code's.

**Wrong parameter scale.** `testRobustGISSttt.m:61-63` runs `for r = 0:R3` with
`R1 = r; R2 = r`, so **R1 and R2 are integers**, not the fractional 0.1 we
guessed. `coverSelect.m:25` excludes coefficients by `dis_to_border < R1`, where
`dis_to_border` is on the scale of `QUANT/2` (~4–42 at QF65), so single-digit
values barely filter and the knob only bites around R1 ≈ 10.

**Wrong threat model.** GMAS's "Facebook" scenario (QF 65 → 71, upward
recompression) is not SCRIPT's. SCRIPT targets *black-box generated images*,
where the generated image is JPEG-compressed and the channel recompresses at the
**same** quality factor.

Measured on 6 BOSSbase covers, payload 0.1, cover QF 65, `R3=0, distortion=0`
(UERD — the J-UNIWARD path is unrunnable, defect B):

| R1=R2 | Attack QF | Usable coeffs | Errorless | Mean BER |
|---|---|---|---|---|
| 0 | **65** | 86.3% | **6/6** | **0.00000** |
| 10 | **65** | 50.5% | **6/6** | **0.00000** |
| 0 | 71 | 86.3% | 0/6 | 0.50720 |
| 10 | 71 | 50.5% | 0/6 | 0.13139 |

SCRIPT is **errorless at same-QF recompression**, and R1 behaves exactly as
`coverSelect` implies: raising it from 0 to 10 at the off-target QF 71 cuts BER
from chance (0.507) to 0.131. The pipeline is correct and the harness is right.

### Protocol consequence: the methods do not share a threat model

This is the most important protocol finding of M1 and it reshapes the grid:

| Method | Channel it targets |
|---|---|
| GMAS, ROAST | **Upward** recompression, Q_cover < Q_channel (the "Facebook" QF 65→71 scenario) |
| SCRIPT | **Same-QF** recompression of generated images |
| PMAS | High-quality images (per its title); operating point still to confirm |

Evaluating all of them on one channel setting would be an unfair comparison that
flatters whichever method the setting was chosen for — and would look, to a
reviewer, exactly like a rigged benchmark. **Each method must be evaluated on
the channel it was designed for, with cross-channel results reported separately
as a generalisation study** (which is itself a result worth having, since the
robust-steganography literature rarely reports it).

## The pattern across the baselines

Every driver examined so far is demo-grade rather than evaluation-grade:

| Method | Defects found |
|---|---|
| GMAS | bundled 10-image demo set avoids the overflow-prone tail (max saturation = 84th percentile of BOSSbase) |
| ROAST | no dither-modulation step at all; `CostUR(...,0,1)` is a mathematical no-op; failures scored as BER=0 |
| MINICER | `wet_rate` read outside the `try` that defines it, so a first-image failure kills the run and masks the error; failures scored as BER=0; no ECC, so its BER is not comparable |
| PMAS | `JPEGrecompress` called with 3 of 5 args at every call site; `J_UNIWARDcost`/`UERDcost` missing from the folder; case-mismatched variable in the driver; failures scored as BER=0 |
| SCRIPT | README contradicts code on `distortion`; `J_UNIWARD_cost` (3-arg) missing with no substitute, so the J-UNIWARD path cannot run; embed/extract index mismatch gives BER 0.493 on a *clean* stego; failures scored as BER=0 |

**Four of the five cannot produce a correct number as shipped.** Only GMAS
runs correctly out of the box, and even its bundled demo set is unrepresentative.
PMAS cannot process a single image while still printing `ave_error_rate: 0`;
SCRIPT extracts from the wrong index set and would report the same.

The common enabler is the `catch → bit_error_rate = 0` idiom, present in four
drivers. It converts every crash, every missing dependency and every index
mismatch into a contribution of *zero error* to the reported mean. Without it,
each of these defects would have surfaced on the authors' first run. None of this means
the *methods* are wrong — the ROAST reconstruction proves its method works
exactly as claimed. It means published robustness numbers in this subfield rest
on code that does not survive contact with a full benchmark, which is the
substance of Paper 1.

Two consequences for the papers:

1. Never quote a robustness number from a bundled demo. Re-measure on the full
   benchmark — this is a concrete instance of the reproducibility gap Paper 1
   is about.
2. Report robustness **conditioned on cover saturation**, not just as a mean.
   A single averaged BER hides a 100%-to-52% spread across the cover source, and
   that spread is itself a result.

If the institutional licence covers PCT (the licence test suggests it does),
installing it through the MATLAB Add-On Explorer would make `parfor` work
natively — but given that scaling saturates at 4 workers, expect it to buy
little. That is a user decision, not a code change.

### How to fix without touching the clone

The ground rules (`README.md`) say never edit `baselines/` — a silently patched
baseline would make every comparison unattributable. So:

- **Path/cover problems** — solve from the caller. A wrapper adds
  `GMAS/JPEG_Toolbox` and `GMAS/STC3` to the MATLAB path and passes explicit
  directory arguments. No source change needed for ROAST and MINICER.
- **`SLURM_CPUS_PER_TASK`** — set the environment variable before launching
  MATLAB (`setenv`), so `parpool` gets a real number. No source change.
- **SCRIPT's hardcoded absolute path** — cannot be fixed from outside. Make a
  patched copy under `baselines/patched/SCRIPT/`, record every changed line in a
  diff committed alongside it, and cite the patched version in the paper. Never
  patch the pristine clone in place.

## Timing, for the M3 grid budget

GMAS ran 10 images in 34.6 s ⇒ **~3.5 s/image** (J-UNIWARD cost + GMAS
preprocessing + STC embed + recompress + extract + RS decode).

At the literature-standard 2,000 images per cell that is **~1.9 h per grid
cell**, and the 84-cell grid is **~160 h ≈ 6.7 days** of embedding alone — before
any steganalysis. Two ways down:

- The GMAS test loops serially; ROAST and PMAS already use `parfor`. Wrapping
  the embedding loop in `parfor` across 16 logical cores should bring the grid
  to roughly 12–16 h.
- Some published work uses 1,000 images rather than 2,000. Halving is defensible
  if stated, but match Cheng et al.'s 2,000 wherever a number is compared
  directly against theirs.

This cost was not in `PLAN.md`'s original M3 estimate, which counted only
steganalysis (~18 h). **M3 is embedding-bound, not steganalysis-bound.**

## Next

1. Point ROAST and MINICER at BOSSbase covers via a wrapper; smoke-test MINICER
   first (`cover_num = 1`).
2. Set `SLURM_CPUS_PER_TASK` and smoke-test PMAS.
3. Patch-copy SCRIPT.
4. Wrap embedding in `parfor` before running anything at grid scale.
