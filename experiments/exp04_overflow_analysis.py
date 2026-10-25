"""Why GMAS is not errorless on real covers, and why the bundled demo hides it.

The M1 gate reproduced `ave_error_rate: 0` on the ten covers bundled with the
baselines repo. On 200 real BOSSbase covers at the same operating point
(payload 0.1, Q_cover 65 -> Q_channel 71) only ~76% of images are errorless and
the mean BER is ~0.006.

That is not a harness bug. Dither modulation is errorless under upward
recompression *only if no pixel is driven outside [0,255]*; spatial overflow
breaks the guarantee, and overflow happens in covers with many near-saturated
pixels. This is precisely the premise ROAST and Cheng et al.'s boundary-
preserving work were built on, so reproducing it independently validates the
pipeline rather than impugning it.

This script quantifies the effect and shows the bundled demo set sits entirely
in the benign regime.

    python experiments/exp04_overflow_analysis.py \
        --results results/exp03_grid/gmas_p0.1_q65to71_w1 \
        --covers  data/BOSSbase_QF65_timing
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

SAT_LO, SAT_HI = 4, 251  # a pixel this close to a clipping bound may overflow


def saturation(path: pathlib.Path) -> float:
    """Fraction of pixels close enough to 0 or 255 to risk clipping."""
    a = np.asarray(Image.open(path).convert("L"))
    return float(((a <= SAT_LO) | (a >= SAT_HI)).mean())


def load_results(results_dir: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for csv_path in sorted(results_dir.glob("shard*/results.csv")):
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows.extend(list(csv.DictReader(fh)))
    return [r for r in rows if r.get("bit_error_rate")not in (None, "")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path,
                    default=paths.RESULTS / "exp03_grid" / "gmas_p0.1_q65to71_w1")
    ap.add_argument("--covers", type=pathlib.Path,
                    default=paths.DATA / "BOSSbase_QF65_timing")
    ap.add_argument("--demo", type=pathlib.Path,
                    default=paths.BASELINES / "Robust_Steganography" / "GMAS" / "cover_dir_QF65")
    args = ap.parse_args()

    rows = load_results(args.results)
    if not rows:
        print(f"no decoded results under {args.results}\n"
              f"run exp03_grid_runner.py first")
        return 1

    ber = np.array([float(r["bit_error_rate"]) for r in rows])
    sat = np.array([saturation(args.covers / f"{r['cover']}.jpg") for r in rows])

    print(f"=== GMAS on {len(rows)} real BOSSbase covers ===")
    print(f"  errorless : {(ber == 0).sum()}/{len(ber)} ({(ber == 0).mean() * 100:.0f}%)")
    print(f"  mean BER  : {ber.mean():.5f}   median {np.median(ber):.5f}   "
          f"max {ber.max():.4f}")

    print("\n=== near-saturated pixel fraction vs extraction failure ===")
    print(f"  corr(saturation, BER)  : {np.corrcoef(sat, ber)[0, 1]:+.3f}")
    print(f"  mean sat | errorless   : {sat[ber == 0].mean():.4f}")
    print(f"  mean sat | has errors  : {sat[ber > 0].mean():.4f}")

    q = np.quantile(sat, [0, .25, .5, .75, 1.0])
    print("\n  by saturation quartile:")
    for i in range(4):
        m = (sat >= q[i]) & (sat <= q[i + 1])
        print(f"    Q{i + 1} ({q[i]:.4f}-{q[i + 1]:.4f}): "
              f"errorless {(ber[m] == 0).sum():>3}/{m.sum():<3} "
              f"({(ber[m] == 0).mean() * 100:>3.0f}%)")

    if args.demo.exists():
        demo = np.array([saturation(p) for p in sorted(args.demo.glob("*.jpg"))])
        pct = float((sat < demo.max()).mean() * 100)
        print(f"\n=== why the bundled demo set reads 0.0 ===")
        print(f"  demo covers (n={len(demo)}): mean {demo.mean():.4f}  max {demo.max():.4f}")
        print(f"  BOSSbase    (n={len(sat)}): mean {sat.mean():.4f}  max {sat.max():.4f}")
        print(f"  the demo set's MAXIMUM saturation is only the {pct:.0f}th percentile")
        print(f"  of BOSSbase -- every demo cover avoids the overflow-prone tail.")

    print("\nTakeaway: a published method's bundled demo can be unrepresentative.")
    print("Always re-measure on the full benchmark before quoting a robustness number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
