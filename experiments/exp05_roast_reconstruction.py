"""ROAST: shipped driver vs reconstructed pipeline, head to head.

The repo's ROAST driver does not reproduce — 0/30 errorless at BER ~0.11 where
GMAS gets 20/30 at 0.0044. Cause: `ROAST/` contains no dither-modulation step
(`grep -ri "gmas|dither" ROAST/` is empty), so it embeds plain J-UNIWARD costs
into an overflow-suppressed cover with no robustness mechanism in the loop.

Inserting GMAS's `gmas()` between `CostUR` and the STC — what the literature's
"ROAST operates within the framework of the GMAS algorithm" implies — makes it
work, and reproduces ROAST's actual claim: overflow suppression repairs exactly
the saturated-cover failures that defeat GMAS.

Runs four configurations over one identical cover set and splits results by
cover saturation, because the mean alone hides the effect entirely.

    python experiments/exp05_roast_reconstruction.py --count 30
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import shutil
import statistics
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = paths.BASELINES / "Robust_Steganography"

# (label, matlab function, alpha, beta) -- alpha/beta ignored by the GMAS driver
CONFIGS = [
    ("GMAS", "gmas_embed_shard", None, None),
    ("ROAST shipped (a=0,b=1)", "roast_embed_shard", 0.0, 1.0),
    ("ROAST shipped (a=.2,b=.7)", "roast_embed_shard", 0.2, 0.7),
    ("ROAST RECONSTRUCTED", "roast_recon_shard", 0.2, 0.7),
]


def find_matlab() -> str | None:
    exe = shutil.which("matlab")
    if exe:
        return exe
    f = sorted(pathlib.Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"))
    return str(f[-1]) if f else None


def saturation(path: pathlib.Path) -> float:
    a = np.asarray(Image.open(path).convert("L"))
    return float(((a <= 4) | (a >= 251)).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers", type=pathlib.Path,
                    default=paths.DATA / "BOSSbase_QF65_timing")
    ap.add_argument("--count", type=int, default=30)
    ap.add_argument("--payload", type=float, default=0.1)
    ap.add_argument("--cover-qf", type=int, default=65)
    ap.add_argument("--attack-qf", type=int, default=71)
    args = ap.parse_args()

    matlab = find_matlab()
    if not matlab:
        print("MATLAB not found")
        return 1

    out = paths.ensure_results("exp05_roast")
    covers = sorted(args.covers.resolve().glob("*.jpg"))[:args.count]
    if not covers:
        print(f"no covers in {args.covers}")
        return 1

    manifest = out / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in covers), encoding="utf-8")

    results: dict[str, dict[str, float | None]] = {}
    for label, fn, alpha, beta in CONFIGS:
        slug = label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")
        csv_path = out / f"{slug}.csv"
        if fn == "gmas_embed_shard":
            call = (f"{fn}('{REPO.as_posix()}','{manifest.as_posix()}',"
                    f"'{(out / slug).as_posix()}','{csv_path.as_posix()}',"
                    f"{args.payload},{args.cover_qf},{args.attack_qf},0)")
        else:
            call = (f"{fn}('{REPO.as_posix()}','{manifest.as_posix()}',"
                    f"'{(out / slug).as_posix()}','{csv_path.as_posix()}',"
                    f"{args.payload},{args.cover_qf},{args.attack_qf},0,{alpha},{beta})")
        print(f"running {label}...", flush=True)
        subprocess.run([matlab, "-batch", call], cwd=str(HERE),
                       capture_output=True, text=True)
        if not csv_path.exists():
            print(f"  no output for {label}")
            continue
        with csv_path.open(newline="", encoding="utf-8") as fh:
            results[label] = {r["cover"]: (float(r["bit_error_rate"])
                                           if r["bit_error_rate"] else None)
                              for r in csv.DictReader(fh)}

    if not results:
        return 1

    keys = sorted(set.intersection(*[set(v) for v in results.values()]))
    sats = {k: saturation(args.covers / f"{k}.jpg") for k in keys}
    median_sat = float(np.median(list(sats.values())))

    print(f"\n=== {len(keys)} identical BOSSbase covers, payload {args.payload}, "
          f"QF {args.cover_qf} -> {args.attack_qf} ===\n")
    hdr = f"{'method':<28}{'errorless':>11}{'mean BER':>11}{'low-sat':>10}{'high-sat':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, d in results.items():
        vals = [d[k] for k in keys if d[k] is not None]
        lo = [d[k] for k in keys if d[k] is not None and sats[k] <= median_sat]
        hi = [d[k] for k in keys if d[k] is not None and sats[k] > median_sat]
        frac = lambda v: f"{sum(1 for x in v if x == 0)}/{len(v)}"  # noqa: E731
        print(f"{label:<28}{frac(vals):>11}{statistics.mean(vals):>11.5f}"
              f"{frac(lo):>10}{frac(hi):>10}")

    print("\nSplit at the median cover saturation. The high-sat column is the "
          "point:\nGMAS degrades there, and overflow suppression is what "
          "repairs it -- but only\nonce the dither-modulation step the shipped "
          "driver omits is put back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
