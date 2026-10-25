"""Does the ROAST reconstruction hold beyond the single point it was found at?

`exp05` established the reconstruction at one operating point (payload 0.1,
QF 65 -> 71): 30/30 errorless vs GMAS's 20/30, and 15/15 vs 6/15 on
high-saturation covers. Paper 1 leans on that result harder than any other, so
it needs to hold across payloads and channels before anything is built on it.

Sweeps GMAS and reconstructed ROAST over payload x channel on one shared cover
set, and splits every cell by cover saturation -- the axis on which the two
methods actually differ.

    python experiments/exp06_roast_sweep.py --count 20
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

METHODS = [
    ("GMAS", "gmas_embed_shard"),
    ("ROAST-recon", "roast_recon_shard"),
]


def find_matlab() -> str | None:
    exe = shutil.which("matlab")
    if exe:
        return exe
    f = sorted(pathlib.Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"))
    return str(f[-1]) if f else None


def saturation(p: pathlib.Path) -> float:
    a = np.asarray(Image.open(p).convert("L"))
    return float(((a <= 4) | (a >= 251)).mean())


def run(matlab: str, fn: str, manifest: pathlib.Path, out: pathlib.Path,
        csv_path: pathlib.Path, payload: float, cqf: int, aqf: int) -> bool:
    if fn == "gmas_embed_shard":
        call = (f"{fn}('{REPO.as_posix()}','{manifest.as_posix()}',"
                f"'{out.as_posix()}','{csv_path.as_posix()}',{payload},{cqf},{aqf},0)")
    else:
        # alpha/beta = CostUR's own suggested values; the shipped 0/1 is a no-op.
        call = (f"{fn}('{REPO.as_posix()}','{manifest.as_posix()}',"
                f"'{out.as_posix()}','{csv_path.as_posix()}',{payload},{cqf},{aqf},"
                f"0,0.2,0.7)")
    subprocess.run([matlab, "-batch", call], cwd=str(HERE),
                   capture_output=True, text=True)
    return csv_path.exists()


def load(csv_path: pathlib.Path) -> dict[str, float | None]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return {r["cover"]: (float(r["bit_error_rate"]) if r["bit_error_rate"]
                             else None)
                for r in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers", type=pathlib.Path,
                    default=paths.DATA / "BOSSbase_QF65_timing")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--cover-qf", type=int, default=65)
    ap.add_argument("--payloads", type=float, nargs="+",
                    default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--channels", type=int, nargs="+", default=[71, 95])
    args = ap.parse_args()

    matlab = find_matlab()
    if not matlab:
        print("MATLAB not found")
        return 1

    out_root = paths.ensure_results("exp06_roast_sweep")
    covers = sorted(args.covers.resolve().glob("*.jpg"))[:args.count]
    manifest = out_root / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in covers), encoding="utf-8")

    sats = {p.stem: saturation(p) for p in covers}
    med = float(np.median(list(sats.values())))

    print(f"{len(covers)} covers, Q_cover={args.cover_qf}, "
          f"payloads={args.payloads}, channels={args.channels}")
    print(f"saturation median {med:.4f} (split point for the low/high columns)\n")

    hdr = (f"{'channel':>8}{'payload':>9}  {'method':<13}"
           f"{'errorless':>11}{'mean BER':>11}{'low-sat':>9}{'high-sat':>10}")
    print(hdr)
    print("-" * len(hdr))

    for aqf in args.channels:
        for payload in args.payloads:
            row_results = {}
            for label, fn in METHODS:
                slug = f"q{aqf}_p{payload}_{label.replace('-', '_')}"
                csv_path = out_root / f"{slug}.csv"
                if not csv_path.exists():
                    run(matlab, fn, manifest, out_root / slug, csv_path,
                        payload, args.cover_qf, aqf)
                if csv_path.exists():
                    row_results[label] = load(csv_path)

            for label, d in row_results.items():
                vals = [v for v in d.values() if v is not None]
                if not vals:
                    print(f"{aqf:>8}{payload:>9}  {label:<13}{'all failed':>11}")
                    continue
                lo = [v for k, v in d.items() if v is not None and sats.get(k, 0) <= med]
                hi = [v for k, v in d.items() if v is not None and sats.get(k, 0) > med]
                f = lambda v: f"{sum(1 for x in v if x == 0)}/{len(v)}"  # noqa: E731
                print(f"{aqf:>8}{payload:>9}  {label:<13}{f(vals):>11}"
                      f"{statistics.mean(vals):>11.5f}{f(lo):>9}{f(hi):>10}")
            print()

    print("The high-sat column is the one that matters: it is where GMAS's "
          "dither\nmodulation breaks down on spatial overflow, and where ROAST's "
          "overflow\nsuppression is supposed to earn its keep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
