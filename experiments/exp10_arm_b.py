"""M3 Arm B — the same-QF methods, on the channel they were designed for.

SCRIPT and MINICER target recompression at the *same* quality factor as the
cover, not the upward recompression GMAS and ROAST assume (established in M1).
Evaluating them on Arm A's QF65->71 channel would misrepresent them, so they get
their own arm and are reported separately. Merging the two arms into one table
would flatter whichever method the shared channel happened to suit.

Same protocol as Arm A otherwise: security is DCTR + FLD ensemble P_E on the
stego **as transmitted**; robustness is extraction after the channel.

Parameters come from the M1 sweeps, not from the papers (neither publishes the
values used for its results):
  * SCRIPT   R1 = R2 = 10 (integers -- `testRobustGISSttt.m` sweeps `r = 0:R3`),
             R3 = 0, distortion = 0. Only the UERD path is runnable; the
             J-UNIWARD path calls a 3-argument `J_UNIWARD_cost` that ships
             nowhere in the repo.
  * MINICER  stego_step = 6 (the shipped `ones(8,8)` is a demo value; 4-6 is the
             robustness optimum). No ECC by design, so its BER is a RAW channel
             error rate and is not comparable with GMAS/ROAST's post-RS figures.

Channel dependence differs between them, which is itself worth recording:
`preprocessQIM` takes `attack_QF`, so MINICER adapts its embedding to the
channel; `coverSelect`/`costGISS` do not, so SCRIPT does not.

    python experiments/exp10_arm_b.py --count 300 --payloads 0.05 0.1 0.2
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import statistics
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402
from ensemble import EnsembleFLD  # noqa: E402
from metrics import min_p_error  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = paths.BASELINES / "Robust_Steganography"
DIM = 8000


def find_matlab() -> str:
    exe = shutil.which("matlab")
    if exe:
        return exe
    f = sorted(pathlib.Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"))
    if not f:
        raise SystemExit("MATLAB not found")
    return str(f[-1])


def matlab(cmd: str) -> None:
    subprocess.run([find_matlab(), "-batch", cmd], cwd=str(HERE),
                   capture_output=True, text=True)


def load(path: pathlib.Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float64).reshape(-1, DIM)


def p_error(Xc: np.ndarray, Xs: np.ndarray, repeats: int = 5) -> tuple[float, float]:
    out = []
    n = min(len(Xc), len(Xs))
    for seed in range(repeats):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        X = np.vstack([Xc[tr], Xs[tr]])
        y = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
        m = EnsembleFLD(d_sub=800, n_estimators=200, seed=seed).fit(X, y)
        out.append(min_p_error(m.decision_function(Xc[te]),
                               m.decision_function(Xs[te])))
    return float(np.mean(out)), float(np.std(out))


def cache_valid(csv_path: pathlib.Path, expected: int) -> bool:
    """A cached CSV is only reusable if it covers the requested image count.

    Without this, `--count 300` silently reuses a 40-cover CSV from an earlier
    smoke test and reports 40 images' results as 300. Found the hard way.
    """
    if not csv_path.exists():
        return False
    with csv_path.open(newline="", encoding="utf-8") as fh:
        n = sum(1 for _ in csv.DictReader(fh))
    if n < expected:
        print(f"  cache miss: {csv_path.name} has {n} rows, need {expected}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--payloads", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--qf", type=int, default=65, help="cover QF == channel QF")
    ap.add_argument("--script-r", type=int, default=10)
    ap.add_argument("--minicer-step", type=int, default=6)
    args = ap.parse_args()

    out = paths.ensure_results("exp10_arm_b")
    cover_dir = paths.DATA / f"BOSSbase_QF{args.qf}_arma"
    if not cover_dir.exists():
        subprocess.run([sys.executable, str(HERE / "prep_covers.py"),
                        "--qf", str(args.qf), "--count", str(args.count),
                        "--tag", "arma"], check=True)
    covers = sorted(cover_dir.glob("*.jpg"),
                    key=lambda p: (len(p.stem), p.stem))[:args.count]
    pcov = paths.DATA / "BOSSbase_1.01"

    cov_man = out / "covers.txt"
    cov_man.write_text("\n".join(str(p) for p in covers), encoding="utf-8")
    # SCRIPT needs precover|cover pairs
    pair_man = out / "pairs.txt"
    pair_man.write_text("\n".join(f"{pcov / (c.stem + '.pgm')}|{c}" for c in covers),
                        encoding="utf-8")
    print(f"{len(covers)} covers, QF{args.qf} -> QF{args.qf} (same-QF channel)")

    cov_feat = paths.RESULTS / "exp08_arm_a" / "cover.f64"
    if not cov_feat.exists():
        cov_feat = out / "cover.f64"
        matlab(f"dctr_extract('{cov_man.as_posix()}','{cov_feat.as_posix()}',{args.qf})")
    Xc = load(cov_feat)

    rows = []
    for payload in args.payloads:
        for label in ("SCRIPT", "MINICER"):
            slug = f"{label}_p{payload}_q{args.qf}"
            csv_path = out / f"{slug}.csv"
            sdir = out / slug

            if not cache_valid(csv_path, len(covers)):
                print(f"embedding {label} @ {payload}...", flush=True)
                if label == "SCRIPT":
                    call = (f"script_embed_shard('{REPO.as_posix()}',"
                            f"'{pair_man.as_posix()}','{sdir.as_posix()}',"
                            f"'{csv_path.as_posix()}',{payload},{args.qf},{args.qf},"
                            f"0,{args.script_r},{args.script_r},0,0)")
                else:
                    call = (f"minicer_embed_shard('{REPO.as_posix()}',"
                            f"'{cov_man.as_posix()}','{sdir.as_posix()}',"
                            f"'{csv_path.as_posix()}',{payload},{args.qf},{args.qf},"
                            f"0,{args.minicer_step})")
                matlab(call)

            stegos = sorted((sdir / "stego").glob("*.jpg"),
                            key=lambda p: (len(p.stem), p.stem))
            if not stegos:
                print(f"  {label} @ {payload}: no stego output, skipping")
                continue

            feat = out / f"{slug}.f64"
            if not feat.exists():
                man = out / f"{slug}_manifest.txt"
                man.write_text("\n".join(str(p) for p in stegos), encoding="utf-8")
                print(f"  extracting {label} features...", flush=True)
                matlab(f"dctr_extract('{man.as_posix()}','{feat.as_posix()}',{args.qf})")

            Xs = load(feat)
            pe, sd = p_error(Xc[:len(Xs)], Xs)

            with csv_path.open(newline="", encoding="utf-8") as fh:
                recs = list(csv.DictReader(fh))
            bers = [float(r["bit_error_rate"]) for r in recs if r.get("bit_error_rate")]
            ber = statistics.mean(bers) if bers else float("nan")
            errorless = sum(1 for b in bers if b == 0)

            rows.append(dict(method=label, payload=payload, qf=args.qf,
                             p_e=pe, p_e_sd=sd, mean_ber=ber,
                             errorless=errorless, n=len(bers),
                             ecc="none" if label in ("SCRIPT", "MINICER") else "RS(31,15)"))
            print(f"  {label:<9} P_E {pe:.4f} +/- {sd:.4f}   raw BER {ber:.5f}   "
                  f"errorless {errorless}/{len(bers)}")

    res = out / f"arm_b_q{args.qf}.json"
    res.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n{'method':<10}{'payload':>8}{'P_E':>9}{'raw BER':>11}{'errorless':>12}")
    print("-" * 50)
    for r in rows:
        print(f"{r['method']:<10}{r['payload']:>8}{r['p_e']:>9.4f}"
              f"{r['mean_ber']:>11.5f}{r['errorless']:>8}/{r['n']}")
    print("\nBoth carry NO error-correcting code, so these BERs are RAW channel\n"
          "error rates and must not be tabled against GMAS/ROAST's post-RS values.")
    print(f"\nwrote {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
