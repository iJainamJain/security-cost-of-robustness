"""Cross-check the Arm A security results with a second, independent feature set.

DCTR (8000-dim, undecimated-DCT histograms) and CC-PEV (548-dim, DCT-domain
features plus Cartesian calibration) come from different families and respond to
different artifacts. If they agree on the ordering and rough magnitude of P_E
across methods, the Arm A conclusions do not rest on a quirk of one feature set.

Reuses the stego images already produced by exp08_arm_a.py -- no re-embedding.

    python experiments/exp11_ccpev_crosscheck.py --payloads 0.1 0.2
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402
from ensemble import EnsembleFLD  # noqa: E402
from metrics import min_p_error  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ARM_A = paths.RESULTS / "exp08_arm_a"
CCPEV_DIM = 548

METHODS = [("J-UNIWARD", False), ("GMAS", True), ("ROAST-recon", True)]


def find_matlab() -> str:
    exe = shutil.which("matlab")
    if exe:
        return exe
    f = sorted(pathlib.Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"))
    if not f:
        raise SystemExit("MATLAB not found")
    return str(f[-1])


def extract(manifest: pathlib.Path, out: pathlib.Path, qf: int) -> None:
    r = subprocess.run([find_matlab(), "-batch",
                        f"ccpev_extract('{manifest.as_posix()}','{out.as_posix()}',{qf})"],
                       cwd=str(HERE), capture_output=True, text=True)
    # A silent failure here leaves a 0-byte file and crashes far away in the
    # classifier with an unhelpful "0 samples" error. Fail here instead.
    if not out.exists() or out.stat().st_size == 0:
        msg = "CCPEV extraction produced no data for " + manifest.name
        raise SystemExit(msg + "\n" + (r.stdout or "")[-1200:])


def load(p: pathlib.Path) -> np.ndarray:
    return np.fromfile(p, dtype=np.float64).reshape(-1, CCPEV_DIM)


def p_error(Xc, Xs, repeats=5, d_sub=200, n_est=200):
    # d_sub is scaled to the 548-dim feature space; 800 would exceed it.
    out = []
    n = min(len(Xc), len(Xs))
    for seed in range(repeats):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        X = np.vstack([Xc[tr], Xs[tr]])
        y = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
        m = EnsembleFLD(d_sub=d_sub, n_estimators=n_est, seed=seed).fit(X, y)
        out.append(min_p_error(m.decision_function(Xc[te]), m.decision_function(Xs[te])))
    return float(np.mean(out)), float(np.std(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payloads", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--qf", type=int, default=65)
    ap.add_argument("--attack-qf", type=int, default=71)
    args = ap.parse_args()

    out = paths.ensure_results("exp11_ccpev")

    cov_feat = out / "cover_ccpev.f64"
    if not cov_feat.exists():
        print("extracting cover CCPEV features...", flush=True)
        extract(ARM_A / "covers.txt", cov_feat, args.qf)
    Xc = load(cov_feat)
    print(f"cover features {Xc.shape}")

    rows = []
    for payload in args.payloads:
        for label, chan_dep in METHODS:
            slug = (f"{label.replace('-', '_')}_p{payload}"
                    + (f"_q{args.attack_qf}" if chan_dep else ""))
            man = ARM_A / f"{slug}_manifest.txt"
            if not man.exists():
                print(f"  {label} @ {payload}: no manifest, skipping")
                continue
            feat = out / f"{slug}_ccpev.f64"
            if not feat.exists():
                print(f"extracting {label} @ {payload}...", flush=True)
                extract(man, feat, args.qf)
            Xs = load(feat)
            pe, sd = p_error(Xc[:len(Xs)], Xs)
            rows.append(dict(method=label, payload=payload, features="CCPEV",
                             p_e=pe, p_e_sd=sd, n=len(Xs)))
            print(f"  {label:<12} @ {payload}  P_E {pe:.4f} +/- {sd:.4f}")

    (out / "ccpev_crosscheck.json").write_text(json.dumps(rows, indent=2),
                                               encoding="utf-8")

    # side-by-side with DCTR
    dctr = json.loads((ARM_A / "arm_a_q71.json").read_text())
    g = lambda m, p: next((r["p_e"] for r in dctr
                           if r["method"] == m and abs(r["payload"] - p) < 1e-9), None)
    print(f"\n{'payload':>8}  {'method':<13}{'DCTR (8000d)':>14}{'CCPEV (548d)':>14}")
    print("-" * 51)
    for r in rows:
        d = g(r["method"], r["payload"])
        print(f"{r['payload']:>8}  {r['method']:<13}"
              f"{d if d is None else f'{d:.4f}':>14}{r['p_e']:>14.4f}")
    print("\nAgreement on ordering and magnitude means the Arm A conclusions are\n"
          "not an artifact of the DCTR feature set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
