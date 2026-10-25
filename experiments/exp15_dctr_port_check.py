"""Gate for the Python DCTR port: it must reproduce the MATLAB features.

Compares `src/dctr.py` against features the DDE Lab reference code already
extracted for the 300-image experiments (exp08). Nothing computed with the port
may be used until this passes.

    python experiments/exp15_dctr_port_check.py --count 300
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402
from dctr import DIM, extract  # noqa: E402

ARM_A = paths.RESULTS / "exp08_arm_a"
SETS = [  # (label, matlab feature file, image dir)
    ("cover", ARM_A / "cover.f64", paths.DATA / "BOSSbase_QF65_arma"),
    ("GMAS 0.1 stego", ARM_A / "GMAS_p0.1_q71.f64", ARM_A / "GMAS_p0.1_q71" / "stego"),
    ("J-UNIWARD 0.2 stego", ARM_A / "J_UNIWARD_p0.2.f64", ARM_A / "J_UNIWARD_p0.2" / "stego"),
]


def p_error(Xc, Xs, repeats=5, d_sub=800, n_est=200):
    """Identical to exp08_arm_a.p_error: same splits, same ensemble, same seeds."""
    from ensemble import EnsembleFLD
    from metrics import min_p_error
    out = []
    n = min(len(Xc), len(Xs))
    for seed in range(repeats):
        idx = np.random.default_rng(seed).permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        X = np.vstack([Xc[tr], Xs[tr]])
        y = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
        m = EnsembleFLD(d_sub=d_sub, n_estimators=n_est, seed=seed).fit(X, y)
        out.append(min_p_error(m.decision_function(Xc[te]),
                               m.decision_function(Xs[te])))
    return float(np.mean(out)), float(np.std(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--tol", type=float, default=2e-6,
                    help="DCTR.m stores single precision, so ~1e-7 is the floor")
    ap.add_argument("--pe-tol", type=float, default=0.01)
    args = ap.parse_args()

    # The port is NOT bit-identical to MATLAB and cannot be. Where a filter
    # response lands exactly on a rounding boundary (e.g. 14 / 5.6 = 2.5),
    # floating-point noise picks the bin, and the two implementations sum in a
    # different order. So the gate has two parts: (1) the disagreement is
    # confined to a small fraction of entries, and (2) P_E computed from port
    # features matches P_E computed from MATLAB features on the same images.
    #
    # Consequence: NEVER mix the two. Cover and stego features in one
    # comparison must come from the same extractor, or the tie-breaking
    # difference itself becomes a detectable "signal".
    feats, refs, bad = {}, {}, 0
    for label, feat, img_dir in SETS:
        ref = np.fromfile(feat, dtype=np.float64).reshape(-1, DIM)
        imgs = sorted(img_dir.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))
        n = min(args.count, len(imgs), len(ref))
        t0 = time.time()
        got = extract(imgs[:n], qf=65)
        dt = (time.time() - t0) / n
        diff = np.abs(got - ref[:n])
        frac = float((diff > args.tol).mean())
        feats[label], refs[label] = got, ref[:n]
        print(f"{label:<20} {n:>3} images   entries differing: {100 * frac:.3f}%   "
              f"max |diff| {diff.max():.1e}   mean {diff.mean():.1e}   "
              f"({dt:.2f} s/image)")
        if frac > 0.01:
            bad += 1
            print("    FAIL: more than 1% of entries differ - not a tie effect")

    print()
    for label in [s[0] for s in SETS[1:]]:
        pe_m, sd_m = p_error(refs["cover"], refs[label])
        pe_p, sd_p = p_error(feats["cover"], feats[label])
        ok = abs(pe_m - pe_p) <= args.pe_tol
        bad += 0 if ok else 1
        print(f"P_E {label:<20} MATLAB {pe_m:.4f} +/- {sd_m:.4f}   "
              f"port {pe_p:.4f} +/- {sd_p:.4f}   diff {pe_p - pe_m:+.4f}   "
              + ("ok" if ok else "FAIL"))
    # The trap this gate exists to document: mixed extractors.
    pe_x, _ = p_error(refs["cover"], feats["cover"])
    print(f"P_E MATLAB-cover vs port-cover (same images): {pe_x:.4f}  "
          f"<- anything well below chance means the extractors are separable")

    print()
    print("PASS - port features give the same P_E as MATLAB features" if not bad
          else f"FAIL - {bad} check(s) failed; do not use the port")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
