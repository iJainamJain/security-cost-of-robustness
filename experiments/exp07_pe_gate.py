"""M2 gate: does our P_E pipeline produce trustworthy security numbers?

Runs three checks, in order of how much they matter:

1. **Null test** — train cover-vs-cover (a random split of the cover set against
   itself). P_E must come out at ~0.50. This is the check that catches the real
   failure modes: label leakage, feature/label misalignment, train/test
   contamination. A pipeline that "detects" a difference between covers and
   other covers is broken, and every downstream number would be fiction.
2. **Monotonicity** — P_E must fall as payload rises.
3. **Band** — P_E at 0.4 bpnzac should land near published values.

Reference (Xia, Guan & Zhao, arXiv:1706.07576, Table 4), J-UNIWARD at QF 75,
BOSSbase 10k, FLD ensemble, ten 5000/5000 splits: GFR (17000D) gives 0.1028 at
0.4 bpnzac. DCTR is 8000D and a weaker detector on J-UNIWARD, so ours should sit
*above* that. We also train on far fewer images, which weakens the detector
further and pushes P_E up again — so the target is a band and an ordering, not a
tolerance.

    python experiments/exp07_pe_gate.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402
from ensemble import EnsembleFLD  # noqa: E402
from metrics import min_p_error  # noqa: E402

DIM = 8000

# Xia, Guan & Zhao, arXiv:1706.07576, Table 4 -- J-UNIWARD at QF 75, GFR
# (17000D) with the FLD ensemble, BOSSbase 10k, averaged over ten 5000/5000
# splits. Our DCTR numbers must sit above these: DCTR is 8000D and a weaker
# detector on J-UNIWARD, and we train on far fewer images.
GFR_QF75 = {0.05: 0.4638, 0.1: 0.4089, 0.2: 0.2866, 0.3: 0.1786, 0.4: 0.1028}


def load_features(path: pathlib.Path) -> np.ndarray:
    a = np.fromfile(path, dtype=np.float64)
    if a.size % DIM:
        raise SystemExit(f"{path}: {a.size} values is not a multiple of {DIM}")
    return a.reshape(-1, DIM)


def evaluate(Xc: np.ndarray, Xs: np.ndarray, seed: int,
             d_sub: int, n_est: int, train_frac: float = 0.5) -> float:
    """P_E from a paired split — cover i and stego i always land on the same side.

    Pairing matters: the two sets are the *same images* with and without a
    payload, so splitting them independently would put a cover in training and
    its own stego in testing, leaking the image content and flattering P_E.
    """
    n = min(len(Xc), len(Xs))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(n * train_frac)
    tr, te = idx[:n_tr], idx[n_tr:]

    X_tr = np.vstack([Xc[tr], Xs[tr]])
    y_tr = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
    model = EnsembleFLD(d_sub=d_sub, n_estimators=n_est, seed=seed).fit(X_tr, y_tr)

    return min_p_error(model.decision_function(Xc[te]),
                       model.decision_function(Xs[te]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", type=pathlib.Path,
                    default=paths.RESULTS / "m2_cover.f64")
    ap.add_argument("--stego", type=pathlib.Path,
                    default=paths.RESULTS / "m2_stego.f64")
    ap.add_argument("--payload", type=float, default=0.4,
                    help="bpnzac of the stego file, for labelling and for "
                         "picking the right published reference")
    ap.add_argument("--d-sub", type=int, default=800)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    Xc = load_features(args.cover)
    Xs = load_features(args.stego)
    print(f"cover features {Xc.shape}   stego features {Xs.shape}")
    print(f"ensemble: d_sub={args.d_sub}, L={args.n_estimators}, "
          f"{args.repeats} random splits\n")

    # --- Check 1: null test -------------------------------------------------
    # Split the cover set in half and label one half "stego". There is no real
    # difference, so any detector worth trusting scores ~0.5.
    half = len(Xc) // 2
    null = [evaluate(Xc[:half], Xc[half:2 * half], s, args.d_sub, args.n_estimators)
            for s in range(args.repeats)]
    null_mean = float(np.mean(null))
    print(f"[1] NULL TEST  cover vs cover   P_E = {null_mean:.4f} "
          f"(+/- {np.std(null):.4f})   expect ~0.50")
    null_ok = abs(null_mean - 0.5) < 0.06
    print(f"    {'PASS' if null_ok else 'FAIL'} — "
          f"{'no spurious signal' if null_ok else 'PIPELINE IS LEAKING; stop here'}")

    # --- Check 3: band ------------------------------------------------------
    real = [evaluate(Xc, Xs, s, args.d_sub, args.n_estimators)
            for s in range(args.repeats)]
    pe = float(np.mean(real))
    print(f"\n[3] J-UNIWARD {args.payload} bpnzac, QF75   P_E = {pe:.4f} "
          f"(+/- {np.std(real):.4f})")

    ref = GFR_QF75.get(args.payload)
    if ref is None:
        print("    no published GFR value tabulated for this payload")
        band_ok = True
    else:
        print(f"    published GFR (17000D, 10k images): {ref:.4f}")
        print(f"    ours is DCTR (8000D, {len(Xc)} images), a weaker detector "
              f"trained on less\n    data, so it must sit ABOVE the reference "
              f"but below chance")
        band_ok = ref < pe < 0.50
        print(f"    {'PASS' if band_ok else 'CHECK'} — "
              f"{'above reference, below chance' if band_ok else 'outside expected range'}")

    print("\n[2] monotonicity across payloads: run this script against feature "
          "files\n    for other payloads once they exist (M3 produces them).")

    print("\n" + ("GATE PASSED" if (null_ok and band_ok)
                  else "GATE NOT PASSED — do not proceed to M3"))
    return 0 if (null_ok and band_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
