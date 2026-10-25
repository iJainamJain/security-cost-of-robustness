"""Paper 2, first test: does spreading a payload over several covers help?

The redundancy-relocation idea needs one thing to be true before anything is
worth building: at equal TOTAL payload, a sender who spreads it over several
covers must leave less pooled evidence than one who concentrates it. That holds
if per-image detectability grows faster than linearly in payload. It fails if
robust embedding has a large fixed footprint, because spreading then multiplies
that footprint.

No new embedding and no new detector: this re-uses per-image test scores from
detectors we already trained, one per (method, payload) cell.

Setup. A sender posts a batch of B images. Strategy (k, p) puts payload p in k
of them and leaves B-k clean; total payload k*p is held fixed across the
strategies compared. The warden knows the strategy (Kerckhoffs), so scores every
image with the detector trained for payload p, pools the B scores, and decides
"steganographer or not". We report batch-level P_E under several pooling rules
and take the warden's best. The sender wants that number HIGH.

    python experiments/exp17_batch_pooling.py
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from math import comb

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

SOURCES = {
    "DCTR+FLD, 2,000 images": paths.RESULTS / "m4_scale_results/scale_results/dctr/scores",
    "CNN 25 ep, 2,000 images": paths.RESULTS / "m4_scale_cnn/scale_results/cnn/scores",
}
# Per-image probability of error-free extraction, measured on the same 2,000
# images (exp13 CSVs). J-UNIWARD is not robust: it never survives the channel,
# so it has no entry and appears here only as the classical reference.
ERRORLESS = {
    ("GMAS", 0.05): 1392 / 2000, ("GMAS", 0.1): 1488 / 2000, ("GMAS", 0.2): 1495 / 2000,
    ("ROAST", 0.05): 1996 / 2000, ("ROAST", 0.1): 1994 / 2000, ("ROAST", 0.2): 1987 / 2000,
    ("SCRIPT", 0.05): 1997 / 1998, ("SCRIPT", 0.1): 1995 / 1996, ("SCRIPT", 0.2): 1988 / 1990,
    ("MINICER", 0.05): 1130 / 2000, ("MINICER", 0.1): 780 / 2000, ("MINICER", 0.2): 338 / 2000,
}


def load(path: pathlib.Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """seed -> (cover scores, stego scores), aligned by image."""
    by: dict[int, dict[str, dict[str, float]]] = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            by.setdefault(int(r["seed"]), {}).setdefault(r["label"], {})[r["image"]] = float(r["score"])
    out = {}
    for seed, d in by.items():
        imgs = sorted(set(d["cover"]) & set(d["stego"]))
        out[seed] = (np.array([d["cover"][i] for i in imgs]),
                     np.array([d["stego"][i] for i in imgs]))
    return out


def p_error(sc: np.ndarray, ss: np.ndarray) -> float:
    """min over thresholds of (P_FA + P_MD)/2, by sorting (handles ties)."""
    v = np.concatenate([sc, ss])
    lab = np.concatenate([np.zeros(len(sc)), np.ones(len(ss))])
    order = np.argsort(v, kind="mergesort")
    v, lab = v[order], lab[order]
    # threshold t = each distinct value: cover >= t is a false alarm, stego < t a miss
    first = np.r_[True, v[1:] != v[:-1]]
    below_stego = np.r_[0, np.cumsum(lab)][:-1][first]          # stego strictly below t
    below_cover = np.r_[0, np.cumsum(1 - lab)][:-1][first]
    pe = 0.5 * ((len(sc) - below_cover) / len(sc) + below_stego / len(ss))
    return float(min(pe.min(), 0.5))


def batches(c, s, B, k, n, rng):
    """n cover batches and n stego batches (k stego + B-k clean), distinct images."""
    m = len(c)
    idx = np.argsort(rng.random((n, m)), axis=1)[:, :B]          # B distinct per row
    cov = c[idx]
    idx2 = np.argsort(rng.random((n, m)), axis=1)[:, :B]
    stg = np.concatenate([s[idx2[:, :k]], c[idx2[:, k:]]], axis=1)
    return cov, stg


def pooled(cov, stg, k):
    rules = {
        "mean": lambda x: x.mean(1),
        "max": lambda x: x.max(1),
        # mean of the k most suspicious: the natural rule for a warden who
        # knows k images carry payload
        "top-k": lambda x: np.sort(x, 1)[:, -k:].mean(1),
    }
    return {name: p_error(f(cov), f(stg)) for name, f in rules.items()}


def covers_needed(p_ok: float, k: int, target: float) -> int:
    """Smallest N with P(at least k of N succeed) >= target (an any-k-of-N
    erasure code across covers)."""
    for N in range(k, 200):
        ok = sum(comb(N, j) * p_ok ** j * (1 - p_ok) ** (N - j) for j in range(k, N + 1))
        if ok >= target:
            return N
    return 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="batches per class per seed")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out_rows = []

    for src_name, src in SOURCES.items():
        if not src.is_dir():
            continue
        print("\n" + "=" * 78)
        print(src_name)
        print("=" * 78)
        for method in ("J-UNIWARD", "GMAS", "ROAST", "SCRIPT", "MINICER"):
            cells = {p: load(src / f"{method}_{p}.csv") for p in (0.05, 0.1, 0.2)
                     if (src / f"{method}_{p}.csv").exists()}
            if len(cells) < 3:
                continue
            single = {p: np.mean([p_error(*cells[p][sd]) for sd in cells[p]]) for p in cells}
            print(f"\n{method}   single-image P_E: " +
                  ", ".join(f"{p}: {single[p]:.4f}" for p in sorted(single)))

            # ---- equal total payload (0.2), batch of 4, no outer code --------
            print("  equal total payload 0.2 in a batch of 4 images")
            print(f"  {'strategy':<22}{'mean':>8}{'max':>8}{'top-k':>8}{'warden best':>13}")
            best = {}
            for k, p in ((1, 0.2), (2, 0.1), (4, 0.05)):
                res = [pooled(*batches(c, s, 4, k, args.n, rng), k)
                       for c, s in cells[p].values()]
                avg = {r: float(np.mean([x[r] for x in res])) for r in res[0]}
                best[(k, p)] = min(avg.values())
                print(f"  {f'{k} x {p}  + {4 - k} clean':<22}{avg['mean']:>8.4f}"
                      f"{avg['max']:>8.4f}{avg['top-k']:>8.4f}{best[(k, p)]:>13.4f}")
                out_rows.append(dict(source=src_name, method=method, batch=4, k=k,
                                     payload=p, matched_robustness=False, **avg,
                                     warden_best=best[(k, p)]))
            conc, spread = best[(1, 0.2)], best[(4, 0.05)]
            verdict = ("SPREADING HELPS" if spread > conc + 0.01 else
                       "SPREADING HURTS" if spread < conc - 0.01 else "no real difference")
            print(f"  -> concentrated {conc:.4f} vs spread {spread:.4f}: {verdict}")

            # ---- matched robustness: any-k-of-N across covers ----------------
            if (method, 0.2) in ERRORLESS:
                target = ERRORLESS[(method, 0.2)]         # what one cover at 0.2 achieves
                rows = []
                for k, p in ((2, 0.1), (4, 0.05)):
                    N = covers_needed(ERRORLESS[(method, p)], k, target)
                    rows.append((k, p, N))
                B = max(8, max(N for _, _, N in rows))
                print(f"  matched robustness (delivery >= {target:.3f}), batch of {B}")
                res = [pooled(*batches(c, s, B, 1, args.n, rng), 1) for c, s in cells[0.2].values()]
                ref = min(float(np.mean([x[r] for x in res])) for r in res[0])
                print(f"    1 x 0.2 + {B - 1} clean".ljust(40) + f"warden best {ref:.4f}")
                out_rows.append(dict(source=src_name, method=method, batch=B, k=1, payload=0.2,
                                     matched_robustness=True, warden_best=ref))
                for k, p, N in rows:
                    res = [pooled(*batches(c, s, B, N, args.n, rng), N) for c, s in cells[p].values()]
                    b = min(float(np.mean([x[r] for x in res])) for r in res[0])
                    print(f"    {N} x {p} (any {k} of {N}) + {B - N} clean".ljust(40)
                          + f"warden best {b:.4f}   "
                          + ("better for the sender" if b > ref + 0.01 else
                             "worse for the sender" if b < ref - 0.01 else "no real difference"))
                    out_rows.append(dict(source=src_name, method=method, batch=B, k=N,
                                         needed=k, payload=p, matched_robustness=True,
                                         warden_best=b))

    out = paths.ensure_results("exp17_batch_pooling")
    (out / "batch_pooling.json").write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'batch_pooling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
