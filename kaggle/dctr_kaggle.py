"""DCTR + FLD-ensemble steganalysis over a stego-eval dataset.

Part A of the scale notebook: the feature-based detector, run where MATLAB is
not available. Uses the Python DCTR port (dctr.py), which is validated against
the DDE Lab MATLAB reference by experiments/exp15_dctr_port_check.py.

Cover and stego features are ALWAYS extracted by the same code here. Never
combine these with MATLAB-extracted features in one comparison: the two break
rounding ties differently, and that difference alone is detectable.

    python dctr_kaggle.py --data /kaggle/input/.../dataset --out /kaggle/working/dctr
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

import numpy as np
import torch

from dctr import DCTR, DIM, read_jpeg
from ensemble import EnsembleFLD


def numeric(stems):
    return sorted(stems, key=lambda s: (len(s), s))


def p_error(sc: np.ndarray, ss: np.ndarray) -> float:
    """min over thresholds of 0.5*(P_FA + P_MD); higher score = more stego."""
    best = 1.0
    for t in np.unique(np.concatenate([sc, ss])):
        best = min(best, 0.5 * ((sc >= t).mean() + (ss < t).mean()))
    return float(best)


def md_at_fa(sc, ss, target_fa=0.01) -> float:
    return float((ss < np.quantile(sc, 1 - target_fa)).mean())


def extract_dir(ex: DCTR, folder: pathlib.Path, stems: list[str],
                cache: pathlib.Path) -> np.ndarray:
    """Features for folder/<stem>.jpg in the given order, cached as float32.

    Reads are deliberately sequential. jpeglib is NOT thread-safe: reading
    from a thread pool kills the interpreter outright, with no traceback and
    no Python exception to catch. Measured, not assumed - do not parallelise
    this with threads.
    """
    meta = cache.with_suffix(".stems.json")
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == stems:
        return np.load(cache).astype(np.float64)
    out = np.empty((len(stems), DIM), dtype=np.float32)
    t0 = time.time()
    for i, s in enumerate(stems):
        out[i] = ex.features(*read_jpeg(folder / f"{s}.jpg"))
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(stems)}  "
                  f"({(time.time() - t0) / (i + 1) * 1000:.0f} ms/image)", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    meta.write_text(json.dumps(stems))
    return out.astype(np.float64)


def run_cell(Xc, Xs, stems, seeds, d_sub, n_est):
    """Paired 50/50 split per seed. Returns (summary dict, score rows)."""
    pes, revs, mds, rows = [], [], [], []
    n = len(stems)
    for seed in range(seeds):
        idx = np.random.default_rng(seed).permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        X = np.vstack([Xc[tr], Xs[tr]])
        y = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
        m = EnsembleFLD(d_sub=d_sub, n_estimators=n_est, seed=seed).fit(X, y)
        sc, ss = m.decision_function(Xc[te]), m.decision_function(Xs[te])
        pes.append(p_error(sc, ss))
        revs.append(p_error(ss, sc))
        mds.append(md_at_fa(sc, ss))
        rows += [(seed, stems[i], "cover", float(v)) for i, v in zip(te, sc)]
        rows += [(seed, stems[i], "stego", float(v)) for i, v in zip(te, ss)]
        print(f"    seed {seed}: P_E {pes[-1]:.4f}", flush=True)
    return dict(p_e_mean=float(np.mean(pes)), p_e_std=float(np.std(pes)),
                p_e_per_seed=[float(v) for v in pes],
                p_e_reversed_mean=float(np.mean(revs)),
                md_at_fa1_mean=float(np.mean(mds)),
                md_at_fa1_std=float(np.std(mds)),
                n_train_pairs=n // 2, n_test_pairs=n - n // 2,
                seeds=list(range(seeds))), rows


def write_scores(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "image", "label", "score"])
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--d-sub", type=int, default=800)
    ap.add_argument("--n-est", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="first N covers (testing)")
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--null-only", action="store_true",
                    help="cover features and the null test, no cells")
    args = ap.parse_args()

    data, out = pathlib.Path(args.data), pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mf = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    qf = int(mf.get("image_spec", {}).get("cover_quality_factor", 65))
    cover_dir = data / mf["cover_dir"]
    cov_stems = numeric(p.stem for p in cover_dir.glob("*.jpg"))
    if args.limit:
        cov_stems = cov_stems[:args.limit]
    print(f"device {args.device} | QF {qf} | {len(cov_stems)} covers", flush=True)

    ex = DCTR(qf, args.device)
    feat = out / "features"
    print("extracting cover features...", flush=True)
    Xcov = extract_dir(ex, cover_dir, cov_stems, feat / "cover.npy")
    pos = {s: i for i, s in enumerate(cov_stems)}

    # Null test: covers against OTHER covers. With ensemble vote counts the
    # scores have many ties, so chance is not the continuous-score value; this
    # measured number is the chance level for every DCTR result below.
    null_path = out / "null_test.json"
    if not null_path.exists():
        # Random groups, never "first half vs second half": BOSSbase is
        # ordered by source, and splitting by file number makes the two groups
        # genuinely different (measured: P_E 0.31 that way, 0.48 at random).
        perm = np.random.default_rng(20261002).permutation(len(cov_stems))
        half = len(cov_stems) // 2
        ia, ib = perm[:half], perm[half:2 * half]
        print("\nNULL TEST (cover vs cover, random groups)", flush=True)
        r, rows = run_cell(Xcov[ia], Xcov[ib], [cov_stems[i] for i in ia],
                           args.seeds, args.d_sub, args.n_est)
        r["grouping"] = "random"
        write_scores(out / "scores" / "NULL_TEST.csv", rows)
        null_path.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"  NULL TEST P_E = {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f} "
              f"at {r['n_test_pairs']} test pairs")
    null = json.loads(null_path.read_text())
    if args.null_only:
        return 0

    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["cell"] for r in results}
    cells = [c for c in mf["cells"] if not args.cells or c["cell"] in args.cells]

    for c in cells:
        if c["cell"] in done:
            print(f"skip {c['cell']} (already in results.json)")
            continue
        sdir = data / c["stego_dir"]
        stems = [s for s in numeric(p.stem for p in sdir.glob("*.jpg")) if s in pos]
        print(f"\n{c['cell']}  ({len(stems)} pairs)", flush=True)
        t0 = time.time()
        try:
            Xs = extract_dir(ex, sdir, stems, feat / f"{c['cell']}.npy")
            Xc = Xcov[[pos[s] for s in stems]]
            r, rows = run_cell(Xc, Xs, stems, args.seeds, args.d_sub, args.n_est)
            write_scores(out / "scores" / f"{c['cell']}.csv", rows)
            entry = dict(cell=c["cell"], method=c["method"],
                         payload_bpnzac=c["payload_bpnzac"], arm=c["arm"],
                         detector="DCTR+FLD", status="ok",
                         p_e_chance_level=null["p_e_mean"],
                         minutes=round((time.time() - t0) / 60, 1), **r)
            print(f"  -> P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}   "
                  f"(null {null['p_e_mean']:.4f})")
        except Exception as exc:  # noqa: BLE001 - record, never substitute
            entry = dict(cell=c["cell"], method=c["method"],
                         payload_bpnzac=c["payload_bpnzac"], arm=c["arm"],
                         detector="DCTR+FLD", status="failed", error=str(exc)[:300])
            print(f"  FAILED: {exc}")
        results.append(entry)
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nwrote {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
