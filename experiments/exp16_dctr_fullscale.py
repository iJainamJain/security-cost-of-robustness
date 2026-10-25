"""DCTR + FLD ensemble over the full-scale stego set (exp13), on this laptop.

Gives the paper its headline tables at 5,000 and 10,000 images, and the
matching baseline for the CNN runs. No MATLAB and no Kaggle quota: features
come from the Python DCTR port (src/dctr.py), which exp15 shows reproduces the
reference extractor's P_E.

Two stages, both resumable:

  extract  Features in 500-image chunks, several worker processes. A chunk is
           written atomically, so a kill costs at most the chunk in flight.
           Order is tiered: every set reaches 5,000 images before any goes on.
  score    Per tier, per cell: five paired 50/50 splits, FLD ensemble
           (d_sub 800, L 200), plus a cover-vs-cover null with random groups.

Cover and stego features always come from this one extractor. Never mix them
with MATLAB-extracted features (exp15: that alone is detectable).

    python experiments/exp16_dctr_fullscale.py            # everything
    python experiments/exp16_dctr_fullscale.py --status
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exp13_fullscale import COVER_QF, cells as grid_cells, row_ok  # noqa: E402

SRC = paths.RESULTS / "exp13_fullscale"
COVERS = paths.DATA / f"BOSSbase_QF{COVER_QF}_full"
DIM = 8000
META = {"juniward": ("A", "J-UNIWARD"), "gmas": ("A", "GMAS"),
        "roast": ("A", "ROAST-recon"), "script": ("B", "SCRIPT"),
        "minicer": ("B", "MINICER")}


# ------------------------------------------------------------- what exists ---
def ok_stems(cell_name: str) -> set[str]:
    """Stems whose embed succeeded, per the embedder's own CSV."""
    good = set()
    for f in sorted((SRC / cell_name / "csv").glob("chunk_*.csv")):
        with f.open(newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                s = (r.get("cover") or "").strip()
                if s.isdigit() and row_ok(r):
                    good.add(s)
    return good


def set_dir(name: str) -> pathlib.Path:
    return COVERS if name == "cover" else SRC / name / "stego"


def jobs(out: pathlib.Path, tiers: list[int], chunk: int) -> list[tuple]:
    """(tier index, chunk index, set name), tier-major so tiers finish in order."""
    names = ["cover"] + [c["name"] for c in grid_cells()]
    n_chunks = (tiers[-1] + chunk - 1) // chunk
    out_jobs = []
    for k in range(n_chunks):
        end = min((k + 1) * chunk, tiers[-1])
        tier = next(i for i, t in enumerate(tiers) if end <= t)
        for n in names:
            out_jobs.append((tier, k, n))
    out_jobs.sort(key=lambda j: (j[0], j[1]))
    return out_jobs


def chunk_path(out: pathlib.Path, name: str, k: int) -> pathlib.Path:
    return out / "features" / name / f"chunk_{k:04d}.npz"


# ----------------------------------------------------------------- extract ---
def worker(out: pathlib.Path, tiers: list[int], chunk: int, k_worker: int,
           n_workers: int, threads: int) -> int:
    import torch

    from dctr import DCTR, read_jpeg

    torch.set_num_threads(threads)
    ex = DCTR(COVER_QF, "cpu")
    ok_cache: dict[str, set[str]] = {}
    mine = [j for i, j in enumerate(jobs(out, tiers, chunk)) if i % n_workers == k_worker]
    for _, k, name in mine:
        dest = chunk_path(out, name, k)
        if dest.exists():
            continue
        stems = [str(i) for i in range(k * chunk + 1, min((k + 1) * chunk, tiers[-1]) + 1)]
        if name != "cover":
            if name not in ok_cache:
                ok_cache[name] = ok_stems(name)
            stems = [s for s in stems if s in ok_cache[name]]
        d = set_dir(name)
        stems = [s for s in stems if (d / f"{s}.jpg").exists()
                 and (d / f"{s}.jpg").stat().st_size > 0]
        feats = np.empty((len(stems), DIM), dtype=np.float32)
        t0 = time.time()
        for i, s in enumerate(stems):
            feats[i] = ex.features(*read_jpeg(d / f"{s}.jpg"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / (dest.stem + f".tmp{k_worker}.npz")
        np.savez(tmp, feats=feats, stems=np.array(stems))
        os.replace(tmp, dest)
        print(f"w{k_worker} {name}#{k:04d} {len(stems)} images "
              f"{(time.time() - t0) / max(len(stems), 1):.2f} s/img", flush=True)
    return 0


def tier_extracted(out, tiers, chunk, tier_idx) -> tuple[int, int]:
    js = [j for j in jobs(out, tiers, chunk) if j[0] <= tier_idx]
    return sum(chunk_path(out, n, k).exists() for _, k, n in js), len(js)


def load_set(out, name, n_images, chunk) -> tuple[np.ndarray, list[str]]:
    feats, stems = [], []
    for k in range((n_images + chunk - 1) // chunk):
        z = np.load(chunk_path(out, name, k))
        feats.append(z["feats"]); stems += [str(s) for s in z["stems"]]
    return np.concatenate(feats), stems


# ------------------------------------------------------------------- score ---
def p_error(sc: np.ndarray, ss: np.ndarray) -> float:
    best = 1.0
    for t in np.unique(np.concatenate([sc, ss])):
        best = min(best, 0.5 * ((sc >= t).mean() + (ss < t).mean()))
    return float(best)


def md_at_fa(sc, ss, target_fa=0.01) -> float:
    return float((ss < np.quantile(sc, 1 - target_fa)).mean())


class LeanFLD:
    """EnsembleFLD (src/ensemble.py), restated to index before converting.

    Same RNG draws, same learners, same votes - it differs only in taking the
    800 columns of a bag out of the float32 matrix and converting THAT to
    float64, instead of converting the whole 8000-dim matrix first. At 10,000
    images the reference path needs several GB this laptop does not have free.
    float32 -> float64 is exact, so the numbers are identical; `--selfcheck`
    verifies that against the reference class.
    """

    def __init__(self, d_sub=800, n_estimators=200, seed=0):
        self.d_sub, self.n_estimators, self.seed = d_sub, n_estimators, seed

    def fit(self, X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        y = np.asarray(y).ravel()
        n, n_features = X.shape
        d = min(self.d_sub, n_features)
        rng = np.random.default_rng(self.seed)
        self._sub, self._clf = [], []
        for _ in range(self.n_estimators):
            cols = rng.choice(n_features, d, replace=False)
            bag = rng.choice(n, n, replace=True)
            clf = LDA(solver="lsqr", shrinkage="auto")
            clf.fit(X[np.ix_(bag, cols)].astype(np.float64), y[bag])
            self._sub.append(cols); self._clf.append(clf)
        return self

    def decision_function(self, X):
        votes = np.zeros(X.shape[0], dtype=np.float64)
        for cols, clf in zip(self._sub, self._clf):
            votes += clf.predict(X[:, cols].astype(np.float64)).astype(np.float64)
        return votes / len(self._clf)


def run_cell(Xc, Xs, stems, seeds, d_sub, n_est, note):
    pes, revs, mds, rows = [], [], [], []
    n = len(stems)
    for seed in range(seeds):
        t0 = time.time()
        idx = np.random.default_rng(seed).permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        X = np.concatenate([Xc[tr], Xs[tr]])
        y = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
        m = LeanFLD(d_sub, n_est, seed).fit(X, y)
        del X
        sc, ss = m.decision_function(Xc[te]), m.decision_function(Xs[te])
        pes.append(p_error(sc, ss)); revs.append(p_error(ss, sc))
        mds.append(md_at_fa(sc, ss))
        rows += [(seed, stems[i], "cover", float(v)) for i, v in zip(te, sc)]
        rows += [(seed, stems[i], "stego", float(v)) for i, v in zip(te, ss)]
        note(f"    seed {seed}: P_E {pes[-1]:.4f}  ({time.time() - t0:.0f} s)")
    return dict(p_e_mean=float(np.mean(pes)), p_e_std=float(np.std(pes)),
                p_e_per_seed=[float(v) for v in pes],
                p_e_reversed_mean=float(np.mean(revs)),
                md_at_fa1_mean=float(np.mean(mds)), md_at_fa1_std=float(np.std(mds)),
                n_train_pairs=n // 2, n_test_pairs=n - n // 2,
                seeds=list(range(seeds))), rows


def write_scores(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["seed", "image", "label", "score"])
        w.writerows(rows)


def score_tier(out, n_images, chunk, seeds, d_sub, n_est, note) -> None:
    tdir = out / f"tier_{n_images}"
    tdir.mkdir(parents=True, exist_ok=True)
    Xcov, cov_stems = load_set(out, "cover", n_images, chunk)
    pos = {s: i for i, s in enumerate(cov_stems)}
    note(f"TIER {n_images}: {len(cov_stems)} covers loaded")

    null_path = tdir / "null_test.json"
    if not null_path.exists():
        # Random groups: BOSSbase is ordered by source, so groups formed by
        # file number differ for reasons unrelated to the pipeline.
        perm = np.random.default_rng(20261002).permutation(len(cov_stems))
        half = len(cov_stems) // 2
        ia, ib = np.sort(perm[:half]), np.sort(perm[half:2 * half])
        note("  NULL TEST (cover vs cover, random groups)")
        r, rows = run_cell(Xcov[ia], Xcov[ib], [cov_stems[i] for i in ia],
                           seeds, d_sub, n_est, note)
        r["grouping"] = "random"
        write_scores(tdir / "scores" / "NULL_TEST.csv", rows)
        null_path.write_text(json.dumps(r, indent=2), encoding="utf-8")
        note(f"  NULL P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f} "
             f"at {r['n_test_pairs']} test pairs")
    null = json.loads(null_path.read_text())

    res_path = tdir / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["cell"] for r in results}
    for c in grid_cells():
        if c["name"] in done:
            continue
        t0 = time.time()
        Xs, stems = load_set(out, c["name"], n_images, chunk)
        keep = [i for i, s in enumerate(stems) if s in pos]
        stems = [stems[i] for i in keep]
        Xs = Xs[keep]
        Xc = Xcov[[pos[s] for s in stems]]
        note(f"  {c['name']} ({len(stems)} pairs)")
        r, rows = run_cell(Xc, Xs, stems, seeds, d_sub, n_est, note)
        write_scores(tdir / "scores" / f"{c['name']}.csv", rows)
        arm, method = META[c["kind"]]
        results.append(dict(cell=c["name"], method=method, arm=arm,
                            payload_bpnzac=c["payload"], detector="DCTR+FLD",
                            status="ok", n_images=n_images,
                            null_p_e=null["p_e_mean"],
                            minutes=round((time.time() - t0) / 60, 1), **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        note(f"  -> {c['name']} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f} "
             f"(null {null['p_e_mean']:.4f})")
        del Xs, Xc
    note(f"TIER {n_images} done")


def selfcheck() -> int:
    from ensemble import EnsembleFLD
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 2000)).astype(np.float32)
    y = np.r_[np.zeros(150), np.ones(150)]
    X[150:, :40] += 0.25
    a = EnsembleFLD(d_sub=200, n_estimators=15, seed=3).fit(X, y).decision_function(X)
    b = LeanFLD(200, 15, 3).fit(X, y).decision_function(X)
    print("LeanFLD vs EnsembleFLD max |diff|:", float(np.abs(a - b).max()))
    return 0 if np.array_equal(a, b) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--tiers", type=int, nargs="+", default=[5000, 10000])
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--d-sub", type=int, default=800)
    ap.add_argument("--n-est", type=int, default=200)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    out = pathlib.Path(args.out) if args.out else paths.ensure_results("exp16_dctr_fullscale")
    out.mkdir(parents=True, exist_ok=True)
    tiers = sorted(args.tiers)

    if args.worker is not None:
        return worker(out, tiers, args.chunk, args.worker, args.workers, args.threads)

    if args.status:
        for i, t in enumerate(tiers):
            d, n = tier_extracted(out, tiers, args.chunk, i)
            rp = out / f"tier_{t}" / "results.json"
            k = len(json.loads(rp.read_text())) if rp.exists() else 0
            print(f"tier {t:>6}: features {d}/{n} chunks ({100 * d / n:.1f}%) | "
                  f"cells scored {k}/{len(grid_cells())}")
        return 0

    runlog = (out / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    note(f"START tiers {tiers}, {args.workers} workers x {args.threads} threads")
    base = [sys.executable, "-u", str(pathlib.Path(__file__).resolve()),
            "--out", str(out), "--tiers", *map(str, tiers), "--chunk", str(args.chunk),
            "--workers", str(args.workers), "--threads", str(args.threads)]
    def launch(k: int) -> subprocess.Popen:
        log = (out / f"worker{k}.log").open("a")
        return subprocess.Popen(base + ["--worker", str(k)], stdout=log,
                                stderr=subprocess.STDOUT)

    procs = [launch(k) for k in range(args.workers)]
    restarts = [0] * args.workers
    scored = 0
    last = time.time()
    while scored < len(tiers):
        # A worker owns a fixed share of the chunks, so one that dies would
        # leave its share undone for ever. Bring it back (native crashes under
        # memory pressure were seen in testing); give up after repeated deaths.
        for k, p in enumerate(procs):
            rc = p.poll()
            if rc not in (None, 0) and restarts[k] < 8:
                restarts[k] += 1
                note(f"worker {k} died (exit {rc}); restart {restarts[k]}/8")
                time.sleep(10)
                procs[k] = launch(k)
        d, n = tier_extracted(out, tiers, args.chunk, scored)
        if d == n:
            score_tier(out, tiers[scored], args.chunk, args.seeds, args.d_sub,
                       args.n_est, note)
            scored += 1
            continue
        if all(p.poll() is not None for p in procs):
            note(f"workers exited with tier {tiers[scored]} incomplete ({d}/{n}) - "
                 f"see worker*.log; rerun to resume")
            return 1
        if time.time() - last > 900:
            note(f"extracting: tier {tiers[scored]} {d}/{n} chunks")
            last = time.time()
        time.sleep(20)
    note("END all tiers scored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
