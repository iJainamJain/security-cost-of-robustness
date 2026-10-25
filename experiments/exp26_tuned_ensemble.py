"""Does tuning the FLD ensemble change DCTR's numbers?  (journal roadmap)

Every DCTR figure so far uses fixed ensemble settings: subspace dimension
d_sub = 800, L = 200 base learners. The reference ensemble (Kodovsky, Fridrich,
Holub 2012) chooses d_sub by OUT-OF-BAG error on the training set. A reviewer
can fairly ask whether our fixed choice handicaps the feature-based detector,
especially now that it is being compared against CNNs. This measures it.

For each cell and each of the five paired 50/50 splits (the SAME splits as
exp16/exp21, so the comparison with the fixed-setting numbers is like for like):
  1. search: for each candidate d_sub, train a small ensemble (L_search
     learners) on the training half and compute its out-of-bag error - each
     training image is voted on only by learners whose bootstrap sample missed
     it. The test half is never touched.
  2. final: train L = 200 learners at the d_sub with the lowest OOB error and
     score the test half.

13 cells at the 5,000-image tier: the ones where P_E is not already ~0.
A cover-vs-cover null test runs first, through the same search.

    python experiments/exp26_tuned_ensemble.py --selftest
    python experiments/exp26_tuned_ensemble.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp16_dctr_fullscale as e16  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402

E16_OUT = paths.RESULTS / "exp16_dctr_fullscale"
E21_OUT = paths.RESULTS / "exp21_script_low_fullscale"
TIER = 5000
CANDIDATES = (200, 400, 800, 1200, 1600, 2000)
CELLS = ["J-UNIWARD_0.2", "SCRIPT-R0_0.1", "GMAS_0.05", "J-UNIWARD_0.1", "ROAST_0.05",
         "MINICER_0.05", "SCRIPT-R0_0.05", "J-UNIWARD_0.3", "SCRIPT-R0_0.2",
         "J-UNIWARD_0.05", "GMAS_0.1", "ROAST_0.1", "MINICER_0.1"]


class OOBFLD:
    """FLD ensemble on random subspaces with bootstrap bags, keeping OOB votes.

    Base learner is that of e16.LeanFLD (sklearn LDA, lsqr, Ledoit-Wolf
    shrinkage). With pair_bags=False the random draws are LeanFLD's too, so the
    two classes give identical scores (checked by --selftest).

    pair_bags=True is what the search needs. The training matrix is covers
    followed by their own stegos, and a bootstrap over single images often
    puts a cover in the bag and leaves its stego out. That stego is then
    "out of bag" while its near-identical twin sits in the bag with the
    OPPOSITE label, so it is voted cover: the OOB error comes out above 0.5 on
    a cell the test set detects easily. Drawing PAIRS keeps twins together,
    as the reference ensemble does.
    """

    def __init__(self, d_sub, n_estimators, seed, pair_bags=True):
        self.d_sub, self.n_estimators, self.seed = d_sub, n_estimators, seed
        self.pair_bags = pair_bags

    def fit(self, X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        y = np.asarray(y).ravel()
        n, n_features = X.shape
        d = min(self.d_sub, n_features)
        rng = np.random.default_rng(self.seed)
        self._sub, self._clf = [], []
        votes, counts = np.zeros(n), np.zeros(n)
        for _ in range(self.n_estimators):
            cols = rng.choice(n_features, d, replace=False)
            if self.pair_bags:
                m = n // 2                           # rows i and i + m are a pair
                bp = rng.choice(m, m, replace=True)
                bag = np.concatenate([bp, bp + m])
                op = np.setdiff1d(np.arange(m), bp)
                oob = np.concatenate([op, op + m])
            else:
                bag = rng.choice(n, n, replace=True)
                oob = np.setdiff1d(np.arange(n), bag)
            clf = LDA(solver="lsqr", shrinkage="auto")
            clf.fit(X[np.ix_(bag, cols)].astype(np.float64), y[bag])
            # Keep only the direction and offset. A fitted sklearn LDA also holds
            # its d x d covariance (32 MB at d = 2000); 200 of those exhausted
            # the laptop's memory on the first run of this script.
            w, b = clf.coef_.ravel().copy(), float(clf.intercept_[0])
            del clf
            votes[oob] += (X[np.ix_(oob, cols)].astype(np.float64) @ w + b > 0)
            counts[oob] += 1
            self._sub.append(cols); self._clf.append((w, b))
        seen = counts > 0
        frac = votes[seen] / counts[seen]
        # majority vote; an exact tie counts as half an error
        pred = np.where(frac > 0.5, 1.0, np.where(frac < 0.5, 0.0, 0.5))
        yy = y[seen]
        fa = np.abs(pred[yy == 0] - 0).mean()
        md = np.abs(pred[yy == 1] - 1).mean()
        self.oob_error = float(0.5 * (fa + md))
        return self

    def decision_function(self, X):
        votes = np.zeros(X.shape[0], dtype=np.float64)
        for cols, (w, b) in zip(self._sub, self._clf):
            votes += (X[:, cols].astype(np.float64) @ w + b > 0)
        return votes / len(self._clf)


def run_cell(Xc, Xs, stems, seeds, candidates, l_search, l_final, note):
    pes, revs, mds, rows, chosen, searches, oob_final = [], [], [], [], [], [], []
    pes800 = []
    n = len(stems)
    for seed in range(seeds):
        t0 = time.time()
        idx = np.random.default_rng(seed).permutation(n)       # same split as exp16
        tr, te = idx[:n // 2], idx[n // 2:]
        X = np.concatenate([Xc[tr], Xs[tr]])
        y = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
        oob = {}
        for d in candidates:
            oob[d] = OOBFLD(d, l_search, 1000 + seed).fit(X, y).oob_error
        best = min(candidates, key=lambda d: (oob[d], d))       # smaller d on ties
        m = OOBFLD(best, l_final, seed).fit(X, y)
        # Same class at the fixed setting, so a difference from the published
        # numbers can be split into "d_sub" and "pair bags".
        m800 = m if best == 800 else OOBFLD(800, l_final, seed).fit(X, y)
        del X
        pes800.append(e16.p_error(m800.decision_function(Xc[te]),
                                  m800.decision_function(Xs[te])))
        sc, ss = m.decision_function(Xc[te]), m.decision_function(Xs[te])
        pes.append(e16.p_error(sc, ss)); revs.append(e16.p_error(ss, sc))
        mds.append(e16.md_at_fa(sc, ss))
        chosen.append(int(best)); searches.append({str(k): round(v, 4) for k, v in oob.items()})
        oob_final.append(round(m.oob_error, 4))
        rows += [(seed, stems[i], "cover", float(v)) for i, v in zip(te, sc)]
        rows += [(seed, stems[i], "stego", float(v)) for i, v in zip(te, ss)]
        note(f"    seed {seed}: d_sub {best}  P_E {pes[-1]:.4f}  at 800: {pes800[-1]:.4f}  (OOB "
             + " ".join(f"{d}:{oob[d]:.3f}" for d in candidates)
             + f")  {time.time() - t0:.0f} s")
    return dict(p_e_mean=float(np.mean(pes)), p_e_std=float(np.std(pes)),
                p_e_per_seed=[float(v) for v in pes],
                p_e_800_pairbags_mean=float(np.mean(pes800)),
                p_e_800_pairbags_per_seed=[float(v) for v in pes800],
                p_e_reversed_mean=float(np.mean(revs)),
                md_at_fa1_mean=float(np.mean(mds)),
                d_sub_per_seed=chosen, oob_search_per_seed=searches,
                oob_error_final_per_seed=oob_final,
                n_train_pairs=n // 2, n_test_pairs=n - n // 2, seeds=list(range(seeds)),
                candidates=list(candidates), l_search=l_search, l_final=l_final), rows


def load_cell(name: str):
    if name.startswith("SCRIPT-R0"):
        return e21.load_new(E21_OUT, name, TIER)
    return e16.load_set(E16_OUT, name, TIER, 500)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--l-search", type=int, default=30)
    ap.add_argument("--l-final", type=int, default=200)
    ap.add_argument("--selftest", action="store_true",
                    help="600 pairs of one cell, two candidates, tiny ensembles")
    args = ap.parse_args()
    out = paths.ensure_results("exp26_tuned_ensemble_selftest" if args.selftest
                               else "exp26_tuned_ensemble")
    runlog = (out / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    fixed = {}
    for f in (E16_OUT / f"tier_{TIER}" / "results.json", E21_OUT / f"tier_{TIER}" / "results.json"):
        if f.exists():
            for r in json.loads(f.read_text()):
                fixed[r["cell"]] = r
    Xcov, cov_stems = e16.load_set(E16_OUT, "cover", TIER, 500)
    pos = {s: i for i, s in enumerate(cov_stems)}

    if args.selftest:
        Xs, stems = load_cell("GMAS_0.05")
        stems = [s for s in stems if s in pos][:600]
        spos = {s: i for i, s in enumerate(load_cell("GMAS_0.05")[1])}
        Xs = Xs[[spos[s] for s in stems]]
        Xc = Xcov[[pos[s] for s in stems]]
        # 1. at d_sub 800 the tuned class must reproduce LeanFLD exactly
        idx = np.random.default_rng(0).permutation(len(stems))
        tr, te = idx[:300], idx[300:]
        X = np.concatenate([Xc[tr], Xs[tr]]); y = np.r_[np.zeros(300), np.ones(300)]
        a = e16.LeanFLD(800, 20, 0).fit(X, y)
        b = OOBFLD(800, 20, 0, pair_bags=False).fit(X, y)
        same = np.array_equal(a.decision_function(Xc[te]), b.decision_function(Xc[te]))
        c = OOBFLD(800, 20, 0, pair_bags=True).fit(X, y)
        sc, ss = c.decision_function(Xc[te]), c.decision_function(Xs[te])
        at_half = 0.5 * ((sc > 0.5).mean() + (ss <= 0.5).mean())
        note(f"selftest: OOBFLD(single-image bags) equals LeanFLD: {same}; its OOB error "
             f"{b.oob_error:.3f} (biased by split pairs)")
        note(f"selftest: pair bags: OOB error {c.oob_error:.3f}, test error at the same "
             f"majority threshold {at_half:.3f} - these two should agree")
        r, _ = run_cell(Xc, Xs, stems, 2, (200, 400), 10, 20, note)
        note(f"selftest: P_E {r['p_e_mean']:.3f}, chosen {r['d_sub_per_seed']}")
        return 0 if same else 1

    note(f"START tuned ensemble: tier {TIER}, candidates {CANDIDATES}, L_search "
         f"{args.l_search}, L_final {args.l_final}, {len(CELLS)} cells")
    null_path = out / "null_test.json"
    if not null_path.exists():
        perm = np.random.default_rng(20261002).permutation(len(cov_stems))
        half = len(cov_stems) // 2
        ia, ib = np.sort(perm[:half]), np.sort(perm[half:2 * half])
        note("NULL TEST (cover vs cover, random groups)")
        r, rows = run_cell(Xcov[ia], Xcov[ib], [cov_stems[i] for i in ia], args.seeds,
                           CANDIDATES, args.l_search, args.l_final, note)
        e16.write_scores(out / "scores" / "NULL_TEST.csv", rows)
        null_path.write_text(json.dumps(r, indent=2), encoding="utf-8")
        note(f"NULL P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}")
    null = json.loads(null_path.read_text())

    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["cell"] for r in results}
    for name in CELLS:
        if name in done:
            continue
        Xs, stems = load_cell(name)
        keep = [i for i, s in enumerate(stems) if s in pos]
        stems, Xs = [stems[i] for i in keep], Xs[keep]
        note(f"{name} ({len(stems)} pairs)")
        r, rows = run_cell(Xcov[[pos[s] for s in stems]], Xs, stems, args.seeds,
                           CANDIDATES, args.l_search, args.l_final, note)
        e16.write_scores(out / "scores" / f"{name}.csv", rows)
        f = fixed.get(name, {})
        results.append(dict(cell=name, detector="DCTR+FLD (d_sub by OOB)", n_images=TIER,
                            fixed_p_e_mean=f.get("p_e_mean"), fixed_p_e_std=f.get("p_e_std"),
                            null_p_e=null["p_e_mean"], **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        fx = f.get("p_e_mean")
        note(f"-> {name} tuned P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}  | 800 with "
             f"pair bags {r['p_e_800_pairbags_mean']:.4f}  | published 800/200: "
             f"{fx if fx is None else round(fx, 4)}  | d_sub {r['d_sub_per_seed']}")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
