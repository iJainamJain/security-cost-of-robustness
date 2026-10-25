"""Is the "first 5,000 images" subset representative?

Every network result and Tables 2e, 2f, 3, 5b use the first 5,000 BOSSbase
images by file number, and BOSSbase is ordered by source (F9), so that subset is
not a random half. The review council asked for the check: DCTR + FLD on a
RANDOM 5,000 of the 10,000, same protocol, against the first 5,000.

Features already exist for all 10,000 (exp16); this only re-scores. SCRIPT
cells are left out because they are being re-embedded (F36).

    python experiments/exp49_random_subset.py
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dctr_sets as ds  # noqa: E402
import exp16_dctr_fullscale as e16  # noqa: E402

E16_OUT = paths.RESULTS / "exp16_dctr_fullscale"
CELLS = ([f"J-UNIWARD_{p}" for p in (0.05, 0.1, 0.2, 0.3)]
         + [f"{m}_{p}" for m in ("GMAS", "ROAST", "MINICER") for p in (0.05, 0.1)])


def main() -> int:
    out = paths.ensure_results("exp49_random_subset")
    log = (out / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    note("START random 5,000 of 10,000, DCTR + FLD")
    Xall, stems_all = e16.load_set(E16_OUT, "cover", 10000, ds.FCHUNK)
    pick = np.sort(np.random.default_rng(20261011).choice(len(stems_all), 5000, replace=False))
    Xcov, cov_stems = Xall[pick], [stems_all[i] for i in pick]
    del Xall
    (out / "subset.json").write_text(json.dumps(cov_stems), encoding="utf-8")
    note(f"subset: {len(cov_stems)} images, {sum(int(s) <= 5000 for s in cov_stems)} of them "
         f"from the first 5,000")
    null = ds.null_test(out, Xcov, cov_stems, note)
    res = ds.score(out / "random_5000", Xcov, cov_stems,
                   [(c, dict(n_images=5000, subset="random"),
                     lambda c=c: e16.load_set(E16_OUT, c, 10000, ds.FCHUNK)) for c in CELLS], note)
    first = {r["cell"]: r for r in json.loads((E16_OUT / "tier_5000" / "results.json").read_text())}
    full = {r["cell"]: r for r in json.loads((E16_OUT / "tier_10000" / "results.json").read_text())}
    note("")
    note(f"null on the random subset: {null['p_e_mean']:.4f} +/- {null['p_e_std']:.4f}")
    note(f"{'cell':<18}{'random 5,000':>14}{'sd':>8}{'first 5,000':>13}{'difference':>12}{'all 10,000':>12}")
    for r in res:
        a, b = r["p_e_mean"], first[r["cell"]]["p_e_mean"]
        note(f"{r['cell']:<18}{a:>14.4f}{r['p_e_std']:>8.4f}{b:>13.4f}{a - b:>12.4f}"
             f"{full[r['cell']]['p_e_mean']:>12.4f}")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
