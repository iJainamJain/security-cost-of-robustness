"""Stage 2 of the SCRIPT flat-detectability question (see exp32): which images
does the detector catch?

exp32 shows SCRIPT's changes at strength 0 scale with the message, so there is
no fixed mark. The other way a P_E can sit still while payload quadruples is a
split population: some images caught at any payload, the rest at none. This
re-reads the 10,000-image DCTR test scores (exp16, exp21) and reports P_E and
the detection rate at a 1% false-alarm rate within fifths of the images ordered
by texture (number of non-zero AC coefficients; Q1 is the smoothest fifth).
No training and no embedding.

    python experiments/exp33_script_by_texture.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp17_batch_pooling as e17  # noqa: E402

R = paths.RESULTS
S16 = R / "exp16_dctr_fullscale/tier_10000/scores"
S21 = R / "exp21_script_low_fullscale/tier_10000/scores"
CELLS = [("SCRIPT, R = 0", p, S21 / f"SCRIPT-R0_{p}.csv") for p in (0.05, 0.1, 0.2)] \
    + [("SCRIPT, R = 10", p, S16 / f"SCRIPT_{p}.csv") for p in (0.05, 0.1, 0.2)] \
    + [("J-UNIWARD", p, S16 / f"J-UNIWARD_{p}.csv") for p in (0.05, 0.1, 0.2, 0.3)]


def main() -> int:
    out = paths.ensure_results("exp33_script_by_texture")
    nz = {}
    for f in sorted((R / "exp13_fullscale/J-UNIWARD_0.05/csv").glob("chunk_*.csv")):
        with f.open(newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                if (r.get("nzAC") or "").strip().isdigit():
                    nz[r["cover"].strip()] = int(r["nzAC"])
    edges = np.quantile(list(nz.values()), [0.2, 0.4, 0.6, 0.8])
    q_of = {s: int(np.searchsorted(edges, v)) for s, v in nz.items()}
    print(f"{len(nz)} images; nzAC quintile edges {[int(e) for e in edges]}\n")
    print(f"{'cell':<22}{'all':>8}" + "".join(f"{'Q' + str(i + 1):>8}" for i in range(5))
          + "    caught at 1% false alarm, Q1..Q5 (one threshold for all images)")
    res = []
    for method, p, f in CELLS:
        by = {}                                     # seed -> lists
        with f.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["image"] in q_of:
                    by.setdefault(r["seed"], []).append(
                        (q_of[r["image"]], r["label"] == "stego", float(r["score"])))
        pe_all, pe_q, caught_q = [], [], []
        for rows in by.values():
            q = np.array([x[0] for x in rows])
            st = np.array([x[1] for x in rows])
            sc = np.array([x[2] for x in rows])
            pe_all.append(e17.p_error(sc[~st], sc[st]))
            pe_q.append([e17.p_error(sc[~st & (q == i)], sc[st & (q == i)]) for i in range(5)])
            thr = np.quantile(sc[~st], 0.99)
            caught_q.append([float((sc[st & (q == i)] > thr).mean()) for i in range(5)])
        pe_q, caught_q = np.mean(pe_q, 0), np.mean(caught_q, 0)
        res.append(dict(method=method, payload=p, p_e=float(np.mean(pe_all)),
                        p_e_by_quintile=[float(v) for v in pe_q],
                        caught_at_fa1_by_quintile=[float(v) for v in caught_q]))
        print(f"{method + ' ' + str(p):<22}{np.mean(pe_all):>8.4f}"
              + "".join(f"{v:>8.4f}" for v in pe_q) + "    "
              + " ".join(f"{100 * v:5.1f}%" for v in caught_q))
    (out / "by_texture.json").write_text(json.dumps(
        dict(nzAC_quintile_edges=[float(e) for e in edges], cells=res), indent=2),
        encoding="utf-8")
    print(f"\nwrote {out / 'by_texture.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
