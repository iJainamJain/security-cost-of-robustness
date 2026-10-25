"""DCTR + FLD on the second-source pair (BOSSbase-256, BOWS2-256).

Reads the stego images written by exp24_second_source.py and scores every cell
with the same detector as the main experiments: DCTR features (Python port),
FLD ensemble d_sub 800, L 200, five paired 50/50 splits, and a cover-vs-cover
null test on randomly drawn groups for each source.

Stages (each resumable):
  features  covers and the 12 cells of each source, one .npz per cell
  null      cover vs cover per source
  score     every cell; per-image test scores are kept for later pooling
  table     both sources side by side, with the 512x512 figures on the same
            5,000 BOSSbase images for reference

The DCTR port was validated against the MATLAB reference at 512x512 only
(exp15). It is size-agnostic by construction, but that is not a test.

    python experiments/exp25_second_source_dctr.py --wait PID
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp13_fullscale as e13  # noqa: E402
import exp16_dctr_fullscale as e16  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402
import exp24_second_source as e24  # noqa: E402

BASE = paths.RESULTS / "exp24_second_source"
REF_512 = [paths.RESULTS / "exp16_dctr_fullscale/tier_5000/results.json",
           paths.RESULTS / "exp21_script_low_fullscale/tier_5000/results.json"]


def stems_sorted(names) -> list[str]:
    return sorted(names, key=lambda s: (len(s), s))


def features(src: str, note) -> None:
    import torch

    from dctr import DCTR, DIM, read_jpeg
    torch.set_num_threads(8)
    ex = DCTR(e13.COVER_QF, "cpu")
    root, fdir = BASE / src, BASE / src / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    todo = [("cover", e24.cover_dir(src), None)]
    todo += [(c["name"], root / c["name"] / "stego", c["name"]) for c in e24.cells()]
    for name, d, cell in todo:
        dest = fdir / f"{name}.npz"
        if dest.exists():
            continue
        have = {p.stem for p in d.glob("*.jpg") if p.stat().st_size > 0}
        if cell is not None:
            have &= e21.ok_set(root, cell)      # only images that embedded
        stems = stems_sorted(have)
        feats = np.empty((len(stems), DIM), dtype=np.float32)
        t0 = time.time()
        for i, s in enumerate(stems):
            feats[i] = ex.features(*read_jpeg(d / f"{s}.jpg"))
        tmp = fdir / f"{name}.tmp.npz"
        np.savez(tmp, feats=feats, stems=np.array(stems))
        tmp.replace(dest)
        note(f"{src} features {name}: {len(stems)} images, "
             f"{(time.time() - t0) / max(len(stems), 1):.3f} s/img")


def load(src: str, name: str):
    z = np.load(BASE / src / "features" / f"{name}.npz")
    return z["feats"], [str(s) for s in z["stems"]]


def score(src: str, seeds: int, n_est: int, note) -> None:
    root = BASE / src / "dctr"
    (root / "scores").mkdir(parents=True, exist_ok=True)
    Xcov, cov_stems = load(src, "cover")
    pos = {s: i for i, s in enumerate(cov_stems)}

    null_path = root / "null_test.json"
    if not null_path.exists():
        # Random groups: image sets are ordered by source camera.
        perm = np.random.default_rng(20261002).permutation(len(cov_stems))
        half = len(cov_stems) // 2
        ia, ib = np.sort(perm[:half]), np.sort(perm[half:2 * half])
        note(f"{src} NULL TEST (cover vs cover, random groups)")
        r, rows = e16.run_cell(Xcov[ia], Xcov[ib], [cov_stems[i] for i in ia],
                               seeds, 800, n_est, note)
        e16.write_scores(root / "scores" / "NULL_TEST.csv", rows)
        null_path.write_text(json.dumps(r, indent=2), encoding="utf-8")
        note(f"{src} NULL P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}")
    null = json.loads(null_path.read_text())

    res_path = root / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["cell"] for r in results}
    for c in e24.cells():
        if c["name"] in done:
            continue
        Xs, stems = load(src, c["name"])
        keep = [i for i, s in enumerate(stems) if s in pos]
        stems, Xs = [stems[i] for i in keep], Xs[keep]
        note(f"{src} {c['name']} ({len(stems)} pairs)")
        r, rows = e16.run_cell(Xcov[[pos[s] for s in stems]], Xs, stems, seeds, 800,
                               n_est, note)
        e16.write_scores(root / "scores" / f"{c['name']}.csv", rows)
        results.append(dict(cell=c["name"], kind=c["kind"], payload_bpnzac=c["payload"],
                            source=src, image_size=256, detector="DCTR+FLD",
                            null_p_e=null["p_e_mean"], null_sd=null["p_e_std"], **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        note(f"-> {src} {c['name']} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f} "
             f"(null {null['p_e_mean']:.4f})")


def table(note) -> None:
    ref = {}
    for f in REF_512:
        if f.exists():
            for r in json.loads(f.read_text()):
                ref[r["cell"]] = r["p_e_mean"]
    res, null = {}, {}
    for src in e24.SOURCES:
        f = BASE / src / "dctr" / "results.json"
        res[src] = {r["cell"]: r for r in json.loads(f.read_text())} if f.exists() else {}
        n = BASE / src / "dctr" / "null_test.json"
        null[src] = json.loads(n.read_text())["p_e_mean"] if n.exists() else float("nan")
    srcs = list(e24.SOURCES)
    note("DCTR + FLD, 5,000 images per source")
    note(f"{'cell':<16}" + "".join(f"{s:>14}" for s in srcs) + f"{'BOSSbase-512':>14}")
    note(f"{'(null)':<16}" + "".join(f"{null[s]:>14.4f}" for s in srcs))
    for c in e24.cells():
        line = f"{c['name']:<16}"
        for s in srcs:
            r = res[s].get(c["name"])
            line += f"{r['p_e_mean']:>14.4f}" if r else f"{'-':>14}"
        v = ref.get(c["name"])
        line += f"{v:>14.4f}" if v is not None else f"{'-':>14}"
        note(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-est", type=int, default=200)
    ap.add_argument("--wait", type=int, default=None,
                    help="wait for this process id to exit before starting")
    args = ap.parse_args()
    runlog = (BASE / "dctr_run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    if args.wait:
        note(f"waiting for process {args.wait} to exit...")
        while str(args.wait) in subprocess.run(
                ["tasklist", "/FI", f"PID eq {args.wait}", "/NH"],
                capture_output=True, text=True).stdout:
            time.sleep(60)
    note("START DCTR on the second-source pair")
    for src in e24.SOURCES:
        features(src, note)
    for src in e24.SOURCES:
        score(src, args.seeds, args.n_est, note)
    table(note)
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
