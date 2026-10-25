"""DCTR + FLD on the side-informed reference (exp36), beside SCRIPT at R = 0.

Same detector, settings, splits and cover features as the 5,000- and
10,000-image tables (exp16, exp21): d_sub 800, L 200, five paired 50/50 splits,
covers from exp16's feature files. So each P_E here sits directly beside
SCRIPT's and J-UNIWARD's in Table 1 (10,000) and Table 3 (5,000):

    P_E(reference) - P_E(SCRIPT, R = 0) = what SCRIPT's coefficient selection
    costs in security.

Resumable: features per 500-image chunk, results per cell and tier.

    python experiments/exp37_si_reference_dctr.py
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp13_fullscale as e13  # noqa: E402
import exp16_dctr_fullscale as e16  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402
import exp36_si_reference as e36  # noqa: E402

ROOT = paths.RESULTS / "exp36_si_reference"
E16_OUT = paths.RESULTS / "exp16_dctr_fullscale"
FCHUNK = 500


def chunk_path(name: str, k: int) -> pathlib.Path:
    return ROOT / "features" / name / f"chunk_{k:03d}.npz"


def features(n_images: int, note) -> None:
    import torch

    from dctr import DCTR, DIM, read_jpeg
    torch.set_num_threads(8)
    ex = DCTR(e13.COVER_QF, "cpu")
    for c in e36.cells():
        ok = e21.ok_set(ROOT, c["name"])
        for k in range((n_images + FCHUNK - 1) // FCHUNK):
            dest = chunk_path(c["name"], k)
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            stems = [str(i) for i in range(k * FCHUNK + 1, min((k + 1) * FCHUNK, n_images) + 1)
                     if str(i) in ok]
            feats = np.empty((len(stems), DIM), dtype=np.float32)
            for j, s in enumerate(stems):
                feats[j] = ex.features(*read_jpeg(ROOT / c["name"] / "stego" / f"{s}.jpg"))
            tmp = dest.with_suffix(".tmp.npz")
            np.savez(tmp, feats=feats, stems=np.array(stems))
            tmp.replace(dest)
            note(f"  features {c['name']} chunk {k}: {len(stems)} images, "
                 f"{(time.time() - t0) / max(len(stems), 1):.2f} s/img")


def load(name: str, n_images: int):
    feats, stems = [], []
    for k in range((n_images + FCHUNK - 1) // FCHUNK):
        z = np.load(chunk_path(name, k))
        feats.append(z["feats"]); stems += [str(s) for s in z["stems"]]
    return np.concatenate(feats), stems


def score(n_images: int, note) -> None:
    tdir = ROOT / f"tier_{n_images}"
    (tdir / "scores").mkdir(parents=True, exist_ok=True)
    res_path = tdir / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["cell"] for r in results}
    if all(c["name"] in done for c in e36.cells()):
        return
    Xcov, cov_stems = e16.load_set(E16_OUT, "cover", n_images, FCHUNK)
    pos = {s: i for i, s in enumerate(cov_stems)}
    null = json.loads((E16_OUT / f"tier_{n_images}" / "null_test.json").read_text())
    note(f"TIER {n_images}: {len(cov_stems)} covers loaded (null {null['p_e_mean']:.4f})")
    for c in e36.cells():
        if c["name"] in done:
            continue
        Xs, stems = load(c["name"], n_images)
        keep = [i for i, s in enumerate(stems) if s in pos]
        stems, Xs = [stems[i] for i in keep], Xs[keep]
        note(f"  {c['name']} ({len(stems)} pairs)")
        r, rows = e16.run_cell(Xcov[[pos[s] for s in stems]], Xs, stems, 5, 800, 200, note)
        e16.write_scores(tdir / "scores" / f"{c['name']}.csv", rows)
        results.append(dict(cell=c["name"], method="SI-UERD (SCRIPT without selection)",
                            payload_bpnzac=c["payload"], detector="DCTR+FLD",
                            n_images=n_images, null_p_e=null["p_e_mean"],
                            null_sd=null["p_e_std"], **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        note(f"  -> {c['name']} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}  "
             f"caught at 1% FA {100 * (1 - r['md_at_fa1_mean']):.1f}%")


def table(note) -> None:
    def ref(folder, tier):
        f = paths.RESULTS / folder / f"tier_{tier}" / "results.json"
        return {r["cell"]: r["p_e_mean"] for r in json.loads(f.read_text())} if f.exists() else {}

    for tier in e36.TIERS:
        f = ROOT / f"tier_{tier}" / "results.json"
        if not f.exists():
            continue
        si = {r["cell"]: r for r in json.loads(f.read_text())}
        sc, ju = ref("exp21_script_low_fullscale", tier), ref("exp16_dctr_fullscale", tier)
        note("")
        note(f"{tier} images, DCTR + FLD     reference   SCRIPT R=0   cost of selection   J-UNIWARD")
        for c in e36.cells():
            p = c["payload"]
            a, b = si[c["name"]]["p_e_mean"], sc.get(f"SCRIPT-R0_{p}")
            note(f"  {p:<24}{a:>9.4f}{b:>13.4f}{a - b:>20.4f}{ju.get(f'J-UNIWARD_{p}', float('nan')):>12.4f}")


def main() -> int:
    log = ROOT / "dctr_run.log"

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    note("START DCTR on the side-informed reference")
    for tier in e36.TIERS:
        have = min(len(e21.ok_set(ROOT, c["name"])) for c in e36.cells())
        if have < 0.95 * tier:
            note(f"tier {tier}: only {have} images embedded so far - skipped")
            continue
        features(tier, note)
        score(tier, note)
    table(note)
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
