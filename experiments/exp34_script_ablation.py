"""Stage 3 of the SCRIPT flat-detectability question: which changes does the
detector see?

exp32: at strength 0 SCRIPT's changes scale with the message - no fixed mark.
exp33: its P_E is about 0.38 in every texture fifth at every payload - not a
split population either. So this takes the stego images apart. For each image
the difference stego - cover is split into three disjoint parts,

    DC          changes to a block's DC coefficient (about 7% of all changes)
    zero AC     changes that turn a zero AC coefficient into +-1
    nonzero AC  changes to an AC coefficient that was already non-zero

and DCTR + FLD is asked to tell the cover from "cover plus one part" and from
"stego minus the DC part". Whatever part carries the signal keeps the P_E of
the full stego image; a part the detector cannot see reads chance. Nothing is
embedded: the variants are built from stego files that already exist, and the
features are computed from the coefficient arrays directly.

First 2,000 images (the set of F15), payloads 0.05 and 0.2, five paired
1,000 / 1,000 splits, d_sub 800, L 200 - the settings of exp16.

Resumable: features are saved per 250-image chunk, results per finished set.

    python experiments/exp34_script_ablation.py
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

COVERS = paths.DATA / "BOSSbase_QF65_full"
E21 = paths.RESULTS / "exp21_script_low_fullscale"
PAYLOADS = (0.05, 0.2)
VARIANTS = ("full", "dc_only", "no_dc", "zero_ac_only", "nonzero_ac_only")
SETS = ["cover"] + [f"{v}_{p}" for p in PAYLOADS for v in VARIANTS]
CHUNK = 250


def variants(c: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    d = s - c
    dc = np.zeros(c.shape, bool)
    dc[:, :, 0, 0] = True
    zero = (c == 0) & ~dc
    nonzero = (c != 0) & ~dc
    return dict(full=s, dc_only=c + d * dc, no_dc=c + d * ~dc,
                zero_ac_only=c + d * zero, nonzero_ac_only=c + d * nonzero)


def extract(out: pathlib.Path, n: int, note) -> None:
    import jpeglib
    import torch
    from dctr import DCTR

    torch.set_num_threads(8)
    ex = DCTR(65, "cpu")
    fdir = out / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    for k in range((n + CHUNK - 1) // CHUNK):
        dest = fdir / f"chunk_{k:03d}.npz"
        if dest.exists():
            continue
        t0 = time.time()
        feats = {name: [] for name in SETS}
        stems = []
        for i in range(k * CHUNK + 1, min((k + 1) * CHUNK, n) + 1):
            files = [E21 / f"SCRIPT-R0_{p}" / "stego" / f"{i}.jpg" for p in PAYLOADS]
            if not all(f.exists() and f.stat().st_size > 0 for f in files):
                continue                    # the embedder failed on this image
            im = jpeglib.read_dct(str(COVERS / f"{i}.jpg"))
            c, qt = np.asarray(im.Y).astype(np.int16), np.asarray(im.qt[0])
            feats["cover"].append(ex.features(c, qt))
            for p, f in zip(PAYLOADS, files):
                s = np.asarray(jpeglib.read_dct(str(f)).Y).astype(np.int16)
                for v, arr in variants(c, s).items():
                    feats[f"{v}_{p}"].append(ex.features(arr, qt))
            stems.append(str(i))
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, stems=np.array(stems),
                 **{name: np.asarray(v, dtype=np.float32) for name, v in feats.items()})
        tmp.replace(dest)
        note(f"  features chunk {k}: {len(stems)} images, "
             f"{(time.time() - t0) / max(len(stems), 1) / len(SETS):.2f} s per extraction, "
             f"{(time.time() - t0) / 60:.1f} min")


def load(out: pathlib.Path):
    X, stems = {name: [] for name in SETS}, []
    for f in sorted((out / "features").glob("chunk_*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f)
        stems += [str(s) for s in z["stems"]]
        for name in SETS:
            X[name].append(z[name])
    return {name: np.concatenate(v) for name, v in X.items()}, stems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="exp34_script_ablation")
    args = ap.parse_args()
    out = paths.ensure_results(args.out)
    log = out / "run.log"

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    note(f"START ablation: first {args.n} images, payloads {PAYLOADS}, {len(SETS)} feature sets")
    extract(out, args.n, note)
    X, stems = load(out)
    note(f"features loaded: {len(stems)} images")

    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["set"] for r in results}
    if "NULL" not in done:
        perm = np.random.default_rng(20261002).permutation(len(stems))
        half = len(stems) // 2
        ia, ib = np.sort(perm[:half]), np.sort(perm[half:2 * half])
        note("  NULL (cover vs cover, random groups)")
        r, _ = e16.run_cell(X["cover"][ia], X["cover"][ib], [stems[i] for i in ia],
                            args.seeds, 800, 200, note)
        results.append(dict(set="NULL", **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        note(f"  -> NULL P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}")
    for name in SETS[1:]:
        if name in done:
            continue
        note(f"  {name}")
        r, rows = e16.run_cell(X["cover"], X[name], stems, args.seeds, 800, 200, note)
        variant, payload = name.rsplit("_", 1)
        results.append(dict(set=name, variant=variant, payload=float(payload), **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        e16.write_scores(out / "scores" / f"{name}.csv", rows)
        note(f"  -> {name} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}  "
             f"caught at 1% FA {100 * (1 - r['md_at_fa1_mean']):.1f}%")

    by = {r["set"]: r for r in results}
    note("")
    note(f"{'variant':<18}" + "".join(f"{'P_E at ' + str(p):>14}" for p in PAYLOADS)
         + f"   (null {by['NULL']['p_e_mean']:.4f})")
    for v in VARIANTS:
        note(f"{v:<18}" + "".join(f"{by[f'{v}_{p}']['p_e_mean']:>14.4f}" for p in PAYLOADS))
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
