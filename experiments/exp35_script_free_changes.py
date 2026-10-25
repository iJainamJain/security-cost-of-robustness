"""Stage 4 of the SCRIPT flat-detectability question: the zero-cost coefficients.

SCRIPT's costs are side-informed: with e = (exact DCT of the precover) - (cover
JPEG coefficient), a change in the direction of e costs (1 - 2|e|) times the
base cost, and costGISS.m clips |e| at 0.5 because, in its own words, "image
compressed with imwrite(imread) is different from that with idct(dct)". Our
covers are compressed with imwrite, as BOSSbase covers usually are. Wherever
that compressor and an exact DCT round a coefficient differently, |e| > 0.5,
the clipped cost is exactly zero, and the coder may change the coefficient for
free whatever the message.

A first look (100 images) found about 340 such coefficients per image, about
150 of them changed at 0.05 bpnzac and the same number at 0.2, mostly turning a
+-1 into 0. That is a payload-independent component, and a candidate for what
keeps SCRIPT's P_E at 0.38-0.40 at every payload.

This measures it the way exp34 measures the DC and AC parts. Per image the
changes are split into those at zero-cost ("free") coefficients and the rest:

    free_only   cover plus only the changes at free coefficients
    no_free     stego with the changes at free coefficients undone

If the free changes are the flat signal, free_only keeps the full image's P_E
at both payloads and no_free behaves like an ordinary scheme: near chance at
0.05, detectable at 0.2. First 2,000 images, cover features from exp34.

    python experiments/exp35_script_free_changes.py
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
import exp16_dctr_fullscale as e16  # noqa: E402
import exp34_script_ablation as e34  # noqa: E402

PGM = paths.DATA / "BOSSbase_1.01"
VARIANTS = ("free_only", "no_free")
SETS = [f"{v}_{p}" for p in e34.PAYLOADS for v in VARIANTS]
N, CHUNK = 2000, 250


def free_mask(i: int, c: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """Coefficients where the cover JPEG differs from the rounded exact DCT."""
    from PIL import Image
    from scipy.fft import dctn

    x = np.asarray(Image.open(PGM / f"{i}.pgm"), float) - 128
    h, w = x.shape
    b = x.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    real = dctn(b, axes=(2, 3), norm="ortho") / qt
    return np.abs(real - c) > 0.5


def extract(out: pathlib.Path, note) -> None:
    import jpeglib
    import torch
    from dctr import DCTR

    torch.set_num_threads(8)
    ex = DCTR(65, "cpu")
    fdir = out / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    for k in range((N + CHUNK - 1) // CHUNK):
        dest = fdir / f"chunk_{k:03d}.npz"
        if dest.exists():
            continue
        t0 = time.time()
        feats = {name: [] for name in SETS}
        stems, counts = [], []
        for i in range(k * CHUNK + 1, min((k + 1) * CHUNK, N) + 1):
            files = [e34.E21 / f"SCRIPT-R0_{p}" / "stego" / f"{i}.jpg" for p in e34.PAYLOADS]
            if not all(f.exists() and f.stat().st_size > 0 for f in files):
                continue
            im = jpeglib.read_dct(str(e34.COVERS / f"{i}.jpg"))
            c, qt = np.asarray(im.Y).astype(np.int16), np.asarray(im.qt[0])
            free = free_mask(i, c, qt.astype(float))
            row = [int(free.sum())]
            for p, f in zip(e34.PAYLOADS, files):
                s = np.asarray(jpeglib.read_dct(str(f)).Y).astype(np.int16)
                d = s - c
                feats[f"free_only_{p}"].append(ex.features(c + d * free, qt))
                feats[f"no_free_{p}"].append(ex.features(c + d * ~free, qt))
                ch = d != 0
                row += [int(ch.sum()), int((ch & free).sum()),
                        int((ch & free & (c != 0) & (s == 0)).sum()),
                        int((ch & free & (c == 0)).sum())]
            stems.append(str(i))
            counts.append(row)
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, stems=np.array(stems), counts=np.array(counts),
                 **{name: np.asarray(v, dtype=np.float32) for name, v in feats.items()})
        tmp.replace(dest)
        note(f"  features chunk {k}: {len(stems)} images, {(time.time() - t0) / 60:.1f} min")


def main() -> int:
    out = paths.ensure_results("exp35_script_free_changes")
    log = out / "run.log"

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    note(f"START free-coefficient split: first {N} images, payloads {e34.PAYLOADS}")
    extract(out, note)
    X, stems, counts = {name: [] for name in SETS}, [], []
    for f in sorted((out / "features").glob("chunk_*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f)
        stems += [str(s) for s in z["stems"]]
        counts.append(z["counts"])
        for name in SETS:
            X[name].append(z[name])
    X = {name: np.concatenate(v) for name, v in X.items()}
    counts = np.concatenate(counts).astype(float)
    Xall, stems34 = e34.load(paths.RESULTS / "exp34_script_ablation")
    assert stems34 == stems, "exp34 and exp35 must cover the same images"
    m = counts.mean(0)
    note(f"per image: {m[0]:.0f} free coefficients; at {e34.PAYLOADS[0]}: {m[1]:.0f} changes, "
         f"{m[2]:.0f} at free coefficients ({m[3]:.0f} made zero, {m[4]:.0f} left zero); "
         f"at {e34.PAYLOADS[1]}: {m[5]:.0f} changes, {m[6]:.0f} at free coefficients "
         f"({m[7]:.0f} made zero, {m[8]:.0f} left zero)")

    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["set"] for r in results}
    for name in SETS:
        if name in done:
            continue
        note(f"  {name}")
        r, rows = e16.run_cell(Xall["cover"], X[name], stems, 5, 800, 200, note)
        variant, payload = name.rsplit("_", 1)
        results.append(dict(set=name, variant=variant, payload=float(payload), **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        e16.write_scores(out / "scores" / f"{name}.csv", rows)
        note(f"  -> {name} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}  "
             f"caught at 1% FA {100 * (1 - r['md_at_fa1_mean']):.1f}%")
    (out / "counts.json").write_text(json.dumps(dict(
        columns=["free", "changes_lo", "free_changes_lo", "free_made_zero_lo", "free_left_zero_lo",
                 "changes_hi", "free_changes_hi", "free_made_zero_hi", "free_left_zero_hi"],
        payloads=e34.PAYLOADS, mean=[float(v) for v in m],
        median=[float(v) for v in np.median(counts, 0)]), indent=2), encoding="utf-8")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
