"""Control for the zero-cost artefact: covers that agree with an exact DCT.

F28 / F35: the released SCRIPT cost is zero at two kinds of coefficient -
A, where the cover's compressor and an exact DCT round differently (|e| > 0.5),
and B, where the exact DCT lies on the rounding boundary (|e| = 0.5). The
review council asked for the control that separates "a property of the released
cost" from "an interaction with the compressor that wrote our covers":

  cover set X   the first 2,000 BOSSbase images compressed at QF 65 by rounding
                an exact DCT (same quantisation table, same file format). By
                construction no coefficient has |e| > 0.5, so kind A cannot
                occur; kind B still can.

Cells (released spatial patterns, i.e. the corrected harness, F36):

  X-rel_{p}       released cost on the exact-DCT covers       (A impossible)
  X-fix4_{p}      + no change where |e| = 0.5                  (A and B removed)
  S-fix4_{p}      standard covers, no change where |e| >= 0.5  (A and B removed,
                  the principled alternative to the floor of repair 3)

p = 0.05 and 0.2 bpnzac. Each is scored with DCTR + FLD (1,000 / 1,000 pairs)
against its own covers, and the changes at A and B are counted. To compare with
exp48 on the same images: released and floor on standard covers.

    python experiments/exp51_exact_dct_control.py --smoke
    python experiments/exp51_exact_dct_control.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dctr_sets as ds  # noqa: E402
import exp13_fullscale as e13  # noqa: E402
import exp16_dctr_fullscale as e16  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402
import exp40_script_fix as e40  # noqa: E402

N, PAYLOADS = 2000, (0.05, 0.2)
XCOVERS = paths.DATA / "BOSSbase_QF65_exactdct"
PGM = paths.DATA / "BOSSbase_1.01"

_prev = e13.matlab_call


def _call(cell, manifest, out_dir, csv_path, seed_base):
    if "ctl" not in cell:
        return _prev(cell, manifest, out_dir, csv_path, seed_base)
    q = e13.COVER_QF
    return (f"script_embed_shard('{e13.REPO.as_posix()}','{manifest.as_posix()}',"
            f"'{out_dir.as_posix()}','{csv_path.as_posix()}',{cell['payload']},{q},{q},"
            f"{seed_base},0,0,0,0,{cell['fix']},1,0)")


e13.matlab_call = _call


def exact(i: int, qt: np.ndarray) -> np.ndarray:
    from PIL import Image
    from scipy.fft import dctn

    x = np.asarray(Image.open(PGM / f"{i}.pgm"), float) - 128
    h, w = x.shape
    b = x.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
    return dctn(b, axes=(2, 3), norm="ortho") / qt


def make_covers(n: int, note) -> None:
    import jpeglib

    XCOVERS.mkdir(parents=True, exist_ok=True)
    made = 0
    for i in range(1, n + 1):
        dest = XCOVERS / f"{i}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            continue
        im = jpeglib.read_dct(str(e40.COVERS / f"{i}.jpg"))
        real = exact(i, np.asarray(im.qt[0]).astype(float))
        im.Y = (np.sign(real) * np.floor(np.abs(real) + 0.5)).astype(np.int16)
        im.write_dct(str(dest))
        made += 1
    note(f"exact-DCT covers: {made} written, {n} present in {XCOVERS.name}")


def counts(covers: pathlib.Path, stego: pathlib.Path, n: int) -> dict:
    import jpeglib

    acc = []
    for i in range(1, n + 1):
        f = stego / f"{i}.jpg"
        if not f.exists():
            continue
        im = jpeglib.read_dct(str(covers / f"{i}.jpg"))
        c = np.asarray(im.Y).astype(np.int32)
        ae = np.abs(exact(i, np.asarray(im.qt[0]).astype(float)) - c)
        B = np.abs(ae - 0.5) < 1e-9
        A = (ae > 0.5) & ~B
        ch = np.asarray(jpeglib.read_dct(str(f)).Y).astype(np.int32) != c
        acc.append([A.sum(), B.sum(), ch.sum(), (ch & A).sum(), (ch & B).sum()])
    m = np.mean(acc, 0)
    return dict(images=len(acc), kind_A=float(m[0]), kind_B=float(m[1]), changes=float(m[2]),
                changes_at_A=float(m[3]), changes_at_B=float(m[4]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 images per cell, no detector")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = paths.ensure_results("exp51_smoke" if args.smoke else "exp51_exact_dct_control")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    n, chunk = (8, 8) if args.smoke else (N, args.chunk)
    note(f"START exact-DCT control, {n} images, payloads {PAYLOADS}")
    make_covers(n, note)
    groups = [("X-rel", 0, XCOVERS), ("X-fix4", 4, XCOVERS), ("S-fix4", 4, e40.COVERS)]
    cs = [dict(name=f"{tag}_{p}", kind="script", payload=p, fix=fix, ctl=1, covers=cov, tag=tag)
          for p in PAYLOADS for tag, fix, cov in groups]
    for cov in (XCOVERS, e40.COVERS):
        part = [c for c in cs if c["covers"] == cov]
        e21.embed(root, part, [cov / f"{i}.jpg" for i in range(1, n + 1)], chunk, args.workers, note)
    note(f"{'cell':<14}{'images':>8}{'error-free':>12}{'PSNR':>8}{'changes':>9}{'at A':>8}{'at B':>8}"
         f"{'(A, B per image)':>20}")
    info = {}
    for c in cs:
        d = e40.delivery(root, c["name"], n)
        k = counts(c["covers"], root / c["name"] / "stego", min(n, 300))
        info[c["name"]] = dict(delivery=d, counts=k)
        note(f"{c['name']:<14}{d['images']:>8}{d['error_free']:>12}{(d['median_psnr_db'] or 0):>8.1f}"
             f"{k['changes']:>9.0f}{k['changes_at_A']:>8.1f}{k['changes_at_B']:>8.1f}"
             f"{k['kind_A']:>12.0f},{k['kind_B']:>5.0f}")
    if args.smoke:
        note("END (smoke)")
        return 0

    fdir = root / "features"
    ds.extract(fdir, "cover_exact", [(str(i), XCOVERS / f"{i}.jpg") for i in range(1, N + 1)],
               e13.COVER_QF, note)
    Xx, xstems = ds.load(fdir, "cover_exact")
    Xs, sstems = e16.load_set(e40.E16_OUT, "cover", N, ds.FCHUNK)
    for c in cs:
        ds.extract(fdir, c["name"], e40.stego_files(root, c["name"], N), e13.COVER_QF, note)
    note("  null on the exact-DCT covers")
    ds.null_test(root / "exact_covers", Xx, xstems, note)
    res = []
    for cov, X, stems, sub in ((XCOVERS, Xx, xstems, "exact_covers"), (e40.COVERS, Xs, sstems, "standard_covers")):
        res += ds.score(root / sub, X, stems,
                        [(c["name"], dict(payload_bpnzac=c["payload"], n_images=N),
                          lambda c=c: ds.load(fdir, c["name"], N // ds.FCHUNK))
                         for c in cs if c["covers"] == cov], note)
    pe = {r["cell"]: r for r in res}
    ref = {}
    f48 = paths.RESULTS / "exp48_script_released_patterns" / "pilot_2000" / "results.json"
    if f48.exists():
        ref = {r["cell"]: r["p_e_mean"] for r in json.loads(f48.read_text())}
    note("")
    note("2,000 images, DCTR + FLD, released spatial patterns")
    note(f"  {'cell':<14}{'covers':<10}{'cost':<34}{'P_E':>8}{'sd':>8}{'delivered':>11}{'at A':>7}{'at B':>7}")
    label = {"X-rel": ("exact DCT", "released"), "X-fix4": ("exact DCT", "no change where |e| = 0.5"),
             "S-fix4": ("standard", "no change where |e| >= 0.5")}
    for c in cs:
        r, i = pe[c["name"]], info[c["name"]]
        note(f"  {c['name']:<14}{label[c['tag']][0]:<10}{label[c['tag']][1]:<34}{r['p_e_mean']:>8.4f}"
             f"{r['p_e_std']:>8.4f}{100 * i['delivery']['rate']:>10.2f}%"
             f"{i['counts']['changes_at_A']:>7.1f}{i['counts']['changes_at_B']:>7.1f}")
    for p in PAYLOADS:
        for tag, text in (("rel", "released"), ("relfix3", "floor 0.1")):
            v = ref.get(f"SCRIPT-{tag}_{p}")
            if v is not None:
                note(f"  {'(exp48) ' + str(p):<14}{'standard':<10}{text:<34}{v:>8.4f}")
    (root / "summary.json").write_text(json.dumps(dict(
        info=info, p_e={k: v["p_e_mean"] for k, v in pe.items()}), indent=2, default=str), encoding="utf-8")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
