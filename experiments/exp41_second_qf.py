"""A second cover quality factor: does the picture hold at QF 75?

Everything so far uses covers at QF 65. This repeats a reduced grid on the same
first 5,000 BOSSbase images compressed at QF 75:

    J-UNIWARD           0.05, 0.1, 0.2     no channel
    GMAS, ROAST         0.05, 0.1          channel QF 75 -> 81 (upward, +6 as before)
    SCRIPT, strength 0  0.05, 0.1, 0.2     channel QF 75 -> 75

MINICER is left out: its step size was found by search at QF 65 and has not
been searched again at QF 75; running it with a borrowed setting would repeat
the mistake made with SCRIPT's strength (F15). ROAST's cost parameters and
SCRIPT at strength 0 carry over unchanged.

Detector: DCTR + FLD as in Table 3 (d_sub 800, L 200, five paired 2,500 / 2,500
splits), with DCTR's quantisation step set for QF 75, and its own null test.

Stages (each resumable): covers, embedding, features, scoring.

    python experiments/exp41_second_qf.py --smoke     # 8 images, no detector
    python experiments/exp41_second_qf.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dctr_sets as ds  # noqa: E402
import exp13_fullscale as e13  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402

QF, CHANNEL_UP, COUNT = 75, 81, 5000
COVERS = paths.DATA / f"BOSSbase_QF{QF}_full"
QF65 = {"exp16_dctr_fullscale": None, "exp21_script_low_fullscale": None}


def cells() -> list[dict]:
    out = [dict(name=f"J-UNIWARD_{p}", kind="juniward", payload=p) for p in (0.05, 0.1, 0.2)]
    for p in (0.05, 0.1):
        out += [dict(name=f"GMAS_{p}", kind="gmas", payload=p),
                dict(name=f"ROAST_{p}", kind="roast", payload=p)]
    out += [dict(name=f"SCRIPT-R0_{p}", kind="script", payload=p, R=0) for p in (0.05, 0.1, 0.2)]
    return out


def prepare(count: int, note) -> list[pathlib.Path]:
    from PIL import Image
    COVERS.mkdir(parents=True, exist_ok=True)
    made = 0
    for i in range(1, count + 1):
        dst = COVERS / f"{i}.jpg"
        if not dst.exists():
            Image.open(e13.PRECOVERS / f"{i}.pgm").save(dst, quality=QF)
            made += 1
    if made:
        note(f"wrote {made} covers at QF{QF} to {COVERS}")
    return [COVERS / f"{i}.jpg" for i in range(1, count + 1)]


def check_writer(root: pathlib.Path, note) -> bool:
    """PIL and MATLAB's imwrite were shown bit-identical at QF 65 (prep_covers.py).
    The channel step of every shard is MATLAB's imwrite, so check QF 75 too."""
    import jpeglib
    tmp = root / "writer_check"
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = "".join(f"imwrite(imread('{(e13.PRECOVERS / f'{i}.pgm').as_posix()}'),"
                  f"'{(tmp / f'{i}.jpg').as_posix()}','quality',{QF});" for i in (1, 2, 3))
    subprocess.run([e13.find_matlab(), "-batch", cmd], capture_output=True, text=True)
    same = 0
    for i in (1, 2, 3):
        a, b = jpeglib.read_dct(str(tmp / f"{i}.jpg")), jpeglib.read_dct(str(COVERS / f"{i}.jpg"))
        same += bool(np.array_equal(np.asarray(a.Y), np.asarray(b.Y))
                     and np.array_equal(np.asarray(a.qt[0]), np.asarray(b.qt[0])))
    note(f"cover writer check at QF{QF}: PIL equals MATLAB imwrite on {same} of 3 images "
         f"(coefficients and quantisation table)")
    return same == 3


def robustness(root: pathlib.Path, note) -> dict:
    out = {}
    note(f"{'cell':<16}{'images':>8}{'embedded':>10}{'error-free':>12}{'rate':>8}")
    for c in cells():
        rows = [r for f in sorted((root / c["name"] / "csv").glob("chunk_*.csv"))
                for r in e13.csv_rows(f)]
        ok = [r for r in rows if e13.row_ok(r)]
        clean = sum(1 for r in ok if (r.get("bit_error_rate") or "") != ""
                    and float(r["bit_error_rate"]) == 0)
        robust = c["kind"] != "juniward"
        out[c["name"]] = dict(images=len(rows), embedded=len(ok),
                              error_free=clean if robust else None,
                              rate=clean / len(rows) if rows and robust else None)
        note(f"{c['name']:<16}{len(rows):>8}{len(ok):>10}"
             f"{(str(clean) if robust else 'n/a'):>12}"
             f"{(f'{100 * clean / len(rows):.1f}%' if rows and robust else 'n/a'):>8}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 images per cell, no detector")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    count, chunk = (8, 8) if args.smoke else (COUNT, args.chunk)
    root = paths.ensure_results("exp41_second_qf_smoke" if args.smoke else "exp41_second_qf")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    note(f"START second quality factor: QF{QF}, {count} images, {len(cells())} cells")
    covers = prepare(count, note)
    if not (root / "writer_ok").exists():
        if not check_writer(root, note):
            note("STOP - covers written by PIL differ from MATLAB's at this quality factor")
            return 1
        (root / "writer_ok").write_text("ok")

    # The shards take both quality factors from these two module values.
    e13.COVER_QF, e13.ARM_A_QF = QF, CHANNEL_UP
    e21.embed(root, cells(), covers, chunk, args.workers, note)
    rob = robustness(root, note)
    (root / "robustness.json").write_text(json.dumps(rob, indent=2), encoding="utf-8")
    if args.smoke:
        note("END (smoke)")
        return 0

    fdir = root / "features"
    ds.extract(fdir, "cover", [(c.stem, c) for c in covers], QF, note)
    for c in cells():
        ok = e21.ok_set(root, c["name"])
        ds.extract(fdir, c["name"], [(str(i), root / c["name"] / "stego" / f"{i}.jpg")
                                     for i in range(1, count + 1) if str(i) in ok], QF, note)
    Xcov, cov_stems = ds.load(fdir, "cover")
    null = ds.null_test(root / "dctr", Xcov, cov_stems, note)
    res = ds.score(root / "dctr", Xcov, cov_stems,
                   [(c["name"], dict(kind=c["kind"], payload_bpnzac=c["payload"], cover_qf=QF,
                                     n_images=count, null_p_e=null["p_e_mean"]),
                     lambda c=c: ds.load(fdir, c["name"])) for c in cells()], note)

    q65 = {}
    for folder in QF65:
        f = paths.RESULTS / folder / "tier_5000" / "results.json"
        if f.exists():
            q65.update({r["cell"]: r["p_e_mean"] for r in json.loads(f.read_text())})
    pe = {r["cell"]: r for r in res}
    note("")
    note(f"DCTR + FLD, 5,000 images; null at QF{QF}: {null['p_e_mean']:.4f} +/- {null['p_e_std']:.4f}")
    note(f"{'cell':<16}{'QF 75':>9}{'sd':>8}{'QF 65':>9}{'error-free at QF 75':>22}")
    for c in cells():
        r = pe[c["name"]]
        rr = rob[c["name"]]["rate"]
        note(f"{c['name']:<16}{r['p_e_mean']:>9.4f}{r['p_e_std']:>8.4f}"
             f"{q65.get(c['name'], float('nan')):>9.4f}"
             f"{(f'{100 * rr:.1f}%' if rr is not None else 'n/a'):>22}")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
