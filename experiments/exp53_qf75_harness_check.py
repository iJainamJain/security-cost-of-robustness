"""Does the harness deviation of F36 matter at QF 75?

exp48: at QF 65 the released spatial pattern (idct2) and ours (dct2) give the
same P_E to 0.002 and the same delivery on 2,000 images. At QF 75 we reported
that SCRIPT at strength 0 delivers only 96-98% of images (Table 5b). The
coefficients our harness used and the release excludes are exactly those where
a change can push a pixel out of range, so that loss may be ours.

First 2,000 images at QF 75, SCRIPT at strength 0, released cost, 0.05 / 0.1 /
0.2 bpnzac, channel QF75 -> QF75, with the released pattern; against the old
harness on the same images (stego and features of exp41): delivery, and DCTR +
FLD with 1,000 / 1,000 pairs.

    python experiments/exp53_qf75_harness_check.py --smoke
    python experiments/exp53_qf75_harness_check.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dctr_sets as ds  # noqa: E402
import exp13_fullscale as e13  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402
import exp40_script_fix as e40  # noqa: E402

QF, N, PAYLOADS = 75, 2000, (0.05, 0.1, 0.2)
COVERS = paths.DATA / f"BOSSbase_QF{QF}_full"
E41 = paths.RESULTS / "exp41_second_qf"

_prev = e13.matlab_call


def _call(cell, manifest, out_dir, csv_path, seed_base):
    if "qf75" not in cell:
        return _prev(cell, manifest, out_dir, csv_path, seed_base)
    return (f"script_embed_shard('{e13.REPO.as_posix()}','{manifest.as_posix()}',"
            f"'{out_dir.as_posix()}','{csv_path.as_posix()}',{cell['payload']},{QF},{QF},"
            f"{seed_base},0,0,0,0,0,1,0)")


e13.matlab_call = _call


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 images per cell, no detector")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = paths.ensure_results("exp53_smoke" if args.smoke else "exp53_qf75_harness_check")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    n, chunk = (8, 8) if args.smoke else (N, args.chunk)
    cs = [dict(name=f"SCRIPT75-rel_{p}", kind="script", payload=p, qf75=1) for p in PAYLOADS]
    note(f"START QF {QF} harness check, {n} images, payloads {PAYLOADS}")
    e21.embed(root, cs, [COVERS / f"{i}.jpg" for i in range(1, n + 1)], chunk, args.workers, note)
    note(f"{'payload':<9}{'released pattern: error-free':>30}{'old harness: error-free':>26}  of {n}")
    deliv = {}
    for c in cs:
        a = e40.delivery(root, c["name"], n)
        b = e40.delivery(E41, f"SCRIPT-R0_{c['payload']}", n)
        deliv[c["payload"]] = dict(new=a, old=b)
        note(f"{c['payload']:<9}{a['error_free']:>22} ({100 * a['rate']:.2f}%)"
             f"{b['error_free']:>18} ({100 * b['rate']:.2f}%)")
    if args.smoke:
        note("END (smoke)")
        return 0

    fdir = root / "features"
    k = N // ds.FCHUNK
    Xcov, cov_stems = ds.load(E41 / "features", "cover", k)
    for c in cs:
        ds.extract(fdir, c["name"], e40.stego_files(root, c["name"], N), QF, note)
    sets = [(c["name"], dict(payload_bpnzac=c["payload"], harness="released pattern"),
             lambda c=c: ds.load(fdir, c["name"], k)) for c in cs]
    sets += [(f"old_SCRIPT-R0_{p}", dict(payload_bpnzac=p, harness="old"),
              lambda p=p: ds.load(E41 / "features", f"SCRIPT-R0_{p}", k)) for p in PAYLOADS]
    res = {r["cell"]: r for r in ds.score(root / "dctr_2000", Xcov, cov_stems, sets, note)}
    note("")
    note(f"QF {QF}, first {N} images, DCTR + FLD (1,000 / 1,000 pairs)")
    note(f"  {'payload':<9}{'new P_E':>9}{'sd':>8}{'old P_E':>9}{'sd':>8}{'change':>9}"
         f"{'delivered new':>15}{'old':>9}")
    for p in PAYLOADS:
        a, b = res[f"SCRIPT75-rel_{p}"], res[f"old_SCRIPT-R0_{p}"]
        note(f"  {p:<9}{a['p_e_mean']:>9.4f}{a['p_e_std']:>8.4f}{b['p_e_mean']:>9.4f}{b['p_e_std']:>8.4f}"
             f"{a['p_e_mean'] - b['p_e_mean']:>9.4f}{100 * deliv[p]['new']['rate']:>14.2f}%"
             f"{100 * deliv[p]['old']['rate']:>8.2f}%")
    (root / "summary.json").write_text(json.dumps(dict(
        delivery={str(p): v for p, v in deliv.items()},
        p_e={k2: v["p_e_mean"] for k2, v in res.items()}), indent=2), encoding="utf-8")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
