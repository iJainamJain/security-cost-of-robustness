"""How much did a deviation in our SCRIPT harness change SCRIPT's results?

Found by the review council on 2026-10-11 and confirmed (exp47): our
`script_embed_shard.m` built the spatial pattern of a unit coefficient change
with `dct2`, as the MINICER and PMAS drivers of the repository do. SCRIPT's own
driver (`testRobustGISS.m`, line 16) uses `idct2`, which is also the physically
right pattern. The patterns enter `coverSelect`'s pixel-headroom test, and at
strength 0 that test is the only one that excludes anything, so every SCRIPT
selection we have run differs from the released one in 0.4-5% of the
coefficients of an image.

Pilot, first 2,000 images, 0.05 and 0.2 bpnzac, with the released pattern:

    SCRIPT-rel_{p}      released cost
    SCRIPT-relfix3_{p}  cost repair 3 (floor 0.1)

Measured: delivery on QF65 -> QF65, both as the driver measures it (extraction
with the sender's coefficient set) and blind (the receiver recomputes the set
from the received image); DCTR + FLD, 1,000 / 1,000 pairs, against the values
of the old harness on the same images (exp34: 0.397 / 0.378; exp40 pilot:
0.489 / 0.459).

    python experiments/exp48_script_released_patterns.py --smoke
    python experiments/exp48_script_released_patterns.py
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
import exp16_dctr_fullscale as e16  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402
import exp40_script_fix as e40  # noqa: E402

N, PAYLOADS = 2000, (0.05, 0.2)
OLD = {("rel", 0.05): 0.3970, ("rel", 0.2): 0.3781, ("relfix3", 0.05): 0.4886, ("relfix3", 0.2): 0.4587}

_prev = e13.matlab_call


def _call(cell, manifest, out_dir, csv_path, seed_base):
    if "sp" not in cell:
        return _prev(cell, manifest, out_dir, csv_path, seed_base)
    r, q = cell["R"], e13.COVER_QF
    return (f"script_embed_shard('{e13.REPO.as_posix()}','{manifest.as_posix()}',"
            f"'{out_dir.as_posix()}','{csv_path.as_posix()}',{cell['payload']},{q},{q},"
            f"{seed_base},{r},{r},0,0,{cell['fix']},{cell['sp']},1)")


e13.matlab_call = _call


def cells() -> list[dict]:
    return [dict(name=f"SCRIPT-{tag}_{p}", kind="script", payload=p, R=0, fix=fix, sp=1, tag=tag)
            for p in PAYLOADS for tag, fix in (("rel", 0), ("relfix3", 3))]


def blind(root: pathlib.Path, name: str, n: int) -> dict:
    rows = [r for f in sorted((root / name / "csv").glob("chunk_*.csv"))
            for r in e13.csv_rows(f) if (r.get("cover") or "").strip().isdigit()
            and int(r["cover"]) <= n and e13.row_ok(r)]
    b = [r for r in rows if (r.get("blind_ber") or "") != ""]
    return dict(images=len(b), blind_error_free=sum(float(r["blind_ber"]) == 0 for r in b),
                same_set=sum(int(float(r["blind_set_differs"])) == 0 for r in b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 images per cell, no detector")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = paths.ensure_results("exp48_smoke" if args.smoke else "exp48_script_released_patterns")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    n, chunk = (8, 8) if args.smoke else (N, args.chunk)
    cs = cells()
    note(f"START SCRIPT with the released spatial patterns, {n} images, payloads {PAYLOADS}")
    e21.embed(root, cs, [e40.COVERS / f"{i}.jpg" for i in range(1, n + 1)], chunk, args.workers, note)
    note(f"{'cell':<22}{'images':>8}{'embedded':>10}{'error-free':>12}{'blind error-free':>18}"
         f"{'same set at receiver':>22}{'PSNR':>8}")
    deliv = {}
    for c in cs:
        d = deliv[c["name"]] = e40.delivery(root, c["name"], n)
        b = blind(root, c["name"], n)
        d.update(b)
        note(f"{c['name']:<22}{d['images']:>8}{d['embedded']:>10}{d['error_free']:>12}"
             f"{b['blind_error_free']:>18}{b['same_set']:>22}{(d['median_psnr_db'] or 0):>8.1f}")
    if args.smoke:
        note("END (smoke)")
        return 0

    fdir = root / "features"
    Xcov, cov_stems = e16.load_set(e40.E16_OUT, "cover", N, ds.FCHUNK)
    for c in cs:
        ds.extract(fdir, c["name"], e40.stego_files(root, c["name"], N), e13.COVER_QF, note)
    res = ds.score(root / "pilot_2000", Xcov, cov_stems,
                   [(c["name"], dict(payload_bpnzac=c["payload"], n_images=N),
                     lambda c=c: ds.load(fdir, c["name"], N // ds.FCHUNK)) for c in cs], note)
    pe = {r["cell"]: r for r in res}
    note("")
    note("2,000 images, DCTR + FLD: released pattern (idct2) against the old harness (dct2)")
    note(f"  {'cell':<22}{'new P_E':>9}{'sd':>8}{'old P_E':>9}{'change':>9}{'delivered':>11}"
         f"{'blind':>8}")
    for c in cs:
        r, d = pe[c["name"]], deliv[c["name"]]
        old = OLD[(c["tag"], c["payload"])]
        note(f"  {c['name']:<22}{r['p_e_mean']:>9.4f}{r['p_e_std']:>8.4f}{old:>9.4f}"
             f"{r['p_e_mean'] - old:>9.4f}{100 * d['rate']:>10.2f}%"
             f"{100 * d['blind_error_free'] / max(d['images'], 1):>7.2f}%")
    (root / "summary.json").write_text(json.dumps(dict(delivery=deliv, p_e={k: v["p_e_mean"] for k, v in pe.items()},
                                                       old=[dict(tag=k[0], payload=k[1], p_e=v) for k, v in OLD.items()]),
                                                  indent=2), encoding="utf-8")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
