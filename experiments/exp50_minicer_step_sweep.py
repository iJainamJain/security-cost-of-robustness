"""MINICER: security and robustness against its QIM step.

The released driver hardcodes a step of 1 for every DCT position. With it the
method barely survived our channel, so the main tables use a step of 6 found by
a search on robustness alone (exp10). SCRIPT taught us what that can do (F24):
a strength picked for robustness cost most of the method's security. The review
council asked for the same sweep here, and pointed out that Butora et al. [33]
describe MINICER without any quantisation-index modulation, i.e. as step 1.

Steps 1, 2, 3, 4 (new) and 6 (the main tables), first 2,000 images, 0.05 and
0.1 bpnzac, channel QF65 -> QF65: images delivered without error, raw bit error
rate, PSNR, and DCTR + FLD (1,000 / 1,000 pairs, five splits).

    python experiments/exp50_minicer_step_sweep.py --smoke
    python experiments/exp50_minicer_step_sweep.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dctr_sets as ds  # noqa: E402
import exp13_fullscale as e13  # noqa: E402
import exp16_dctr_fullscale as e16  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402
import exp40_script_fix as e40  # noqa: E402

N, PAYLOADS, STEPS = 2000, (0.05, 0.1), (1, 2, 3, 4)
E13_OUT = paths.RESULTS / "exp13_fullscale"

_prev = e13.matlab_call


def _call(cell, manifest, out_dir, csv_path, seed_base):
    if "step" not in cell:
        return _prev(cell, manifest, out_dir, csv_path, seed_base)
    q = e13.COVER_QF
    return (f"minicer_embed_shard('{e13.REPO.as_posix()}','{manifest.as_posix()}',"
            f"'{out_dir.as_posix()}','{csv_path.as_posix()}',{cell['payload']},{q},{q},"
            f"{seed_base},{cell['step']})")


e13.matlab_call = _call


def stats(root: pathlib.Path, name: str, n: int) -> dict:
    rows = [r for f in sorted((root / name / "csv").glob("chunk_*.csv"))
            for r in e13.csv_rows(f) if (r.get("cover") or "").strip().isdigit()
            and int(r["cover"]) <= n]
    ok = [r for r in rows if e13.row_ok(r) and (r.get("bit_error_rate") or "") != ""]
    ber = [float(r["bit_error_rate"]) for r in ok]
    psnr = sorted(float(r["psnr_db"]) for r in ok if (r.get("psnr_db") or "") not in ("", "Inf"))
    return dict(images=len(rows), embedded=len(ok), error_free=sum(b == 0 for b in ber),
                mean_ber=st.fmean(ber) if ber else float("nan"),
                median_psnr=psnr[len(psnr) // 2] if psnr else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 images per cell, no detector")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = paths.ensure_results("exp50_smoke" if args.smoke else "exp50_minicer_step_sweep")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    n, chunk = (8, 8) if args.smoke else (N, args.chunk)
    cs = [dict(name=f"MINICER-s{s}_{p}", kind="minicer", payload=p, step=s)
          for p in PAYLOADS for s in STEPS]
    note(f"START MINICER step sweep {STEPS} (+6 from the main tables), {n} images")
    e21.embed(root, cs, [e40.COVERS / f"{i}.jpg" for i in range(1, n + 1)], chunk, args.workers, note)
    table = {c["name"]: dict(step=c["step"], payload=c["payload"], **stats(root, c["name"], n)) for c in cs}
    for p in PAYLOADS:
        table[f"MINICER_{p}"] = dict(step=6, payload=p, **stats(E13_OUT, f"MINICER_{p}", n))
    note(f"{'cell':<18}{'step':>5}{'images':>8}{'error-free':>12}{'mean BER':>10}{'PSNR':>8}")
    for name, t in table.items():
        note(f"{name:<18}{t['step']:>5}{t['images']:>8}{t['error_free']:>12}{t['mean_ber']:>10.4f}"
             f"{t['median_psnr']:>8.1f}")
    if args.smoke:
        note("END (smoke)")
        return 0

    fdir = root / "features"
    Xcov, cov_stems = e16.load_set(e40.E16_OUT, "cover", N, ds.FCHUNK)
    for c in cs:
        ds.extract(fdir, c["name"], e40.stego_files(root, c["name"], N), e13.COVER_QF, note)
    sets = [(c["name"], dict(step=c["step"], payload_bpnzac=c["payload"], n_images=N),
             lambda c=c: ds.load(fdir, c["name"], N // ds.FCHUNK)) for c in cs]
    sets += [(f"MINICER_{p}", dict(step=6, payload_bpnzac=p, n_images=N),
              lambda p=p: e16.load_set(e40.E16_OUT, f"MINICER_{p}", N, ds.FCHUNK)) for p in PAYLOADS]
    res = {r["cell"]: r for r in ds.score(root / "sweep_2000", Xcov, cov_stems, sets, note)}
    note("")
    note("MINICER against its step, 2,000 images, DCTR + FLD (unpaired null 0.473)")
    note(f"{'payload':<9}{'step':>5}{'P_E':>9}{'sd':>8}{'error-free':>12}{'mean BER':>10}{'PSNR':>8}")
    for p in PAYLOADS:
        for name, t in sorted(table.items(), key=lambda kv: kv[1]["step"]):
            if t["payload"] != p:
                continue
            r = res[name]
            note(f"{p:<9}{t['step']:>5}{r['p_e_mean']:>9.4f}{r['p_e_std']:>8.4f}"
                 f"{100 * t['error_free'] / max(t['images'], 1):>11.1f}%{t['mean_ber']:>10.4f}"
                 f"{t['median_psnr']:>8.1f}")
    (root / "summary.json").write_text(json.dumps(dict(
        table=table, p_e={k: v["p_e_mean"] for k, v in res.items()}), indent=2), encoding="utf-8")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
