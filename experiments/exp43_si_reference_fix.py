"""Side-informed reference for the repaired SCRIPT: is the selection still free?

F29 (Table 2c): with the released cost, SCRIPT at strength 0 is no easier to
detect than its own embedder with the coefficient selection switched off, and
the selection takes delivery from 27% to 99.9%. Both sides of that comparison
carried the zero-cost changes of F28, which set most of what the detectors saw.

F32: a floor of 0.1 on the discount factor removes that artefact. The repaired
SCRIPT is then compared with J-UNIWARD only, which is not like for like - SCRIPT
has the uncompressed image. This runs the like-for-like reference: the same
embedder, the same floor, every coefficient usable (R1 = R2 = -1, as in exp36).

    security   P_E(reference, floor) - P_E(SCRIPT, floor)  = what the selection
               costs once the artefact is gone
    delivery   error-free rate on QF65 -> QF65              = what it buys

First 5,000 images, DCTR + FLD as in Table 3. Tier 1 is 0.2, 0.4 and 0.6
bpnzac, where the repaired SCRIPT is detectable (0.459 / 0.380 / 0.278) and the
comparison has something to measure; tier 2 adds 0.05 and 0.1, where it is at
the null and only delivery is informative.

    python experiments/exp43_si_reference_fix.py --smoke
    python experiments/exp43_si_reference_fix.py
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
import exp40_script_fix as e40  # noqa: E402  (installs the si_fix call)

TIERS = ((0.2, 0.4, 0.6), (0.05, 0.1))
N = 5000
E40_ROOT = paths.RESULTS / "exp40_script_fix"


def cell(v: int, p: float) -> dict:
    return dict(name=f"SI-fix{v}_{p}", kind="script", payload=p, R=-1, fix=v)


def script_pe(best: int) -> dict:
    """P_E of the repaired SCRIPT on the same 5,000 images (exp40, exp42)."""
    pe = {}
    for sub in ("tier_5000", "tier_5000_high"):
        f = E40_ROOT / sub / "results.json"
        if f.exists():
            for r in json.loads(f.read_text()):
                pe[r["cell"]] = r["p_e_mean"]
    return pe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 images per cell, no detector")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    best = json.loads((E40_ROOT / "decision.json").read_text())["chosen"]
    root = paths.ensure_results("exp43_smoke" if args.smoke else "exp43_si_reference_fix")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    n, chunk = (8, 8) if args.smoke else (N, args.chunk)
    covers = [e40.COVERS / f"{i}.jpg" for i in range(1, n + 1)]
    note(f"START side-informed reference with cost repair {best} "
         f"({e40.VARIANTS[best]}), {n} images, tiers {TIERS}")
    spe = script_pe(best)
    for t, payloads in enumerate(TIERS, 1):
        cells = [cell(best, p) for p in payloads]
        e21.embed(root, cells, covers, chunk, args.workers, note)
        note(f"--- tier {t}")
        note(f"{'cell':<16}{'images':>8}{'embedded':>10}{'error-free':>12}{'PSNR':>8}")
        deliv = {}
        for c in cells:
            d = deliv[c["name"]] = e40.delivery(root, c["name"], n)
            note(f"{c['name']:<16}{d['images']:>8}{d['embedded']:>10}{d['error_free']:>12}"
                 f"{(d['median_psnr_db'] or 0):>8.1f}")
        if args.smoke:
            continue
        fdir = root / "features"
        Xcov, cov_stems = e16.load_set(e40.E16_OUT, "cover", N, ds.FCHUNK)
        for c in cells:
            ds.extract(fdir, c["name"], e40.stego_files(root, c["name"], N), e13.COVER_QF, note)
        res = ds.score(root / f"tier{t}_5000", Xcov, cov_stems,
                       [(c["name"], dict(variant=best, payload_bpnzac=c["payload"], n_images=N),
                         lambda c=c: ds.load(fdir, c["name"], N // ds.FCHUNK)) for c in cells],
                       note)
        pe = {r["cell"]: r for r in res}
        note("")
        note("5,000 images, DCTR + FLD (null 0.484), cost repair on both sides")
        note(f"  {'payload':<10}{'no selection':>14}{'sd':>8}{'SCRIPT':>10}{'difference':>12}"
             f"{'delivered, no selection':>26}{'SCRIPT':>10}")
        for c in cells:
            a = pe[c["name"]]
            b = spe.get(f"SCRIPT-fix{best}_{c['payload']}", float("nan"))
            ds_ = e40.delivery(E40_ROOT, f"SCRIPT-fix{best}_{c['payload']}", N)
            note(f"  {c['payload']:<10}{a['p_e_mean']:>14.4f}{a['p_e_std']:>8.4f}{b:>10.4f}"
                 f"{a['p_e_mean'] - b:>12.4f}{100 * deliv[c['name']]['rate']:>25.2f}%"
                 f"{100 * ds_['rate']:>9.2f}%")
    note("END (smoke)" if args.smoke else "END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
