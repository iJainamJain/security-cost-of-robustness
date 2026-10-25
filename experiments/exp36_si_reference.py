"""Side-informed reference for SCRIPT: the same embedder without the selection.

F28 shows that SCRIPT's security at strength 0 is the security of side-informed
embedding, so J-UNIWARD (which does not use the uncompressed image) is not its
fair non-robust reference. The fair one is SCRIPT's own embedder with the
robustness step taken out: same UERD costs, same side-information discount,
same coder, but every coefficient usable.

The released code already provides that. `coverSelect.m` only selects when its
first threshold is >= 0; called with -1 it returns every coefficient as usable.
So this is `script_embed_shard` with R1 = R2 = -1 (and T_max = 0, UERD, as for
all our SCRIPT runs) - no new embedder and no patch. The reference carries the
same zero-cost changes as SCRIPT (F28), so the comparison is like for like:

    security   P_E(reference) - P_E(SCRIPT, R = 0)  = what the selection costs
    delivery   error-free rate on the QF65 -> QF65 channel, reference against
               SCRIPT's 99.8-99.95%                 = what the selection buys

Cells: 0.05, 0.1, 0.2 bpnzac on BOSSbase 512x512 at QF65, the images of
Tables 1-3. Tiered: all cells reach 5,000 images before any goes on to 10,000.
Embedding only; DCTR and SRNet follow separately.

    python experiments/exp36_si_reference.py --pilot     # 8 images per cell
    python experiments/exp36_si_reference.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp13_fullscale as e13  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402  (crash-resuming embed, R call)

COVERS = paths.DATA / f"BOSSbase_QF{e13.COVER_QF}_full"
TIERS = (5000, 10000)


def cells() -> list[dict]:
    return [dict(name=f"SI-UERD_{p}", kind="script", payload=p, R=-1) for p in (0.05, 0.1, 0.2)]


def summary(root: pathlib.Path, note) -> None:
    note(f"{'cell':<16}{'images':>8}{'embedded':>10}{'error-free':>12}{'mean BER':>10}{'s/image':>9}")
    for c in cells():
        rows = [r for f in sorted((root / c["name"] / "csv").glob("chunk_*.csv"))
                for r in e13.csv_rows(f)]
        ok = [r for r in rows if e13.row_ok(r)]
        ber = [float(r["bit_error_rate"]) for r in ok if (r.get("bit_error_rate") or "") != ""]
        secs = [float(r["seconds"]) for r in rows if (r.get("seconds") or "").strip()]
        note(f"{c['name']:<16}{len(rows):>8}{len(ok):>10}{sum(1 for b in ber if b == 0):>12}"
             f"{(sum(ber) / len(ber) if ber else float('nan')):>10.4f}"
             f"{(sum(secs) / len(secs) if secs else float('nan')):>9.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", action="store_true", help="8 images per cell, separate folder")
    args = ap.parse_args()
    tiers = (8,) if args.pilot else TIERS
    if args.pilot:
        args.chunk = 8
    root = paths.ensure_results("exp36_si_reference_pilot" if args.pilot
                                else "exp36_si_reference")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    note(f"START side-informed reference: {len(cells())} cells, tiers {tiers}, "
         f"{args.workers} workers")
    for n in tiers:
        covers = [COVERS / f"{i}.jpg" for i in range(1, n + 1)]
        e21.embed(root, cells(), covers, args.chunk, args.workers, note)
        note(f"--- tier {n}")
        summary(root, note)
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
