"""Pilot for the "J-UNIWARD plus error correction" baseline (F15).

Question: on the QF65 -> QF65 channel, where SCRIPT delivers 99.8% of images
and MINICER 19-57%, how well does ordinary J-UNIWARD do once an error-correcting
code is added - and what does the code cost?

No new stego images and no new detector are needed for the answer:

  * `juniward_channel_shard.m` embeds exactly as exp13 did (same seeds, so the
    same stego files - checked below), recompresses, extracts, and records the
    raw error pattern per image.
  * A Reed-Solomon code RS(31,k) decodes or fails according to the error
    pattern alone, so every k is evaluated from one run.
  * A code of rate k/31 turns a gross payload p into a net payload p*k/31.
    Security is that of plain J-UNIWARD at the gross payload - Tables 1 and 3.

This pilot: the first 300 images, gross payloads 0.05-0.3, about 25 minutes on
four workers. The first 300 BOSSbase images are not a random sample (F9); the
pilot is for choosing the code and deciding whether the full run is worth it.

    python experiments/exp39_juniward_channel_pilot.py
    python experiments/exp39_juniward_channel_pilot.py --count 5000
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp13_fullscale as e13  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402  (crash-resuming embed)

COVERS = paths.DATA / f"BOSSbase_QF{e13.COVER_QF}_full"
PAYLOADS = (0.05, 0.1, 0.2, 0.3)
CODES = (27, 23, 19, 15, 11, 7, 3)          # k of RS(31,k); corrects (31-k)/2 symbols

_prev_call = e13.matlab_call


def _call(cell, manifest, out_dir, csv_path, seed_base):
    if cell["kind"] != "juniward_chan":
        return _prev_call(cell, manifest, out_dir, csv_path, seed_base)
    return (f"juniward_channel_shard('{e13.REPO.as_posix()}','{manifest.as_posix()}',"
            f"'{out_dir.as_posix()}','{csv_path.as_posix()}',{cell['payload']},"
            f"{e13.COVER_QF},{seed_base})")


e13.matlab_call = _call


def cells() -> list[dict]:
    return [dict(name=f"J-UNIWARD-chan_{p}", kind="juniward_chan", payload=p) for p in PAYLOADS]


def md5(p: pathlib.Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=75)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = paths.ensure_results(f"exp39_juniward_channel_{args.count}")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    note(f"START J-UNIWARD through the QF{e13.COVER_QF} channel: {args.count} images, "
         f"gross payloads {PAYLOADS}")
    covers = [COVERS / f"{i}.jpg" for i in range(1, args.count + 1)]
    e21.embed(root, cells(), covers, args.chunk, args.workers, note)

    # The stego files must be the ones exp13 produced, or Tables 1 and 3 do not apply.
    same = total = 0
    for c in cells():
        ref = paths.RESULTS / "exp13_fullscale" / f"J-UNIWARD_{c['payload']}" / "stego"
        for i in range(1, min(args.count, 40) + 1):
            a, b = root / c["name"] / "stego" / f"{i}.jpg", ref / f"{i}.jpg"
            if a.exists() and b.exists():
                total += 1
                same += md5(a) == md5(b)
    note(f"stego files identical to exp13's: {same} of {total} compared")

    out = []
    note("")
    note(f"{'gross payload':<15}{'images':>7}{'unchanged':>10}{'coef changed':>13}{'raw BER':>9}"
         f"{'sym err':>9}   delivered by RS(31,k), k = " + ", ".join(str(k) for k in CODES))
    for c in cells():
        rows = [r for f in sorted((root / c["name"] / "csv").glob("chunk_*.csv"))
                for r in e13.csv_rows(f) if e13.row_ok(r)]
        if not rows:
            continue
        ber = [float(r["bit_error_rate"]) for r in rows]
        worst = [int(float(r["max_block_sym_err"])) for r in rows]
        chg = [int(float(r["coef_changed"])) for r in rows]
        ser = [float(r["sym_err_rate"]) for r in rows]
        n = len(rows)
        deliv = {k: sum(w <= (31 - k) // 2 for w in worst) / n for k in CODES}
        out.append(dict(gross_payload=c["payload"], images=n,
                        error_free_uncoded=sum(b == 0 for b in ber) / n,
                        mean_coef_changed=sum(chg) / n, mean_raw_ber=sum(ber) / n,
                        mean_sym_err_rate=sum(ser) / n,
                        delivered={str(k): v for k, v in deliv.items()},
                        net_payload={str(k): c["payload"] * k / 31 for k in CODES}))
        note(f"{c['payload']:<15}{n:>7}{100 * out[-1]['error_free_uncoded']:>9.1f}%"
             f"{out[-1]['mean_coef_changed']:>13.1f}{out[-1]['mean_raw_ber']:>9.4f}"
             f"{out[-1]['mean_sym_err_rate']:>9.4f}   "
             + "  ".join(f"{100 * deliv[k]:5.1f}%" for k in CODES))
    note("net payload = gross payload x k/31; e.g. k = 15 halves it (rate 0.48)")
    (root / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
