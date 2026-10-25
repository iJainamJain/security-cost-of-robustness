"""Second image source (journal roadmap WP5): does the picture change on other images?

The full-size (512x512) BOWS2 originals are no longer downloadable; what is
available is the 256x256 BOSSbase + BOWS2 set used throughout deep-learning
steganalysis (`data/BOSSBASE+BOWS2.zip`, the GBRAS-Net release). So this runs a
MATCHED PAIR at 256x256:

    BOSSbase-256   the same photographs as our main experiments, downscaled
    BOWS2-256      a different source, same size, same processing

Two sets that differ only in where the photographs came from isolate the effect
of the source; BOSSbase-256 against the 512x512 results shows the effect of
image size. First 5,000 images of each (the same 5,000 BOSSbase images as the
main 5,000-image tier).

Cells: the 12 of the 5,000-image CNN grid, with SCRIPT at strength R = 0.

Stages (each resumable):
  prepare   unpack the PGMs, compress to QF65 with PIL (bit-identical to
            MATLAB's imwrite on BOSSbase - prep_covers.py)
  embed     MATLAB, chunks of 250, crash-resume as in exp21

    python experiments/exp24_second_source.py --pilot      # 8 images per cell
    python experiments/exp24_second_source.py --wait PID   # start when PID exits
    python experiments/exp24_second_source.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp13_fullscale as e13  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402  (crash-resuming embed, R=0 call)

ARCHIVE = paths.DATA / "BOSSBASE+BOWS2.zip"
SOURCES = {"BOSSbase256": "GBRASNET/BOSSbase-1.01/cover",
           "BOWS2_256": "GBRASNET/BOWS2/cover"}
COUNT = 5000


def cells() -> list[dict]:
    out = [dict(name=f"J-UNIWARD_{p}", kind="juniward", payload=p)
           for p in (0.05, 0.1, 0.2, 0.3)]
    for p in (0.05, 0.1):
        out += [dict(name=f"GMAS_{p}", kind="gmas", payload=p),
                dict(name=f"ROAST_{p}", kind="roast", payload=p),
                dict(name=f"SCRIPT-R0_{p}", kind="script", payload=p, R=0),
                dict(name=f"MINICER_{p}", kind="minicer", payload=p)]
    return out


def pgm_dir(src: str) -> pathlib.Path:
    return paths.DATA / f"{src}_pgm"


def cover_dir(src: str) -> pathlib.Path:
    return paths.DATA / f"{src}_QF{e13.COVER_QF}"


def prepare(src: str, count: int, note) -> list[pathlib.Path]:
    from PIL import Image
    pd, cd = pgm_dir(src), cover_dir(src)
    pd.mkdir(parents=True, exist_ok=True)
    cd.mkdir(parents=True, exist_ok=True)
    need = [i for i in range(1, count + 1) if not (pd / f"{i}.pgm").exists()]
    if need:
        with zipfile.ZipFile(ARCHIVE) as z:
            for i in need:
                (pd / f"{i}.pgm").write_bytes(z.read(f"{SOURCES[src]}/{i}.pgm"))
        note(f"{src}: unpacked {len(need)} PGMs")
    made = 0
    for i in range(1, count + 1):
        dst = cd / f"{i}.jpg"
        if not dst.exists():
            im = Image.open(pd / f"{i}.pgm")
            if im.size != (256, 256) or im.mode != "L":
                raise SystemExit(f"{src} {i}.pgm is {im.size} {im.mode}, expected 256x256 L")
            im.save(dst, quality=e13.COVER_QF)
            made += 1
    if made:
        note(f"{src}: wrote {made} covers at QF{e13.COVER_QF}")
    return [cd / f"{i}.jpg" for i in range(1, count + 1)]


def summary(root: pathlib.Path, note) -> None:
    note(f"{'cell':<16}{'images':>8}{'embedded':>10}{'error-free':>12}{'s/image':>9}")
    for c in cells():
        rows = [r for f in sorted((root / c["name"] / "csv").glob("chunk_*.csv"))
                for r in e13.csv_rows(f)]
        ok = sum(1 for r in rows if e13.row_ok(r))
        ber = [r.get("bit_error_rate") for r in rows if e13.row_ok(r)]
        errless = sum(1 for b in ber if b not in (None, "") and float(b) == 0)
        secs = [float(r["seconds"]) for r in rows if (r.get("seconds") or "").strip()]
        robust = c["kind"] != "juniward"
        note(f"{c['name']:<16}{len(rows):>8}{ok:>10}"
             f"{(str(errless) if robust else 'n/a'):>12}"
             f"{(sum(secs) / len(secs) if secs else float('nan')):>9.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=COUNT)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", action="store_true",
                    help="8 images per cell per source, separate folder")
    ap.add_argument("--wait", type=int, default=None,
                    help="wait for this process id to exit before starting")
    args = ap.parse_args()
    if args.pilot:
        args.count, args.chunk = 8, 8
    base = paths.ensure_results("exp24_second_source_pilot" if args.pilot
                                else "exp24_second_source")
    runlog = (base / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    if args.wait:
        note(f"waiting for process {args.wait} to exit...")
        while str(args.wait) in subprocess.run(
                ["tasklist", "/FI", f"PID eq {args.wait}", "/NH"],
                capture_output=True, text=True).stdout:
            time.sleep(60)
    note(f"START second source: {args.count} images x {len(cells())} cells x "
         f"{len(SOURCES)} sources, {args.workers} workers")
    for src in SOURCES:
        covers = prepare(src, args.count, note)
        # SCRIPT's manifest pairs each cover with its uncompressed original.
        e13.PRECOVERS = pgm_dir(src)
        root = base / src
        root.mkdir(parents=True, exist_ok=True)
        e21.embed(root, cells(), covers, args.chunk, args.workers, note)
        note(f"--- {src}")
        summary(root, note)
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
