"""Package one tier of the full-scale run as a Kaggle dataset zip.

Reads straight from exp13's output and writes a single stored (uncompressed)
zip - JPEGs do not compress, and skipping an intermediate copy keeps the disk
quiet while the embedding run is still going.

Only images whose CSV row says the embed succeeded are shipped. A failed embed
can leave a partial or stale file behind; shipping it would put a non-stego
image in the stego class.

    python experiments/exp14_pack_tier.py --count 2000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from exp13_fullscale import COVER_QF, cells, row_ok  # noqa: E402

SRC = paths.RESULTS / "exp13_fullscale"
COVERS = paths.DATA / f"BOSSbase_QF{COVER_QF}_full"

META = {  # kind -> (arm, method label, channel)
    "juniward": ("A", "J-UNIWARD", "none (non-robust ceiling)"),
    "gmas": ("A", "GMAS", "QF65->QF71"),
    "roast": ("A", "ROAST-recon", "QF65->QF71"),
    "script": ("B", "SCRIPT", "QF65->QF65"),
    "minicer": ("B", "MINICER", "QF65->QF65"),
}


def ok_stems(cell_dir: pathlib.Path, count: int) -> tuple[list[str], int]:
    """Stems 1..count whose embed succeeded, and how many rows were seen."""
    good, seen = [], 0
    for f in sorted((cell_dir / "csv").glob("chunk_*.csv")):
        with f.open(newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                stem = (r.get("cover") or "").strip()
                if not stem.isdigit() or int(stem) > count:
                    continue
                seen += 1
                if row_ok(r):
                    good.append(stem)
    return sorted(set(good), key=int), seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    args = ap.parse_args()

    out_dir = paths.ensure_results(f"exp14_tier{args.count}")
    zpath = out_dir / f"stego_eval_{args.count}.zip"
    manifest = {
        "description": f"Cover/stego sets, first {args.count} BOSSbase images.",
        "cover_dir": "cover", "cover_count": args.count,
        "image_spec": {"format": "JPEG", "size": "512x512", "channels": 1,
                       "cover_quality_factor": COVER_QF},
        "pairing": "stego/<cell>/N.jpg is the same source image as cover/N.jpg.",
        "source": "experiments/exp13_fullscale.py, seeds = global image index",
        "cells": [],
    }

    problems = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        h = hashlib.sha1()
        for i in range(1, args.count + 1):
            f = COVERS / f"{i}.jpg"
            b = f.read_bytes()
            h.update(f.name.encode()); h.update(b)
            z.writestr(f"dataset/cover/{i}.jpg", b)
        manifest["cover_sha1"] = h.hexdigest()[:16]
        print(f"cover: {args.count} images")

        for c in cells():
            cdir = SRC / c["name"]
            stems, seen = ok_stems(cdir, args.count)
            if seen < args.count:
                problems.append(f"{c['name']}: only {seen} of {args.count} rows "
                                f"present - tier not finished")
            h = hashlib.sha1()
            n = 0
            for s in stems:
                f = cdir / "stego" / f"{s}.jpg"
                if not f.exists() or f.stat().st_size == 0:
                    problems.append(f"{c['name']}: {s}.jpg marked ok but missing")
                    continue
                b = f.read_bytes()
                h.update(f.name.encode()); h.update(b)
                z.writestr(f"dataset/stego/{c['name']}/{s}.jpg", b)
                n += 1
            arm, method, channel = META[c["kind"]]
            manifest["cells"].append({
                "cell": c["name"], "arm": arm, "method": method,
                "payload_bpnzac": c["payload"], "channel": channel,
                "stego_dir": f"stego/{c['name']}", "count": n,
                "embed_failures": args.count - n, "sha1": h.hexdigest()[:16]})
            print(f"{c['name']:<17} {n:>5} images"
                  + (f"   ({args.count - n} failed embeds left out)"
                     if n < args.count else ""))
        z.writestr("dataset/MANIFEST.json", json.dumps(manifest, indent=2))

    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(f"\n{zpath}  {zpath.stat().st_size / 1e9:.2f} GB")
    for p in problems:
        print("PROBLEM:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
