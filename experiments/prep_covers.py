"""Convert BOSSbase PGMs into JPEG covers at a given quality factor.

The robust-steganography protocol embeds into JPEG covers, so the spatial-domain
BOSSbase PGMs have to be compressed first. Cheng et al. use Q_cover = 65; match
that for any number compared directly against theirs.

Verified 2026-08-07: PIL and MATLAB `imwrite(...,'quality',65)` produce
**bit-identical** output on BOSSbase — same quantization table and same DCT
coefficient array. So generating covers here rather than in MATLAB introduces no
divergence from the baselines, and is much faster.

    python experiments/prep_covers.py --qf 65 --count 200 --tag timing
    python experiments/prep_covers.py --qf 65                # all 10,000
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

SOURCE = paths.DATA / "BOSSbase_1.01"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qf", type=int, default=65)
    ap.add_argument("--count", type=int, default=None,
                    help="how many covers; default all 10,000")
    ap.add_argument("--tag", type=str, default=None,
                    help="suffix for the output dir, e.g. 'timing'")
    ap.add_argument("--start", type=int, default=1,
                    help="1-based index of the first BOSSbase image to use")
    args = ap.parse_args()

    from PIL import Image

    if not SOURCE.exists():
        print(f"BOSSbase not found at {SOURCE} — see data/README.md")
        return 1

    name = f"BOSSbase_QF{args.qf}" + (f"_{args.tag}" if args.tag else "")
    out = paths.DATA / name
    out.mkdir(parents=True, exist_ok=True)

    # BOSSbase files are 1.pgm .. 10000.pgm. Keep the numbering so a cover can
    # always be traced back to its source image.
    srcs = [SOURCE / f"{i}.pgm"
            for i in range(args.start, args.start + (args.count or 10_000))]
    srcs = [p for p in srcs if p.exists()]
    if not srcs:
        print(f"no source images found from index {args.start}")
        return 1

    t0 = time.time()
    for n, src in enumerate(srcs, 1):
        # BOSSbase is grayscale, so no chroma subsampling question arises.
        Image.open(src).save(out / f"{src.stem}.jpg", quality=args.qf)
        if n % 1000 == 0:
            print(f"  {n}/{len(srcs)}...", flush=True)

    dt = time.time() - t0
    print(f"wrote {len(srcs)} covers at QF{args.qf} to {out}")
    print(f"  {dt:.1f}s ({dt / len(srcs) * 1000:.1f} ms/image)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
