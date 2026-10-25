"""Generate JPEG covers for TIMING benchmarks only.

The bundled GMAS set is 10 images — far too few to measure parallel scaling,
because MATLAB's ~10 s startup dominates a run that short. This cuts a larger
set of 512x512 QF-65 covers from real video frames so the sharding benchmark has
something to chew on.

    python experiments/make_timing_covers.py --count 48

**These are for timing only.** They are consecutive-ish frames from one clip, so
they are correlated and are not a valid cover source for any steganalysis
result. Use BOSSbase for anything that goes in the paper — see data/README.md.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

SOURCE_VIDEO = pathlib.Path(r"D:\Stego_tool.project\main 2.mp4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=48)
    ap.add_argument("--stride", type=int, default=17,
                    help="frames to skip between crops, to decorrelate a little")
    ap.add_argument("--qf", type=int, default=65)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    import av
    from PIL import Image

    out = paths.DATA / f"timing_covers_QF{args.qf}"
    out.mkdir(parents=True, exist_ok=True)

    if not SOURCE_VIDEO.exists():
        print(f"source video not found: {SOURCE_VIDEO}")
        return 1

    container = av.open(str(SOURCE_VIDEO))
    stream = container.streams.video[0]

    written = 0
    for n, frame in enumerate(container.decode(stream)):
        if n % args.stride:
            continue
        img = frame.to_ndarray(format="rgb24")
        h, w, _ = img.shape
        if h < args.size or w < args.size:
            continue
        # Walk the crop window across the frame so successive covers differ.
        y0 = ((written * 53) % max(1, h - args.size))
        x0 = ((written * 97) % max(1, w - args.size))
        crop = img[y0:y0 + args.size, x0:x0 + args.size]

        written += 1
        # GMAS reads a single (luminance) coefficient array, so save grayscale.
        Image.fromarray(crop).convert("L").save(
            out / f"{written}.jpg", quality=args.qf, subsampling=0)
        if written >= args.count:
            break

    print(f"wrote {written} covers to {out}")
    print("TIMING USE ONLY - correlated frames, not a valid steganalysis cover source")
    return 0 if written == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
