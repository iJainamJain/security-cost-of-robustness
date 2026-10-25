"""Pack a small stand-in for the 5,000-image grid dataset: covers plus GMAS,
ROAST and MINICER at 0.05 and 0.1 only.

On 2026-10-06 Kaggle could not mount the full dataset (stego-5000-bin) from two
accounts, and the last SRNet sessions need only these six cells. This writes
the same files the full pack holds for them - it reads the same exp13 output
with the same "embed succeeded" filter (exp14_pack_tier.ok_stems) - at about a
third of the size, so it uploads quickly.

    python experiments/exp27_pack_robust_subset.py
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp14_pack_tier as e14  # noqa: E402
from exp13_fullscale import COVER_QF, cells  # noqa: E402

COUNT = 5000
KEEP = ("GMAS_0.05", "GMAS_0.1", "ROAST_0.05", "ROAST_0.1", "MINICER_0.05", "MINICER_0.1")


def main() -> int:
    out = paths.ensure_results("exp27_robust_subset_5000")
    tmp, dest = out / "stego_5000_robust.zip", out / "stego_5000_robust.bin"
    manifest = dict(
        description=f"Subset of the 5,000-image grid: covers plus {', '.join(KEEP)}.",
        cover_dir="cover", cover_count=COUNT,
        image_spec=dict(format="JPEG", size="512x512", channels=1,
                        cover_quality_factor=COVER_QF),
        pairing="stego/<cell>/N.jpg is the same source image as cover/N.jpg.",
        source="experiments/exp13_fullscale.py, seeds = global image index", cells=[])
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for i in range(1, COUNT + 1):
            z.write(e14.COVERS / f"{i}.jpg", f"dataset/cover/{i}.jpg")
        for c in cells():
            if c["name"] not in KEEP:
                continue
            cdir = e14.SRC / c["name"]
            stems, seen = e14.ok_stems(cdir, COUNT)
            if seen < COUNT:
                raise SystemExit(f"{c['name']}: only {seen} of {COUNT} rows present")
            n = 0
            for s in stems:
                f = cdir / "stego" / f"{s}.jpg"
                if f.exists() and f.stat().st_size > 0:
                    z.write(f, f"dataset/stego/{c['name']}/{s}.jpg"); n += 1
            arm, method, channel = e14.META[c["kind"]]
            manifest["cells"].append(dict(cell=c["name"], arm=arm, method=method,
                                          payload_bpnzac=c["payload"], channel=channel,
                                          stego_dir=f"stego/{c['name']}", count=n))
            print(f"{c['name']}: {n} of {COUNT} images")
        z.writestr("dataset/MANIFEST.json", json.dumps(manifest, indent=2))
    with dest.open("wb") as fo, tmp.open("rb") as fi:
        # A header and a non-archive extension keep Kaggle from unpacking it
        # into thousands of loose files (which stalls sessions for hours).
        fo.write(b"STEGO-EVAL packed dataset; open with zipfile.".ljust(64, b" "))
        shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)
    tmp.unlink()
    print(f"wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
