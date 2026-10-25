"""Pack the side-informed reference for the SRNet run on Kaggle: covers plus
SI-UERD at 0.05, 0.1 and 0.2, first 5,000 images (the images of Table 3).

One `.bin` file, as for every other dataset (exp27): a short header in front of
a stored zip, so Kaggle does not unpack it into thousands of loose files.

    python experiments/exp38_pack_si_reference.py
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
import exp13_fullscale as e13  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402
import exp36_si_reference as e36  # noqa: E402

COUNT = 5000
SRC = paths.RESULTS / "exp36_si_reference"


def main() -> int:
    out = paths.ensure_results("exp38_si_reference_5000")
    tmp, dest = out / "stego_SI-UERD_5000.zip", out / "stego_SI-UERD_5000.bin"
    manifest = dict(
        description="Side-informed reference: SCRIPT's embedder with its coefficient "
                    "selection switched off (coverSelect R1 = -1), covers included.",
        cover_dir="cover", cover_count=COUNT,
        image_spec=dict(format="JPEG", size="512x512", channels=1,
                        cover_quality_factor=e13.COVER_QF),
        pairing="stego/<cell>/N.jpg is the same source image as cover/N.jpg.",
        source="experiments/exp36_si_reference.py, seeds = global image index", cells=[])
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for i in range(1, COUNT + 1):
            z.write(e36.COVERS / f"{i}.jpg", f"dataset/cover/{i}.jpg")
        for c in e36.cells():
            ok = e21.ok_set(SRC, c["name"])
            n = 0
            for i in range(1, COUNT + 1):
                f = SRC / c["name"] / "stego" / f"{i}.jpg"
                if str(i) in ok and f.exists() and f.stat().st_size > 0:
                    z.write(f, f"dataset/stego/{c['name']}/{i}.jpg"); n += 1
            if n < 0.95 * COUNT:
                raise SystemExit(f"{c['name']}: only {n} of {COUNT} images - embedding not finished?")
            manifest["cells"].append(dict(cell=c["name"], arm="B",
                                          method="SI-UERD (SCRIPT without selection)",
                                          payload_bpnzac=c["payload"], channel="QF65->QF65",
                                          stego_dir=f"stego/{c['name']}", count=n))
            print(f"{c['name']}: {n} of {COUNT} images")
        z.writestr("dataset/MANIFEST.json", json.dumps(manifest, indent=2))
    with dest.open("wb") as fo, tmp.open("rb") as fi:
        fo.write(b"STEGO-EVAL packed dataset; open with zipfile.".ljust(64, b" "))
        shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)
    tmp.unlink()
    print(f"wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
