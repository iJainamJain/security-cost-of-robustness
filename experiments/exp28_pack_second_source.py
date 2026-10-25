"""Pack the two 256x256 image sets (exp24) as one-file Kaggle datasets for SRNet.

One .bin per source: 5,000 covers and the 12 cells, only images whose embed
succeeded. Same container as the other packs (64-byte header + stored zip).

    python experiments/exp28_pack_second_source.py
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
import exp24_second_source as e24  # noqa: E402

BASE = paths.RESULTS / "exp24_second_source"
META = {"juniward": ("J-UNIWARD", "none (non-robust reference)"),
        "gmas": ("GMAS (patched embedder)", "QF65->QF71"),
        "roast": ("ROAST-recon (patched embedder)", "QF65->QF71"),
        "script": ("SCRIPT (R=0)", "QF65->QF65"),
        "minicer": ("MINICER", "QF65->QF65")}


def main() -> int:
    out = paths.ensure_results("exp28_second_source_packs")
    for src in e24.SOURCES:
        dest, tmp = out / f"stego_{src}_5000.bin", out / f"stego_{src}_5000.zip"
        if dest.exists():
            print(f"{dest.name} exists, skipping")
            continue
        root, cdir = BASE / src, e24.cover_dir(src)
        manifest = dict(description=f"{src}: first 5,000 images at 256x256, QF65, 12 cells.",
                        cover_dir="cover", cover_count=e24.COUNT, source=src,
                        image_spec=dict(format="JPEG", size="256x256", channels=1,
                                        cover_quality_factor=e13.COVER_QF), cells=[])
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for i in range(1, e24.COUNT + 1):
                z.write(cdir / f"{i}.jpg", f"dataset/cover/{i}.jpg")
            for c in e24.cells():
                good, d, n = e21.ok_set(root, c["name"]), root / c["name"] / "stego", 0
                for i in range(1, e24.COUNT + 1):
                    f = d / f"{i}.jpg"
                    if str(i) in good and f.exists() and f.stat().st_size > 0:
                        z.write(f, f"dataset/stego/{c['name']}/{i}.jpg"); n += 1
                method, channel = META[c["kind"]]
                manifest["cells"].append(dict(cell=c["name"], arm="256", method=method,
                                              payload_bpnzac=c["payload"], channel=channel,
                                              stego_dir=f"stego/{c['name']}", count=n))
            z.writestr("dataset/MANIFEST.json", json.dumps(manifest, indent=2))
        with dest.open("wb") as fo, tmp.open("rb") as fi:
            fo.write(b"STEGO-EVAL packed dataset; open with zipfile.".ljust(64, b" "))
            shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)
        tmp.unlink()
        print(f"wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB): "
              + ", ".join(f"{c['cell']}:{c['count']}" for c in manifest["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
