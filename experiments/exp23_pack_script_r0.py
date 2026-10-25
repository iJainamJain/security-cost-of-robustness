"""Pack SCRIPT at strength R = 0 (0.05, 0.1, 0.2 bpnzac), first 5,000 images,
as a one-file dataset for the CNN on Kaggle.

Paper 1's SCRIPT rows were run at R = 10, a leftover default that makes SCRIPT
far more detectable than it needs to be (FINDINGS F15). The DCTR numbers at
R = 0 come from exp21 phase B; this gives the CNN the same images, so both
detectors report the same SCRIPT setting.

The first 5,000 images are complete as soon as exp21 phase B has embedded
chunks 0-19 of each cell; this does not wait for the rest.

    python experiments/exp23_pack_script_r0.py
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

COUNT = 5000


def main() -> int:
    root = paths.RESULTS / "exp21_script_low_fullscale"
    out = paths.ensure_results("exp23_script_r0_5000")
    dest, tmp = out / "stego_script_r0_5000.bin", out / "stego_script_r0_5000.zip"
    manifest = dict(
        description="SCRIPT at strength R=0 (T_gamma=T_lambda=0), 0.05/0.1/0.2 bpnzac, "
                    "first 5,000 BOSSbase images, same-QF channel.",
        cover_dir="cover", cover_count=COUNT,
        image_spec=dict(format="JPEG", size="512x512", channels=1,
                        cover_quality_factor=e13.COVER_QF), cells=[])
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for i in range(1, COUNT + 1):
            z.write(e21.COVERS / f"{i}.jpg", f"dataset/cover/{i}.jpg")
        for c in e21.CELLS_B:
            name = c["name"]
            rows = [r for k in range(COUNT // 250)
                    for r in e13.csv_rows(root / name / "csv" / f"chunk_{k:03d}.csv")]
            if len(rows) != COUNT:
                raise SystemExit(f"{name}: {len(rows)} rows for the first {COUNT} images - "
                                 "phase B has not finished them")
            good = {(r.get("cover") or "").strip() for r in rows if e13.row_ok(r)}
            d, m = root / name / "stego", 0
            for i in range(1, COUNT + 1):
                f = d / f"{i}.jpg"
                if str(i) in good and f.exists() and f.stat().st_size > 0:
                    z.write(f, f"dataset/stego/{name}/{i}.jpg"); m += 1
            manifest["cells"].append(dict(cell=name, arm="B", method="SCRIPT (R=0)",
                                          payload_bpnzac=c["payload"], channel="QF65->QF65",
                                          stego_dir=f"stego/{name}", count=m))
            print(f"{name}: {m} of {COUNT} images")
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
