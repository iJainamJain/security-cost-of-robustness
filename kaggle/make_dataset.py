"""Build the dataset bundle to hand to the CNN team.

Run this on OUR machine before sending. It ships the exact stego images our
DCTR results were measured on, so the CNN numbers are directly comparable to
ours rather than to a re-run. That removes MATLAB, the baseline
implementations, and every reproduction risk from the receiving team's side:
their job becomes "train a detector on these labelled folders", nothing more.

    python handoff_M4/code/make_dataset.py

Output: handoff_M4/dataset/ (~260 MB). Zip it and upload as a Kaggle Dataset.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent   # .../paper
RESULTS = ROOT / "results"
OUT = ROOT / "handoff_M4" / "dataset"
COVERS = ROOT / "data" / "BOSSbase_QF65_arma"

# (cell name, arm, method, payload, source stego dir, channel description)
CELLS: list[tuple[str, str, str, float, pathlib.Path, str]] = []
for p in ("0.05", "0.1", "0.2", "0.3"):
    CELLS.append((f"J-UNIWARD_{p}", "A", "J-UNIWARD", float(p),
                  RESULTS / "exp08_arm_a" / f"J_UNIWARD_p{p}" / "stego",
                  "none (non-robust ceiling)"))
    CELLS.append((f"GMAS_{p}", "A", "GMAS", float(p),
                  RESULTS / "exp08_arm_a" / f"GMAS_p{p}_q71" / "stego",
                  "QF65->QF71"))
    CELLS.append((f"ROAST_{p}", "A", "ROAST-recon", float(p),
                  RESULTS / "exp08_arm_a" / f"ROAST_recon_p{p}_q71" / "stego",
                  "QF65->QF71"))
for p in ("0.05", "0.1", "0.2"):
    CELLS.append((f"SCRIPT_{p}", "B", "SCRIPT", float(p),
                  RESULTS / "exp10_arm_b" / f"SCRIPT_p{p}_q65" / "stego",
                  "QF65->QF65"))
    CELLS.append((f"MINICER_{p}", "B", "MINICER", float(p),
                  RESULTS / "exp10_arm_b" / f"MINICER_p{p}_q65" / "stego",
                  "QF65->QF65"))
CELLS.append(("StegoTool_0.01", "C", "StegoTool robust.py", 0.01,
              RESULTS / "exp09_stegotool" / "stego", "QF65->QF71"))


def sha1_of_dir(d: pathlib.Path) -> str:
    h = hashlib.sha1()
    for f in sorted(d.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem)):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    if not COVERS.is_dir():
        print(f"cover directory not found: {COVERS}")
        return 1
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "cover").mkdir(parents=True)

    covers = sorted(COVERS.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))[:300]
    if len(covers) != 300:
        print(f"expected 300 covers, found {len(covers)}")
        return 1
    for c in covers:
        shutil.copy2(c, OUT / "cover" / c.name)
    cover_stems = {c.stem for c in covers}
    print(f"cover: 300 images  sha1 {sha1_of_dir(OUT / 'cover')}")

    manifest = {
        "description": "Cover/stego image sets for CNN steganalysis (paper M4).",
        "cover_dir": "cover",
        "cover_count": 300,
        "cover_sha1": sha1_of_dir(OUT / "cover"),
        "image_spec": {
            "format": "JPEG",
            "size": "512x512",
            "channels": 1,
            "cover_quality_factor": 65,
            "note": ("Grayscale JPEG. Decode ONCE; never re-encode. The payload "
                     "lives in the quantised DCT coefficients and any "
                     "re-compression destroys it."),
        },
        "pairing": ("stego/<cell>/N.jpg is the SAME source image as cover/N.jpg. "
                    "Match by filename and keep pairs on the same side of any "
                    "train/test split."),
        "cells": [],
    }

    missing_cells = 0
    for name, arm, method, payload, src, channel in CELLS:
        if not src.is_dir():
            print(f"  MISSING source for {name}: {src}")
            missing_cells += 1
            continue
        dst = OUT / "stego" / name
        dst.mkdir(parents=True)
        imgs = sorted(src.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))
        for i in imgs:
            shutil.copy2(i, dst / i.name)
        unmatched = [i.stem for i in imgs if i.stem not in cover_stems]
        manifest["cells"].append({
            "cell": name, "arm": arm, "method": method,
            "payload_bpnzac": payload, "channel": channel,
            "stego_dir": f"stego/{name}", "count": len(imgs),
            "sha1": sha1_of_dir(dst),
            "stego_without_matching_cover": unmatched,
        })
        flag = "" if not unmatched else f"   !! {len(unmatched)} unmatched"
        print(f"{name:<20} {len(imgs):>3} images{flag}")

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"\n{len(manifest['cells'])} cells, {total / 1e6:.0f} MB at {OUT}")
    if missing_cells:
        print(f"WARNING: {missing_cells} cell(s) had no source directory")
    print("Zip this directory and upload it as a Kaggle Dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
