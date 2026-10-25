"""Pack the curriculum dataset of exp42 as soon as its embedding is done, instead
of after its DCTR stage - the Kaggle run on it takes about ten hours and need
not wait two more for a detector it does not use.

Waits for exp42's delivery table (written when the embedding stage ends), then
writes the same file exp42 would write; exp42 finds it and skips its own pack.

    python experiments/exp42_pack_early.py
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys
import time
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp13_fullscale as e13  # noqa: E402
import exp40_script_fix as e40  # noqa: E402
import exp42_script_fix_curriculum as e42  # noqa: E402

ROOT = paths.RESULTS / "exp40_script_fix"


def main() -> int:
    best = json.loads((ROOT / "decision.json").read_text())["chosen"]
    top = max(e42.HIGH)
    done = re.compile(rf"SCRIPT-fix{best}_{top}\s+{e42.N}\s")
    while not done.search((ROOT / "run.log").read_text(encoding="utf-8", errors="replace")):
        time.sleep(60)
    out = paths.ensure_results("exp42_script_fix_curriculum_5000")
    dest = out / "stego_SCRIPT-fix-curriculum_5000.bin"
    if dest.exists():
        print("already packed")
        return 0
    tmp = out / "early_pack.zip"
    manifest = dict(
        description=f"SCRIPT at strength 0 with cost repair {best} "
                    f"({e40.VARIANTS[best]}), payloads {e42.CHAIN}; covers included.",
        cover_dir="cover", cover_count=e42.N, cells=[],
        image_spec=dict(format="JPEG", size="512x512", channels=1,
                        cover_quality_factor=e13.COVER_QF),
        pairing="stego/<cell>/N.jpg is the same source image as cover/N.jpg.",
        source="experiments/exp40_script_fix.py and exp42_script_fix_curriculum.py")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for i in range(1, e42.N + 1):
            z.write(e40.COVERS / f"{i}.jpg", f"dataset/cover/{i}.jpg")
        for p in e42.CHAIN:
            files = e40.stego_files(ROOT, f"SCRIPT-fix{best}_{p}", e42.N)
            if len(files) < 0.95 * e42.N:
                raise SystemExit(f"SCRIPT-fix{best}_{p}: only {len(files)} images")
            for s, f in files:
                z.write(f, f"dataset/stego/SCRIPT-fix_{p}/{s}.jpg")
            manifest["cells"].append(dict(
                cell=f"SCRIPT-fix_{p}", arm="B", method=f"SCRIPT (R=0, cost repair {best})",
                payload_bpnzac=p, channel="QF65->QF65", stego_dir=f"stego/SCRIPT-fix_{p}",
                count=len(files)))
            print(f"SCRIPT-fix_{p}: {len(files)} images", flush=True)
        z.writestr("dataset/MANIFEST.json", json.dumps(manifest, indent=2))
    part = out / "early_pack.bin.part"
    with part.open("wb") as fo, tmp.open("rb") as fi:
        fo.write(b"STEGO-EVAL packed dataset; open with zipfile.".ljust(64, b" "))
        shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)
    tmp.unlink()
    part.replace(dest)                       # appears only when complete
    print(f"wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
