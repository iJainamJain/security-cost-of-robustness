"""Fetch the five reference implementations and rebuild our patched copies.

    python setup_baselines.py

The implementations of GMAS, ROAST, MINICER, PMAS and SCRIPT are distributed together at
https://github.com/coriverchen/Robust_Steganography. That repository carries no licence, so
this one does not redistribute it. This script clones it at the commit we used, then builds
baselines/patched/ from the upstream files plus our patches in baselines/patches/.

The clone itself is never modified. The patched files shadow it through
addpath(..., '-begin') in the experiments' MATLAB shards; baselines/patched_readme/ explains
each change.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
URL = "https://github.com/coriverchen/Robust_Steganography.git"
COMMIT = "f1765807f50f2bfedbc152aef8f885e34ca77c99"
CLONE = ROOT / "baselines" / "Robust_Steganography"
PATCHED = ROOT / "baselines" / "patched"
PATCHES = ROOT / "baselines" / "patches"

# patched file -> upstream file it starts from
FILES = {
    "GMAS/stc3_embed.m": "GMAS/stc3_embed.m",
    "PMAS/preprocessUNICORE.m": "PMAS/preprocessUNICORE.m",
    "PMAS/postprocessUNICORE.m": "PMAS/postprocessUNICORE.m",
    "PMAS/J_UNIWARDcost.m": "MINICER/J_UNIWARDcost.m",   # copied unmodified: a missing dependency
    "PMAS/UERDcost.m": "MINICER/UERDcost.m",             # copied unmodified: a missing dependency
    "SCRIPT/costGISS.m": "SCRIPT/costGISS.m",
}


def git(*args: str, cwd: pathlib.Path = ROOT) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True)


def main() -> int:
    if not CLONE.exists():
        git("clone", URL, str(CLONE))
    git("checkout", "--quiet", COMMIT, cwd=CLONE)

    for target, origin in FILES.items():
        dst = PATCHED / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CLONE / origin, dst)
        patch = PATCHES / (target.replace("/", "__") + ".patch")
        if patch.exists():
            git("apply", "--unsafe-paths", f"--directory={dst.parent.relative_to(ROOT).as_posix()}", str(patch))
        print("built", dst.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
