"""M1 gate driver: invoke MATLAB non-interactively and record the outcome.

Runs m1_verify_baselines.m, which loads the mex toolchain and reproduces the
GMAS robustness test bundled with the baselines repo. See PLAN.md M1 — this is
the project's most important gate, because an unreproducible baseline makes
every comparison in both papers unanchored.

    python experiments/exp02_m1_gate.py [--timeout 3600]
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = paths.BASELINES / "Robust_Steganography"


def find_matlab() -> str | None:
    exe = shutil.which("matlab")
    if exe:
        return exe
    # Newest install wins if several are present.
    candidates = sorted(pathlib.Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"))
    return str(candidates[-1]) if candidates else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=3600,
                    help="seconds; MATLAB startup alone is ~30-60 s")
    args = ap.parse_args()

    matlab = find_matlab()
    if not matlab:
        print("MATLAB not found. It is required for M1 — see baselines/README.md.")
        return 1
    if not REPO.exists():
        print(f"baselines repo not found at {REPO}\n"
              f"  git clone https://github.com/coriverchen/Robust_Steganography.git")
        return 1

    out_dir = paths.ensure_results("m1_gate")
    cmd = (f"m1_verify_baselines('{REPO.as_posix()}', '{out_dir.as_posix()}')")

    print(f"matlab : {matlab}")
    print(f"repo   : {REPO}")
    print(f"output : {out_dir}")
    print(f"running: {cmd}\n(MATLAB startup takes ~30-60 s)\n", flush=True)

    try:
        proc = subprocess.run(
            [matlab, "-batch", cmd],
            cwd=str(HERE), timeout=args.timeout,
            capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {args.timeout}s. Raise --timeout, or run the .m "
              f"interactively to see where it stalls.")
        return 2

    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print("--- stderr ---")
        print(proc.stderr)

    log = out_dir / "m1_gate_log.txt"
    print(f"\nlog: {log if log.exists() else '(not written)'}")
    print(f"matlab exit code: {proc.returncode}")

    if proc.returncode != 0:
        return proc.returncode

    # The gate is a human read of ave_error_rate, but flag the obvious cases.
    text = (log.read_text(encoding="utf-8", errors="replace") if log.exists()
            else proc.stdout)
    if "GATE: FAIL" in text:
        print("\nGATE FAILED — see baselines/README.md for fallbacks.")
        return 3
    if "ave_error_rate" in text:
        for line in text.splitlines():
            if "ave_error_rate" in line:
                print(f"\n>>> {line.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
