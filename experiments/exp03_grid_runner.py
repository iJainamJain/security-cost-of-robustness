"""Sharded MATLAB runner — parallelism without Parallel Computing Toolbox.

PCT is not installed on this machine (see baselines/BASELINE_STATUS.md), so
MATLAB's `parfor` degrades to a serial loop and buys nothing. This gets real
parallelism the only way left: shard the cover list and run N independent MATLAB
processes, one per shard.

Each process pays MATLAB's ~15-30 s startup once, so shards want to be large
enough to amortise it. `--bench` measures that trade-off directly rather than
guessing at it.

    # time one worker vs several on the bundled GMAS covers
    python experiments/exp03_grid_runner.py --bench

    # a real run
    python experiments/exp03_grid_runner.py \
        --covers data/BOSSbase_1.01_QF65 --workers 6 --payload 0.1

Memory is the binding constraint: ~1-2 GB per MATLAB process against 15.6 GB of
RAM, so 6 workers is the practical ceiling. --workers defaults accordingly.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = paths.BASELINES / "Robust_Steganography"
BUNDLED_COVERS = REPO / "GMAS" / "cover_dir_QF65"

DEFAULT_WORKERS = 6  # RAM-bound, not core-bound


def find_matlab() -> str | None:
    exe = shutil.which("matlab")
    if exe:
        return exe
    found = sorted(pathlib.Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"))
    return str(found[-1]) if found else None


def collect_covers(cover_dir: pathlib.Path, limit: int | None) -> list[pathlib.Path]:
    # Must be absolute: the MATLAB workers run with cwd=experiments/, so a
    # relative path written into a manifest resolves against the wrong
    # directory and every image fails instantly inside the try/catch.
    cover_dir = cover_dir.resolve()
    files = sorted(cover_dir.glob("*.jpg"),
                   key=lambda p: (len(p.stem), p.stem))  # 1,2,...,10 not 1,10,2
    if not files:
        raise SystemExit(f"no .jpg covers in {cover_dir}")
    return files[:limit] if limit else files


def shard(items: list, n: int) -> list[list]:
    """Round-robin, so every shard gets a similar mix of image sizes."""
    out: list[list] = [[] for _ in range(n)]
    for i, it in enumerate(items):
        out[i % n].append(it)
    return [s for s in out if s]


def run_shards(matlab: str, covers: list[pathlib.Path], workers: int,
               out_root: pathlib.Path, payload: float,
               cover_qf: int, attack_qf: int, quiet: bool = False,
               single_thread: bool = False) -> tuple[float, list[pathlib.Path]]:
    """Launch one MATLAB process per shard, in parallel. Returns (seconds, csvs).

    MATLAB multithreads its own BLAS and image built-ins, so a single process
    already saturates the machine. Running several then costs more in core
    contention than it gains in parallelism. ``single_thread`` passes
    -singleCompThread so each process takes one core and N processes give real
    N-way parallelism — measure both with --bench before committing.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    groups = shard(covers, workers)

    procs, csvs = [], []
    t0 = time.time()
    for i, group in enumerate(groups):
        sdir = out_root / f"shard{i:02d}"
        sdir.mkdir(parents=True, exist_ok=True)
        manifest = sdir / "manifest.txt"
        manifest.write_text("\n".join(str(p) for p in group), encoding="utf-8")
        out_csv = sdir / "results.csv"
        csvs.append(out_csv)

        # seed_base is offset per shard so a cover's message never depends on
        # which shard it landed in.
        seed_base = i * 100_000
        cmd = (f"gmas_embed_shard('{REPO.as_posix()}','{manifest.as_posix()}',"
               f"'{sdir.as_posix()}','{out_csv.as_posix()}',{payload},"
               f"{cover_qf},{attack_qf},{seed_base})")
        argv = [matlab]
        if single_thread:
            argv.append("-singleCompThread")
        argv += ["-batch", cmd]
        procs.append(subprocess.Popen(
            argv, cwd=str(HERE),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))

    outputs = [p.communicate()[0] for p in procs]
    elapsed = time.time() - t0

    if not quiet:
        for i, (p, o) in enumerate(zip(procs, outputs)):
            if p.returncode != 0:
                print(f"  shard {i} exited {p.returncode}:\n{(o or '')[-800:]}")
    return elapsed, csvs


def summarise(csvs: list[pathlib.Path]) -> dict:
    rows = []
    for c in csvs:
        if not c.exists():
            continue
        with c.open(newline="", encoding="utf-8") as fh:
            rows.extend(list(csv.DictReader(fh)))
    if not rows:
        return {"n": 0}

    def nums(key):
        out = []
        for r in rows:
            try:
                out.append(float(r[key]))
            except (ValueError, KeyError, TypeError):
                pass
        return out

    bers = nums("bit_error_rate")
    secs = nums("seconds")
    psnrs = nums("psnr_db")
    return {
        "n": len(rows),
        "decoded": len(bers),
        "mean_ber": sum(bers) / len(bers) if bers else float("nan"),
        "errorless": sum(1 for b in bers if b == 0),
        "mean_psnr": sum(psnrs) / len(psnrs) if psnrs else float("nan"),
        "cpu_seconds": sum(secs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers", type=pathlib.Path, default=BUNDLED_COVERS)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--payload", type=float, default=0.1)
    ap.add_argument("--cover-qf", type=int, default=65)
    ap.add_argument("--attack-qf", type=int, default=71)
    ap.add_argument("--bench", action="store_true",
                    help="time 1 vs N workers, multithreaded and "
                         "-singleCompThread, to find the real optimum")
    ap.add_argument("--single-thread", action="store_true",
                    help="run each MATLAB with -singleCompThread; use when "
                         "sharding across many processes")
    args = ap.parse_args()

    matlab = find_matlab()
    if not matlab:
        print("MATLAB not found — see baselines/README.md")
        return 1
    if not REPO.exists():
        print(f"baselines repo missing at {REPO}")
        return 1

    covers = collect_covers(args.covers, args.limit)
    print(f"matlab : {matlab}")
    print(f"covers : {len(covers)} from {args.covers}")

    if args.bench:
        print("\n=== worker-count benchmark ===")
        print("(MATLAB startup is ~10 s per process, paid once per shard, so "
              "short runs are startup-dominated. Watch the COMPUTE column: if it "
              "rises with worker count, processes are contending for cores.)\n")
        results = {}
        counts = sorted({1, min(3, len(covers)), min(args.workers, len(covers))})
        for st in (False, True):
            label = "-singleCompThread" if st else "multithreaded"
            print(f"  --- {label} ---")
            for w in counts:
                out = paths.ensure_results("exp03_bench", f"{'st' if st else 'mt'}_w{w}")
                elapsed, csvs = run_shards(matlab, covers, w, out, args.payload,
                                           args.cover_qf, args.attack_qf,
                                           quiet=True, single_thread=st)
                s = summarise(csvs)
                results[(st, w)] = (elapsed, s)
                n = max(s.get("n", 0), 1)
                print(f"    workers={w:<2} wall {elapsed:6.1f}s  "
                      f"compute {s.get('cpu_seconds', 0):6.1f}s  "
                      f"({s.get('cpu_seconds', 0) / n:.2f}s/img)  "
                      f"BER {s.get('mean_ber', float('nan')):.4g}")

        print("\n  wall-clock speedup vs 1 multithreaded worker:")
        base = results.get((False, 1))
        if base:
            for (st, w), (el, _) in sorted(results.items()):
                tag = "st" if st else "mt"
                print(f"    {tag} workers={w:<2} {base[0] / el:.2f}x")
            print("\n  Pick the configuration with the best wall time on a shard "
                  "large enough\n  that startup is amortised. Re-run --bench on "
                  "~200 covers before\n  committing to the full grid.")
        return 0

    out = paths.ensure_results(
        "exp03_grid",
        f"gmas_p{args.payload}_q{args.cover_qf}to{args.attack_qf}_w{args.workers}")
    print(f"workers: {args.workers}\noutput : {out}\n")
    elapsed, csvs = run_shards(matlab, covers, args.workers, out,
                               args.payload, args.cover_qf, args.attack_qf,
                               single_thread=args.single_thread)
    s = summarise(csvs)
    n, decoded = s.get("n", 0), s.get("decoded", 0)
    print(f"\n=== {len(covers)} covers in {elapsed:.1f}s "
          f"({elapsed / len(covers):.2f}s/image wall) ===")
    print(f"  decoded        : {decoded}/{n}")
    print(f"  errorless      : {s.get('errorless', 0)}")
    print(f"  mean BER       : {s.get('mean_ber', float('nan')):.6g}")
    print(f"  mean PSNR      : {s.get('mean_psnr', float('nan')):.2f} dB")
    print(f"\nper-image CSVs under {out}")

    # A run where most images fail is a broken configuration, not a result.
    # Fail loudly rather than letting a table of NaNs look like data.
    if n == 0 or decoded < 0.9 * n:
        print(f"\nERROR: only {decoded}/{n} images produced a result.")
        print("  Check a shard's MATLAB output above. Common causes: cover paths "
              "not absolute,\n  covers with zero non-zero AC coefficients (flat "
              "images), or a missing toolbox.")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
