"""Full-scale stego generation (journal roadmap WP1 / Paper 2 groundwork).

Embeds every cell of the grid over the whole of BOSSbase - 10,000 covers - in
small, independently resumable chunks, several MATLAB processes at a time.

Why not just call the shard scripts with a 10,000-line manifest: each one opens
its CSV with 'w' and walks the manifest from the top, so an interruption at
image 9,000 throws the first 9,000 away. Here a cell is cut into chunks of
CHUNK images; a chunk is one MATLAB call with its own manifest and CSV, and is
marked done only when its CSV has one row per image. Kill it, reboot, rerun the
same command: finished chunks are skipped.

Seeds: the shard scripts seed image i of a manifest with seed_base + i. We pass
seed_base = (index of the chunk's first image), so every image is seeded by its
global 1-based index regardless of chunking. That is the same seeding the
300-image experiments used, so their stego images are reproduced bit-for-bit -
see --verify.

Order: tiers. All cells are taken to 2,000 images, then 5,000, then 10,000, so
a complete (smaller) grid exists long before the run ends.

    python experiments/exp13_fullscale.py --status
    python experiments/exp13_fullscale.py --pilot          # 2 tiny chunks/cell
    python experiments/exp13_fullscale.py --verify         # vs exp08/exp10 stegos
    python experiments/exp13_fullscale.py                  # the real run
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = paths.BASELINES / "Robust_Steganography"
PRECOVERS = paths.DATA / "BOSSbase_1.01"

CHUNK = 250
TIERS = [2000, 5000, 10000]
COVER_QF = 65
# Parameters are exactly those of exp08_arm_a.py / exp10_arm_b.py.
ARM_A_QF, ROAST_AB = 71, (0.2, 0.7)
SCRIPT_R, MINICER_STEP = 10, 6


def cells() -> list[dict]:
    out = []
    for p in (0.05, 0.1, 0.2, 0.3):
        out.append(dict(name=f"J-UNIWARD_{p}", kind="juniward", payload=p))
        out.append(dict(name=f"GMAS_{p}", kind="gmas", payload=p))
        out.append(dict(name=f"ROAST_{p}", kind="roast", payload=p))
    for p in (0.05, 0.1, 0.2):
        out.append(dict(name=f"SCRIPT_{p}", kind="script", payload=p))
        out.append(dict(name=f"MINICER_{p}", kind="minicer", payload=p))
    return out


def matlab_call(cell: dict, manifest: pathlib.Path, out_dir: pathlib.Path,
                csv_path: pathlib.Path, seed_base: int) -> str:
    r, m, o, c = (REPO.as_posix(), manifest.as_posix(), out_dir.as_posix(),
                  csv_path.as_posix())
    p, k = cell["payload"], cell["kind"]
    if k == "juniward":
        return f"juniward_embed_shard('{r}','{m}','{o}','{c}',{p},{seed_base})"
    if k == "gmas":
        return (f"gmas_embed_shard('{r}','{m}','{o}','{c}',{p},{COVER_QF},"
                f"{ARM_A_QF},{seed_base})")
    if k == "roast":
        return (f"roast_recon_shard('{r}','{m}','{o}','{c}',{p},{COVER_QF},"
                f"{ARM_A_QF},{seed_base},{ROAST_AB[0]},{ROAST_AB[1]})")
    if k == "script":
        return (f"script_embed_shard('{r}','{m}','{o}','{c}',{p},{COVER_QF},"
                f"{COVER_QF},{seed_base},{SCRIPT_R},{SCRIPT_R},0,0)")
    if k == "minicer":
        return (f"minicer_embed_shard('{r}','{m}','{o}','{c}',{p},{COVER_QF},"
                f"{COVER_QF},{seed_base},{MINICER_STEP})")
    raise ValueError(k)


def find_matlab() -> str:
    exe = shutil.which("matlab")
    if exe:
        return exe
    f = sorted(pathlib.Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"))
    if not f:
        raise SystemExit("MATLAB not found")
    return str(f[-1])


def csv_rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def row_ok(r: dict) -> bool:
    """Did this image embed? The shard scripts disagree on the column: GMAS
    writes embed_ok (1/0), the others write status ("ok" or an error)."""
    if "embed_ok" in r:
        return (r["embed_ok"] or "").strip() not in ("", "0")
    return (r.get("status") or "").strip().lower() == "ok"


class Job:
    def __init__(self, root: pathlib.Path, cell: dict, k: int,
                 covers: list[pathlib.Path], chunk: int):
        self.cell, self.k = cell, k
        self.covers = covers[k * chunk:(k + 1) * chunk]
        self.seed_base = k * chunk          # image i gets seed (k*chunk + i)
        self.dir = root / cell["name"]
        self.manifest = self.dir / "manifests" / f"chunk_{k:03d}.txt"
        self.csv = self.dir / "csv" / f"chunk_{k:03d}.csv"
        self.log = self.dir / "logs" / f"chunk_{k:03d}.log"
        self.proc: subprocess.Popen | None = None
        self.t0 = 0.0

    @property
    def label(self) -> str:
        return f"{self.cell['name']}#{self.k:03d}"

    def done(self) -> bool:
        return len(csv_rows(self.csv)) == len(self.covers)

    def start(self, matlab: str) -> None:
        for d in (self.manifest.parent, self.csv.parent, self.log.parent):
            d.mkdir(parents=True, exist_ok=True)
        if self.cell["kind"] == "script":   # SCRIPT wants precover|cover pairs
            lines = [f"{PRECOVERS / (c.stem + '.pgm')}|{c}" for c in self.covers]
        else:
            lines = [str(c) for c in self.covers]
        self.manifest.write_text("\n".join(lines), encoding="utf-8")
        if self.csv.exists():
            self.csv.unlink()               # a partial chunk is redone whole
        cmd = matlab_call(self.cell, self.manifest, self.dir, self.csv,
                          self.seed_base)
        self.t0 = time.time()
        self.proc = subprocess.Popen([matlab, "-batch", cmd], cwd=str(HERE),
                                     stdout=self.log.open("w"),
                                     stderr=subprocess.STDOUT)

    def finish(self) -> tuple[bool, str]:
        """Check the chunk, then drop its after-channel images.

        The channel output is only needed to measure BER, which is already in
        the CSV; keeping 10,000 recompressed JPEGs per cell would double the
        disk footprint for nothing. The stego images are kept.
        """
        rows = csv_rows(self.csv)
        if len(rows) != len(self.covers):
            return False, f"csv has {len(rows)} rows, expected {len(self.covers)}"
        chan = self.dir / "afterchannel"
        for c in self.covers:
            f = chan / f"{c.stem}.jpg"
            if f.exists():
                f.unlink()
        ok = sum(1 for r in rows if row_ok(r))
        return True, f"{ok}/{len(rows)} embedded"


def build_jobs(root, covers, chunk, limit_chunks=None) -> list[Job]:
    n_chunks = (len(covers) + chunk - 1) // chunk
    jobs = [Job(root, c, k, covers, chunk) for k in range(n_chunks) for c in cells()]

    def tier(j: Job) -> int:
        end = min((j.k + 1) * chunk, len(covers))
        return next((i for i, t in enumerate(TIERS) if end <= t), len(TIERS))

    jobs.sort(key=lambda j: (tier(j), j.k))
    if limit_chunks is not None:
        jobs = [j for j in jobs if j.k < limit_chunks]
    return jobs


def status(root, jobs) -> None:
    per: dict[str, list[int]] = {}
    secs, imgs = 0.0, 0
    for j in jobs:
        d = per.setdefault(j.cell["name"], [0, 0, 0, 0])
        rows = csv_rows(j.csv) if j.csv.exists() else []
        d[1] += len(j.covers)
        if len(rows) == len(j.covers):
            d[0] += len(rows)
            d[2] += sum(1 for r in rows if row_ok(r))
            d[3] += sum(1 for r in rows if (r.get("bit_error_rate") or "") != ""
                        and float(r["bit_error_rate"]) == 0)
            for r in rows:
                try:
                    secs += float(r["seconds"]); imgs += 1
                except (KeyError, ValueError, TypeError):
                    pass
    print(f"{'cell':<17}{'done':>8}{'of':>8}{'embedded':>10}{'errorless':>11}")
    print("-" * 54)
    for name, (done, tot, ok, errless) in per.items():
        print(f"{name:<17}{done:>8}{tot:>8}{ok:>10}{errless:>11}")
    done = sum(v[0] for v in per.values())
    tot = sum(v[1] for v in per.values())
    print("-" * 54)
    print(f"{'total':<17}{done:>8}{tot:>8}   ({100 * done / max(tot, 1):.1f}%)")
    if imgs:
        print(f"mean compute {secs / imgs:.2f} s/image (single-process seconds)")
    size = sum(f.stat().st_size for f in root.rglob("*.jpg")) if root.exists() else 0
    print(f"stego on disk: {size / 1e9:.2f} GB at {root}")


def verify(root, n=40) -> int:
    """New stegos for the first images must equal the 300-image runs' bytes."""
    old = {"J-UNIWARD": "exp08_arm_a/J_UNIWARD_p{p}", "GMAS": "exp08_arm_a/GMAS_p{p}_q71",
           "ROAST": "exp08_arm_a/ROAST_recon_p{p}_q71",
           "SCRIPT": "exp10_arm_b/SCRIPT_p{p}_q65", "MINICER": "exp10_arm_b/MINICER_p{p}_q65"}
    bad = 0
    for c in cells():
        method = c["name"].rsplit("_", 1)[0]
        ref = paths.RESULTS / old[method].format(p=c["payload"]) / "stego"
        new = root / c["name"] / "stego"
        both = [f for f in sorted(new.glob("*.jpg"), key=lambda p: (len(p.stem), p.stem))
                if (ref / f.name).exists()][:n]
        if not both:
            print(f"{c['name']:<17} nothing to compare yet")
            continue
        same = sum(1 for f in both if f.read_bytes() == (ref / f.name).read_bytes())
        print(f"{c['name']:<17} {same}/{len(both)} byte-identical to the 300-image run")
        bad += len(both) - same
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent MATLAB processes; scaling peaks at 4 here")
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="8-image chunks, 2 per cell, into a separate folder")
    ap.add_argument("--timeout-per-image", type=float, default=40.0,
                    help="seconds; a chunk exceeding chunk*this is killed and retried later")
    args = ap.parse_args()

    if args.pilot:
        args.chunk, args.count = 8, 16
    root = paths.ensure_results("exp13_fullscale_pilot" if args.pilot
                                else "exp13_fullscale")

    cover_dir = paths.DATA / f"BOSSbase_QF{COVER_QF}_full"
    have = len(list(cover_dir.glob("*.jpg"))) if cover_dir.exists() else 0
    if have < args.count and not (args.status or args.verify):
        print(f"preparing {args.count} covers at QF{COVER_QF}...", flush=True)
        subprocess.run([sys.executable, str(HERE / "prep_covers.py"), "--qf",
                        str(COVER_QF), "--count", str(args.count), "--tag", "full"],
                       check=True)
    covers = sorted(cover_dir.resolve().glob("*.jpg"),
                    key=lambda p: (len(p.stem), p.stem))[:args.count]
    jobs = build_jobs(root, covers, args.chunk)

    if args.status:
        status(root, jobs)
        return 0
    if args.verify:
        return 1 if verify(root) else 0

    matlab = find_matlab()
    todo = [j for j in jobs if not j.done()]
    print(f"{len(covers)} covers, {len(cells())} cells, chunk {args.chunk}: "
          f"{len(jobs) - len(todo)} chunks done, {len(todo)} to do, "
          f"{args.workers} workers", flush=True)

    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        runlog.write(line + "\n"); runlog.flush()

    note(f"START {len(todo)} chunks to do")
    running: list[Job] = []
    failed: list[str] = []
    finished = 0
    t_start = time.time()
    limit = args.timeout_per_image * args.chunk
    try:
        while todo or running:
            while todo and len(running) < args.workers:
                j = todo.pop(0)
                j.start(matlab)
                running.append(j)
            time.sleep(5)
            for j in list(running):
                rc = j.proc.poll()
                if rc is None:
                    if time.time() - j.t0 > limit:
                        j.proc.kill()
                        note(f"TIMEOUT {j.label} after {limit:.0f}s - will retry on rerun")
                        failed.append(j.label)
                        running.remove(j)
                    continue
                running.remove(j)
                ok, msg = j.finish()
                finished += 1
                if ok:
                    el = time.time() - t_start
                    left = len(todo) + len(running)
                    eta = el / finished * left / 3600
                    note(f"done {j.label}  {msg}  {time.time() - j.t0:.0f}s  "
                         f"| {left} chunks left, ~{eta:.1f} h")
                else:
                    note(f"FAILED {j.label} (exit {rc}): {msg} - see {j.log}")
                    failed.append(j.label)
    except KeyboardInterrupt:
        note("INTERRUPTED - killing workers; rerun the same command to resume")
        for j in running:
            j.proc.kill()
        return 130

    note(f"END  {finished} chunks run, {len(failed)} failed")
    (root / "last_run.json").write_text(json.dumps(
        dict(finished=finished, failed=failed,
             hours=(time.time() - t_start) / 3600), indent=2), encoding="utf-8")
    if failed:
        print("failed chunks (rerun to retry):", ", ".join(failed[:20]))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
