"""SCRIPT at full scale: the low-payload scale check and Paper 1's corrected rows.

Phase A  SCRIPT (R = 10) at 0.01 and 0.02 bpnzac on all 10,000 covers.
         exp18 found 0.01 at chance with 1,000 training pairs, and that
         spreading 0.2 bpnzac over 20 such covers lifts batch P_E to 0.43. A
         better-trained detector is the likeliest way for that to evaporate
         (J-UNIWARD at 0.05 was invisible at 300 images, detected at 10,000).

Phase B  SCRIPT (R = 0) at 0.05, 0.1 and 0.2 on all 10,000 covers. exp19 showed
         R = 0 is as error-free as R = 10 on SCRIPT's own channel and far less
         detectable, so Paper 1's R = 10 SCRIPT rows must be replaced.

Why Phase A is NOT at R = 0: at 0.01-0.02 bpnzac a smooth image carries only
100-450 message bits; with R = 0 those are spread over ~250,000 usable
coefficients, and the STC library fails ("Maximum number of trials in layered
construction"), sometimes as a caught error and sometimes as an access
violation that kills MATLAB. A first attempt lost 7 of 12 chunks that way.
R = 10 leaves fewer usable coefficients and embedded 3,999 of 4,000 images at
these payloads in exp18. Keeping only the images R = 0 can embed would bias the
sample towards textured covers, so the low-payload check stays at R = 10.

A MEX crash takes the whole MATLAB process with it, so the embed stage here
resumes a chunk after the image that crashed, recording that image as failed.

    python experiments/exp21_script_low_fullscale.py
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
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp13_fullscale as e13  # noqa: E402
import exp16_dctr_fullscale as e16  # noqa: E402
import exp17_batch_pooling as e17  # noqa: E402

E16_OUT = paths.RESULTS / "exp16_dctr_fullscale"
E18_OUT = paths.RESULTS / "exp18_low_payload"
E19_OUT = paths.RESULTS / "exp19_followups"
COVERS = paths.DATA / f"BOSSbase_QF{e13.COVER_QF}_full"
CELLS_A = [dict(name=f"SCRIPT_{p}", kind="script", payload=p, reuse=E18_OUT)
           for p in (0.01, 0.02)]
CELLS_B = [dict(name=f"SCRIPT-R0_{p}", kind="script", payload=p, R=0, reuse=E19_OUT)
           for p in (0.05, 0.1, 0.2)]
TIERS = (5000, 10000)
FCHUNK = 500

_orig_call = e13.matlab_call


def _call(cell, manifest, out_dir, csv_path, seed_base):
    if cell.get("R") is None:
        return _orig_call(cell, manifest, out_dir, csv_path, seed_base)
    r, q = cell["R"], e13.COVER_QF
    return (f"script_embed_shard('{e13.REPO.as_posix()}','{manifest.as_posix()}',"
            f"'{out_dir.as_posix()}','{csv_path.as_posix()}',{cell['payload']},{q},{q},"
            f"{seed_base},{r},{r},0,0)")


e13.matlab_call = _call


class Part(e13.Job):
    """The remainder of a chunk after MATLAB died part-way through it."""

    def __init__(self, parent, covers, seed_base, part_no):
        self.cell, self.k, self.covers, self.seed_base = parent.cell, parent.k, covers, seed_base
        self.dir = parent.dir
        tag = f"chunk_{parent.k:03d}_part{part_no}"
        self.manifest = self.dir / "manifests" / f"{tag}.txt"
        self.csv = self.dir / "csv_parts" / f"{tag}.csv"     # NOT in csv/: globbed there
        self.log = self.dir / "logs" / f"{tag}.log"
        self.proc, self.t0 = None, 0.0
        self.main_csv = getattr(parent, "main_csv", parent.csv)
        self.part_no = part_no
        self.csv.parent.mkdir(parents=True, exist_ok=True)

    @property
    def label(self) -> str:
        return f"{self.cell['name']}#{self.k:03d}.part{self.part_no}"


def _append_lines(main: pathlib.Path, lines: list[str]) -> None:
    with main.open("a", encoding="utf-8", newline="") as fh:
        for ln in lines:
            fh.write(ln.rstrip("\r\n") + "\n")


def _placeholder(main: pathlib.Path, stem: str) -> str:
    header = main.read_text(encoding="utf-8", errors="replace").splitlines()[0].split(",")
    vals = {"cover": stem, "status": "MATLAB crash (access violation)", "embed_ok": "0"}
    return ",".join(vals.get(h.strip(), "") for h in header)


def settle(j, rc, note):
    """Fold a finished job into its chunk's CSV. Returns a continuation or None."""
    main = getattr(j, "main_csv", j.csv)
    rows = e13.csv_rows(j.csv)
    if isinstance(j, Part) and j.csv.exists():
        body = j.csv.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
        _append_lines(main, body[:len(rows)])
    if len(rows) == len(j.covers):
        chan = j.dir / "afterchannel"
        for c in j.covers:
            (chan / f"{c.stem}.jpg").unlink(missing_ok=True)
        return None
    # MATLAB died on image number len(rows) of this job (0-based).
    bad = j.covers[len(rows)]
    _append_lines(main, [_placeholder(main, bad.stem)])
    note(f"CRASH {j.label} (exit {rc}) on image {bad.stem} after {len(rows)} rows - "
         f"recorded as failed, resuming after it")
    rest = j.covers[len(rows) + 1:]
    if not rest:
        return None
    return Part(j, rest, j.seed_base + len(rows) + 1, getattr(j, "part_no", 0) + 1)


def reuse(root, cells, chunk, note) -> None:
    """Earlier experiments embedded the first 2,000 images of these cells with
    the same per-image seeds (seed = global image index), so those files are
    what this run would produce. Copy them."""
    for c in cells:
        src, dst = c["reuse"] / c["name"], root / c["name"]
        if not (src / "csv").is_dir() or (dst / "csv" / "chunk_000.csv").exists():
            continue
        (dst / "csv").mkdir(parents=True, exist_ok=True)
        (dst / "stego").mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted((src / "csv").glob("chunk_*.csv")):
            if len(e13.csv_rows(f)) == chunk:
                shutil.copy2(f, dst / "csv" / f.name); n += 1
        for f in (src / "stego").glob("*.jpg"):
            shutil.copy2(f, dst / "stego" / f.name)
        note(f"reused {n} chunks of {c['name']} from {c['reuse'].name}")


def _kill_tree(pid: int) -> None:
    """matlab.exe is a launcher; the worker is its child. Kill both."""
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


def embed(root, cells, covers, chunk, workers, note) -> None:
    n_chunks = (len(covers) + chunk - 1) // chunk
    jobs = [e13.Job(root, c, k, covers, chunk) for k in range(n_chunks) for c in cells]
    todo = [j for j in jobs if not j.done()]
    note(f"embed {[c['name'] for c in cells]}: {len(jobs) - len(todo)} chunks done, "
         f"{len(todo)} to do")
    matlab, running, limit = e13.find_matlab(), [], 40.0 * chunk

    def restart(j, why):
        # 2026-10-12: four MATLAB workers hung at once (three before writing a
        # line, one after finishing its chunk) and stalled a queue for an hour.
        _kill_tree(j.proc.pid)
        if why.startswith("HUNG"):
            # The cause on 2026-10-12 was MATLAB's licensing helper, stuck after two
            # days up: every new MATLAB waited on it for ever. MATLAB restarts it.
            subprocess.run(["taskkill", "/F", "/IM", "MathWorksServiceHost.exe"],
                           capture_output=True)
        running.remove(j)
        j.retries = getattr(j, "retries", 0) + 1
        note(f"{why} {j.label} - restart {j.retries}")
        if j.retries <= 5:
            j.start(matlab); running.append(j)

    while todo or running:
        while todo and len(running) < workers:
            j = todo.pop(0); j.start(matlab); running.append(j)
        time.sleep(5)
        for j in list(running):
            rc = j.proc.poll()
            if rc is None:
                age = time.time() - j.t0
                rows = len(e13.csv_rows(j.csv))
                if age > limit:
                    # Also what a laptop that slept for hours looks like (2026-10-13):
                    # the chunk is redone rather than dropped.
                    restart(j, "TIMEOUT")
                elif rows == len(j.covers) and time.time() - j.csv.stat().st_mtime > 180:
                    _kill_tree(j.proc.pid)          # finished, MATLAB did not exit
                    j.proc.wait()
                    note(f"MATLAB did not exit after finishing {j.label} - killed")
                elif age > 600 and not j.csv.exists() and (not j.log.exists()
                                                           or j.log.stat().st_size == 0):
                    restart(j, "HUNG AT START (no output for 10 min)")
                continue
            if not j.csv.exists() and getattr(j, "retries", 0) <= 5:
                restart(j, f"exited (code {rc}) before writing anything:")
                continue
            running.remove(j)
            cont = settle(j, rc, note)
            if cont is not None:
                cont.start(matlab); running.append(cont)
                continue
            main = getattr(j, "main_csv", j.csv)
            rows = e13.csv_rows(main)
            ok = sum(1 for r in rows if e13.row_ok(r))
            note(f"done {j.cell['name']}#{j.k:03d}  {ok}/{len(rows)} embedded  "
                 f"| {len(todo) + len(running)} left")
    for c in cells:
        shutil.rmtree(root / c["name"] / "afterchannel", ignore_errors=True)


def ok_set(root, name) -> set[str]:
    good = set()
    for f in sorted((root / name / "csv").glob("chunk_*.csv")):
        for r in e13.csv_rows(f):
            s = (r.get("cover") or "").strip()
            if s.isdigit() and e13.row_ok(r):
                good.add(s)
    return good


def features(root, cells, count, note) -> None:
    import torch

    from dctr import DCTR, DIM, read_jpeg
    torch.set_num_threads(8)
    ex = DCTR(e13.COVER_QF, "cpu")
    for c in cells:
        good, d = ok_set(root, c["name"]), root / c["name"] / "stego"
        for k in range((count + FCHUNK - 1) // FCHUNK):
            dest = root / "features" / c["name"] / f"chunk_{k:04d}.npz"
            if dest.exists():
                continue
            stems = [str(i) for i in range(k * FCHUNK + 1, min((k + 1) * FCHUNK, count) + 1)
                     if str(i) in good and (d / f"{i}.jpg").exists()
                     and (d / f"{i}.jpg").stat().st_size > 0]
            feats = np.empty((len(stems), DIM), dtype=np.float32)
            for i, s in enumerate(stems):
                feats[i] = ex.features(*read_jpeg(d / f"{s}.jpg"))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.parent / (dest.stem + ".tmp.npz")
            np.savez(tmp, feats=feats, stems=np.array(stems))
            tmp.replace(dest)
        note(f"features {c['name']} complete ({len(good)} embedded images)")


def load_new(root, name, n_images):
    feats, stems = [], []
    for k in range((n_images + FCHUNK - 1) // FCHUNK):
        z = np.load(root / "features" / name / f"chunk_{k:04d}.npz")
        feats.append(z["feats"]); stems += [str(s) for s in z["stems"]]
    return np.concatenate(feats), stems


def score(root, cells, seeds, n_est, note) -> None:
    for tier in TIERS:
        tdir = root / f"tier_{tier}"
        tdir.mkdir(parents=True, exist_ok=True)
        res_path = tdir / "results.json"
        results = json.loads(res_path.read_text()) if res_path.exists() else []
        done = {r["cell"] for r in results}
        if all(c["name"] in done for c in cells):
            continue
        Xcov, cov_stems = e16.load_set(E16_OUT, "cover", tier, FCHUNK)
        pos = {s: i for i, s in enumerate(cov_stems)}
        null = json.loads((E16_OUT / f"tier_{tier}" / "null_test.json").read_text())
        for c in cells:
            if c["name"] in done:
                continue
            Xs, stems = load_new(root, c["name"], tier)
            keep = [i for i, s in enumerate(stems) if s in pos]
            stems, Xs = [stems[i] for i in keep], Xs[keep]
            note(f"tier {tier}: {c['name']} ({len(stems)} pairs)")
            r, rows = e16.run_cell(Xcov[[pos[s] for s in stems]], Xs, stems, seeds,
                                   800, n_est, note)
            e16.write_scores(tdir / "scores" / f"{c['name']}.csv", rows)
            results.append(dict(cell=c["name"], method="SCRIPT", R=c.get("R", 10),
                                payload_bpnzac=c["payload"], detector="DCTR+FLD",
                                n_images=tier, null_p_e=null["p_e_mean"],
                                null_sd=null["p_e_std"], **r))
            res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            note(f"-> tier {tier} {c['name']} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f} "
                 f"(null {null['p_e_mean']:.4f} +/- {null['p_e_std']:.4f})")


def pool(root, note, B=20, n=3000) -> None:
    rng = np.random.default_rng(0)
    out = []
    tables = {
        "R=10": {0.01: root, 0.02: root, 0.05: E16_OUT, 0.1: E16_OUT, 0.2: E16_OUT},
        "R=0": {0.05: root, 0.1: root, 0.2: root},
    }
    for label, where in tables.items():
        for tier in TIERS:
            line = f"SCRIPT {label}, tier {tier}, batch of {B}, total payload 0.2:"
            found = False
            for k, p in ((1, 0.2), (2, 0.1), (4, 0.05), (10, 0.02), (20, 0.01)):
                if p not in where:
                    continue
                name = f"SCRIPT_{p}" if label == "R=10" else f"SCRIPT-R0_{p}"
                f = where[p] / f"tier_{tier}" / "scores" / f"{name}.csv"
                if not f.exists():
                    continue
                per_seed = [e17.pooled(*e17.batches(c, s, B, k, n, rng), k)
                            for c, s in e17.load(f).values()]
                avg = {r: float(np.mean([x[r] for x in per_seed])) for r in per_seed[0]}
                out.append(dict(setting=label, tier=tier, k=k, payload=p, **avg,
                                warden_best=min(avg.values())))
                line += f"  {k}x{p}: {min(avg.values()):.4f}"
                found = True
            if found:
                note(line)
    (root / "batch_pooling.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def pack(root, note, count=5000) -> None:
    """One-file dataset for the CNN warden: 5,000 covers + SCRIPT at 0.01, 0.02."""
    dest = root / "stego_script_low_5000.bin"
    if dest.exists():
        return
    tmp = root / "stego_script_low_5000.zip"
    manifest = dict(description="SCRIPT (R=10) at 0.01 and 0.02 bpnzac, first 5,000 BOSSbase images.",
                    cover_dir="cover", cover_count=count,
                    image_spec=dict(format="JPEG", size="512x512", channels=1,
                                    cover_quality_factor=e13.COVER_QF), cells=[])
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for i in range(1, count + 1):
            z.write(COVERS / f"{i}.jpg", f"dataset/cover/{i}.jpg")
        for c in CELLS_A:
            good, d, m = ok_set(root, c["name"]), root / c["name"] / "stego", 0
            for i in range(1, count + 1):
                if str(i) in good and (d / f"{i}.jpg").exists():
                    z.write(d / f"{i}.jpg", f"dataset/stego/{c['name']}/{i}.jpg"); m += 1
            manifest["cells"].append(dict(cell=c["name"], arm="B", method="SCRIPT",
                                          payload_bpnzac=c["payload"], channel="QF65->QF65",
                                          stego_dir=f"stego/{c['name']}", count=m))
        z.writestr("dataset/MANIFEST.json", json.dumps(manifest, indent=2))
    with dest.open("wb") as fo, tmp.open("rb") as fi:
        # A header and a non-archive extension keep Kaggle from unpacking it
        # into thousands of loose files (which stalls sessions for hours).
        fo.write(b"STEGO-EVAL packed dataset; open with zipfile.".ljust(64, b" "))
        shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)
    tmp.unlink()
    note(f"packed {dest} ({dest.stat().st_size / 1e9:.2f} GB), cells "
         + ", ".join(f"{c['cell']}:{c['count']}" for c in manifest["cells"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-est", type=int, default=200)
    args = ap.parse_args()
    root = paths.ensure_results("exp21_script_low_fullscale")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    covers = sorted(COVERS.resolve().glob("*.jpg"),
                    key=lambda p: (len(p.stem), p.stem))[:args.count]
    note(f"START (restructured): {len(covers)} covers; phase A = R10 at 0.01/0.02, "
         f"phase B = R0 at 0.05/0.1/0.2")
    for label, cells, do_pack in (("A", CELLS_A, True), ("B", CELLS_B, False)):
        reuse(root, cells, args.chunk, note)
        embed(root, cells, covers, args.chunk, args.workers, note)
        if do_pack:
            pack(root, note)
        features(root, cells, args.count, note)
        score(root, cells, args.seeds, args.n_est, note)
        pool(root, note)
        note(f"PHASE {label} done")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
