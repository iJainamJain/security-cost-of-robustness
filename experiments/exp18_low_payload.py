"""Paper 2, decisive test: does robust embedding become undetectable as the
per-cover payload goes to zero, or does it keep a fixed footprint?

exp17 showed that spreading 0.2 bpnzac over four covers at 0.05 each is not
nearly enough: a single robust stego image at 0.05 is already detected. Whether
spreading can EVER work depends on what happens below 0.05. If single-image P_E
climbs to chance as payload falls, many covers at very low payload can hide a
message. If it plateaus below chance, the method changes the cover in a way that
does not depend on the message, and no allocation rescues it.

Stages (each resumable):
  embed     GMAS, ROAST, SCRIPT at 0.01 and 0.02 bpnzac, first 2,000 images
  features  DCTR (Python port) for the six new cells
  score     FLD ensemble, five paired splits, for payloads 0.01 ... 0.2, all
            with features from the SAME extractor (exp16 supplies 0.05-0.2)
  pool      batch-level P_E at equal total payload 0.2 over a batch of 20:
            20 x 0.01, 10 x 0.02, 4 x 0.05, 2 x 0.1, 1 x 0.2

By default it waits for exp16 (the full-scale DCTR run) to finish, so the two
do not fight over this laptop's cores and memory.

    python experiments/exp18_low_payload.py --pilot     # 8 images, embed only
    python experiments/exp18_low_payload.py             # the real thing
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp16_dctr_fullscale as e16  # noqa: E402
import exp17_batch_pooling as e17  # noqa: E402
from exp13_fullscale import COVER_QF, Job, csv_rows, find_matlab, row_ok  # noqa: E402

E16_OUT = paths.RESULTS / "exp16_dctr_fullscale"
COVERS = paths.DATA / f"BOSSbase_QF{COVER_QF}_full"
NEW_PAYLOADS = (0.01, 0.02)
OLD_PAYLOADS = (0.05, 0.1, 0.2)
METHODS = {"GMAS": "gmas", "ROAST": "roast", "SCRIPT": "script"}
REFERENCE = ("J-UNIWARD",)          # scored at 0.05-0.2 only, for comparison


def new_cells() -> list[dict]:
    return [dict(name=f"{m}_{p}", kind=k, payload=p)
            for p in NEW_PAYLOADS for m, k in METHODS.items()]


def wait_for_exp16(note) -> None:
    # Wait for exp16's FEATURE EXTRACTION only, not its scoring. Extraction is
    # what saturates the cores and memory; the scoring that follows is one
    # process and takes many hours at 10,000 images, and this test matters more
    # than finishing that a little sooner. Everything needed from exp16 (the
    # first 2,000 images' features) exists as soon as extraction is done.
    pid_file = E16_OUT / "driver.pid"
    while True:
        d, n = e16.tier_extracted(E16_OUT, [5000, 10000], 500, 1)
        if d == n:
            note("exp16 feature extraction is complete; starting")
            return
        pid = pid_file.read_text().strip() if pid_file.exists() else ""
        alive = pid and pid in subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        if not alive:
            note("exp16 driver is not running (and not finished) - laptop is free; starting")
            return
        time.sleep(120)


def embed(root, covers, chunk, workers, note) -> None:
    n_chunks = (len(covers) + chunk - 1) // chunk
    jobs = [Job(root, c, k, covers, chunk) for k in range(n_chunks) for c in new_cells()]
    todo = [j for j in jobs if not j.done()]
    note(f"embed: {len(jobs) - len(todo)} chunks done, {len(todo)} to do")
    matlab = find_matlab()
    running, limit = [], 40.0 * chunk
    while todo or running:
        while todo and len(running) < workers:
            j = todo.pop(0); j.start(matlab); running.append(j)
        time.sleep(5)
        for j in list(running):
            rc = j.proc.poll()
            if rc is None:
                if time.time() - j.t0 > limit:
                    j.proc.kill(); running.remove(j)
                    note(f"TIMEOUT {j.label}")
                continue
            running.remove(j)
            ok, msg = j.finish()
            note(("done " if ok else f"FAILED (exit {rc}) ") + f"{j.label}  {msg}  "
                 f"{time.time() - j.t0:.0f}s | {len(todo) + len(running)} left")


def ok_stems(root, name, count) -> list[str]:
    good = set()
    for f in sorted((root / name / "csv").glob("chunk_*.csv")):
        for r in csv_rows(f):
            s = (r.get("cover") or "").strip()
            if s.isdigit() and int(s) <= count and row_ok(r):
                good.add(s)
    d = root / name / "stego"
    return sorted((s for s in good if (d / f"{s}.jpg").exists()
                   and (d / f"{s}.jpg").stat().st_size > 0), key=int)


def features(root, count, note) -> None:
    import torch

    from dctr import DCTR, DIM, read_jpeg
    torch.set_num_threads(6)
    ex = DCTR(COVER_QF, "cpu")
    for c in new_cells():
        dest = root / "features" / f"{c['name']}.npz"
        if dest.exists():
            continue
        stems = ok_stems(root, c["name"], count)
        feats = np.empty((len(stems), DIM), dtype=np.float32)
        t0 = time.time()
        for i, s in enumerate(stems):
            feats[i] = ex.features(*read_jpeg(root / c["name"] / "stego" / f"{s}.jpg"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, feats=feats, stems=np.array(stems))
        note(f"features {c['name']}: {len(stems)} images, "
             f"{(time.time() - t0) / max(len(stems), 1):.2f} s/img")


def score(root, count, seeds, n_est, note) -> None:
    Xcov, cov_stems = e16.load_set(E16_OUT, "cover", count, 500)
    Xcov, cov_stems = Xcov[:count], cov_stems[:count]
    pos = {s: i for i, s in enumerate(cov_stems)}

    null_path = root / "null_test.json"
    if not null_path.exists():
        perm = np.random.default_rng(20261002).permutation(len(cov_stems))
        half = len(cov_stems) // 2
        ia, ib = np.sort(perm[:half]), np.sort(perm[half:2 * half])
        note("NULL TEST (cover vs cover, random groups)")
        r, rows = e16.run_cell(Xcov[ia], Xcov[ib], [cov_stems[i] for i in ia],
                               seeds, 800, n_est, note)
        e16.write_scores(root / "scores" / "NULL_TEST.csv", rows)
        null_path.write_text(json.dumps(r, indent=2), encoding="utf-8")
        note(f"NULL P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}")

    res_path = root / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["cell"] for r in results}
    todo = [(m, p) for m in METHODS for p in NEW_PAYLOADS + OLD_PAYLOADS]
    todo += [(m, p) for m in REFERENCE for p in OLD_PAYLOADS]
    for m, p in todo:
        name = f"{m}_{p}"
        if name in done:
            continue
        if p in NEW_PAYLOADS:
            z = np.load(root / "features" / f"{name}.npz")
            Xs, stems = z["feats"], [str(s) for s in z["stems"]]
        else:
            Xs, stems = e16.load_set(E16_OUT, name, count, 500)
        keep = [i for i, s in enumerate(stems) if s in pos]
        stems = [stems[i] for i in keep]
        Xs = Xs[keep]
        note(f"{name} ({len(stems)} pairs)")
        r, rows = e16.run_cell(Xcov[[pos[s] for s in stems]], Xs, stems,
                               seeds, 800, n_est, note)
        e16.write_scores(root / "scores" / f"{name}.csv", rows)
        results.append(dict(cell=name, method=m, payload_bpnzac=p,
                            detector="DCTR+FLD", n_images=count, **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        note(f"-> {name} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}")


def pool(root, note, n=3000, B=20) -> None:
    rng = np.random.default_rng(0)
    res = {r["cell"]: r for r in json.loads((root / "results.json").read_text())}
    null = json.loads((root / "null_test.json").read_text())["p_e_mean"]
    out = []
    note(f"single-image P_E by payload (null {null:.4f}):")
    for m in list(METHODS) + list(REFERENCE):
        ps = [p for p in NEW_PAYLOADS + OLD_PAYLOADS if f"{m}_{p}" in res]
        note("  " + m.ljust(10) + "  ".join(f"{p}: {res[f'{m}_{p}']['p_e_mean']:.4f}" for p in ps))
    note(f"batch of {B}, total payload 0.2, warden's best pooling rule:")
    for m in list(METHODS) + list(REFERENCE):
        line = "  " + m.ljust(10)
        for k, p in ((1, 0.2), (2, 0.1), (4, 0.05), (10, 0.02), (20, 0.01)):
            f = root / "scores" / f"{m}_{p}.csv"
            if not f.exists():
                continue
            per_seed = [e17.pooled(*e17.batches(c, s, B, k, n, rng), k)
                        for c, s in e17.load(f).values()]
            avg = {r: float(np.mean([x[r] for x in per_seed])) for r in per_seed[0]}
            best = min(avg.values())
            out.append(dict(method=m, k=k, payload=p, batch=B, **avg, warden_best=best))
            line += f"  {k}x{p}: {best:.4f}"
        note(line)
    (root / "batch_pooling.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-est", type=int, default=200)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="8 images, 2 workers, embedding only, separate folder")
    args = ap.parse_args()
    if args.pilot:
        args.count, args.chunk, args.workers = 8, 8, 2
    root = paths.ensure_results("exp18_low_payload_pilot" if args.pilot
                                else "exp18_low_payload")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    covers = sorted(COVERS.resolve().glob("*.jpg"),
                    key=lambda p: (len(p.stem), p.stem))[:args.count]
    if not (args.pilot or args.no_wait):
        note("waiting for exp16 to finish...")
        wait_for_exp16(note)
    note(f"START low-payload test: {len(covers)} covers, payloads {NEW_PAYLOADS}")
    embed(root, covers, args.chunk, args.workers, note)
    if args.pilot:
        for c in new_cells():
            rows = [r for f in sorted((root / c["name"] / "csv").glob("*.csv")) for r in csv_rows(f)]
            ok = sum(row_ok(r) for r in rows)
            bits = [r.get("raw_msg_bits") or r.get("msg_bits") for r in rows]
            note(f"pilot {c['name']}: {ok}/{len(rows)} embedded, message bits {bits}")
        return 0
    features(root, args.count, note)
    score(root, args.count, args.seeds, args.n_est, note)
    pool(root, note)
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
