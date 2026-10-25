"""Three cheap follow-up tests, queued behind exp18 (which waits for exp16).

T1  Fixed footprint.  GMAS, ROAST and SCRIPT with a one-bit message
    (payload 1e-5 bpnzac). Whatever the detector finds there does not depend on
    the message: it is the method's own mark on the cover.

T3  Strength knob.  SCRIPT with T_gamma = T_lambda = 0 and 5 instead of the 10
    used so far, on its own same-QF channel. Our six-image search already
    showed R = 0 is error-free there and costs 6.6 dB less distortion. If that
    holds at 2,000 images, (a) embedding strength can be traded against channel
    severity - the premise of a channel-adaptive scheme - and (b) the SCRIPT
    numbers in Paper 1 were measured at a needlessly strong setting and must be
    redone.

T2  Cover selection.  No embedding. Using the 10,000-image scores (exp16) and
    the embedders' per-image logs (exp13): do covers with more non-zero AC
    coefficients carry the same relative payload more safely or more reliably?

Stages are resumable. T1 and T3 share one embed -> features -> score pass on
the first 2,000 images; T2 is pure analysis.

    python experiments/exp19_followups.py --pilot    # 8 images, embed only
    python experiments/exp19_followups.py            # waits for exp18, then runs
"""
from __future__ import annotations

import argparse
import csv
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
import exp13_fullscale as e13  # noqa: E402
import exp16_dctr_fullscale as e16  # noqa: E402

E13_OUT = paths.RESULTS / "exp13_fullscale"
E16_OUT = paths.RESULTS / "exp16_dctr_fullscale"
E18_OUT = paths.RESULTS / "exp18_low_payload"
COVERS = paths.DATA / f"BOSSbase_QF{e13.COVER_QF}_full"
ZERO = 1e-5                      # ceil(1e-5 * nzAC) = 1 message bit

CELLS = (
    # SCRIPT fails on a one-bit message for some covers (pilot: 3 of 8), so its
    # near-empty cell carries 0.001 bpnzac instead - about 40 bits.
    [dict(name=f"{m}-zero", kind=k, payload=p0, test="T1", method=m)
     for m, k, p0 in (("GMAS", "gmas", ZERO), ("ROAST", "roast", ZERO),
                      ("SCRIPT", "script", 1e-3))]
    + [dict(name=f"SCRIPT-R{r}_{p}", kind="script", payload=p, R=r, test="T3",
            method="SCRIPT")
       for r in (0, 5) for p in (0.05, 0.1, 0.2)]
)

# exp13's call builder hard-wires SCRIPT's R to 10; honour a per-cell value.
_orig_call = e13.matlab_call


def _call(cell, manifest, out_dir, csv_path, seed_base):
    if cell.get("R") is None:
        return _orig_call(cell, manifest, out_dir, csv_path, seed_base)
    r, q = cell["R"], e13.COVER_QF
    return (f"script_embed_shard('{e13.REPO.as_posix()}','{manifest.as_posix()}',"
            f"'{out_dir.as_posix()}','{csv_path.as_posix()}',{cell['payload']},{q},{q},"
            f"{seed_base},{r},{r},0,0)")


e13.matlab_call = _call


def wait_for_exp18(note) -> None:
    log, pid_file = E18_OUT / "run.log", E18_OUT / "driver.pid"
    while True:
        if log.exists() and "  END" in log.read_text(encoding="utf-8", errors="replace"):
            note("exp18 has finished; starting"); return
        pid = pid_file.read_text().strip() if pid_file.exists() else ""
        alive = pid and pid in subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        if not alive:
            note("exp18 driver is not running - starting anyway"); return
        time.sleep(120)


def embed(root, covers, chunk, workers, note) -> None:
    n_chunks = (len(covers) + chunk - 1) // chunk
    jobs = [e13.Job(root, c, k, covers, chunk) for k in range(n_chunks) for c in CELLS]
    todo = [j for j in jobs if not j.done()]
    note(f"embed: {len(jobs) - len(todo)} chunks done, {len(todo)} to do")
    matlab, running, limit = e13.find_matlab(), [], 40.0 * chunk
    while todo or running:
        while todo and len(running) < workers:
            j = todo.pop(0); j.start(matlab); running.append(j)
        time.sleep(5)
        for j in list(running):
            rc = j.proc.poll()
            if rc is None:
                if time.time() - j.t0 > limit:
                    j.proc.kill(); running.remove(j); note(f"TIMEOUT {j.label}")
                continue
            running.remove(j)
            ok, msg = j.finish()
            note(("done " if ok else f"FAILED (exit {rc}) ") + f"{j.label}  {msg}  "
                 f"{time.time() - j.t0:.0f}s | {len(todo) + len(running)} left")


def cell_rows(root, name, count) -> list[dict]:
    rows = []
    for f in sorted((root / name / "csv").glob("chunk_*.csv")):
        rows += [r for r in e13.csv_rows(f)
                 if (r.get("cover") or "").strip().isdigit() and int(r["cover"]) <= count]
    return rows


def features(root, count, note) -> None:
    import torch

    from dctr import DCTR, DIM, read_jpeg
    torch.set_num_threads(6)
    ex = DCTR(e13.COVER_QF, "cpu")
    for c in CELLS:
        dest = root / "features" / f"{c['name']}.npz"
        if dest.exists():
            continue
        d = root / c["name"] / "stego"
        stems = sorted((r["cover"].strip() for r in cell_rows(root, c["name"], count)
                        if e13.row_ok(r) and (d / f"{r['cover'].strip()}.jpg").exists()
                        and (d / f"{r['cover'].strip()}.jpg").stat().st_size > 0), key=int)
        feats = np.empty((len(stems), DIM), dtype=np.float32)
        for i, s in enumerate(stems):
            feats[i] = ex.features(*read_jpeg(d / f"{s}.jpg"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, feats=feats, stems=np.array(stems))
        note(f"features {c['name']}: {len(stems)} images")


def score(root, count, seeds, n_est, note) -> None:
    Xcov, cov_stems = e16.load_set(E16_OUT, "cover", count, 500)
    pos = {s: i for i, s in enumerate(cov_stems)}
    res_path = root / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["cell"] for r in results}
    # R = 10 references, same images, same extractor (exp16 features).
    todo = [(c["name"], None, c) for c in CELLS]
    todo += [(f"SCRIPT_{p}", "exp16", dict(test="T3-ref", method="SCRIPT", payload=p, R=10))
             for p in (0.05, 0.1, 0.2)]
    for name, src, meta in todo:
        if name in done:
            continue
        if src == "exp16":
            Xs, stems = e16.load_set(E16_OUT, name, count, 500)
        else:
            z = np.load(root / "features" / f"{name}.npz")
            Xs, stems = z["feats"], [str(s) for s in z["stems"]]
        keep = [i for i, s in enumerate(stems) if s in pos]
        stems, Xs = [stems[i] for i in keep], Xs[keep]
        note(f"{name} ({len(stems)} pairs)")
        r, rows = e16.run_cell(Xcov[[pos[s] for s in stems]], Xs, stems, seeds, 800,
                               n_est, note)
        e16.write_scores(root / "scores" / f"{name}.csv", rows)
        rob = {}
        if src is None:
            rr = [x for x in cell_rows(root, name, count) if e13.row_ok(x)]
            b = [float(x["bit_error_rate"]) for x in rr if (x.get("bit_error_rate") or "") != ""]
            ps = [float(x["psnr_db"]) for x in rr if (x.get("psnr_db") or "") != ""]
            rob = dict(embedded=len(rr), errorless=int(sum(v == 0 for v in b)),
                       mean_ber=float(np.mean(b)) if b else None,
                       mean_psnr=float(np.mean(ps)) if ps else None)
        results.append(dict(cell=name, test=meta["test"], method=meta["method"],
                            payload_bpnzac=meta["payload"], R=meta.get("R"),
                            detector="DCTR+FLD", n_images=count, **rob, **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        note(f"-> {name} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}")


def cover_selection(root, note) -> None:
    """T2: P_E and robustness by quintile of nzAC, from data already on disk."""
    tier = None
    for t in (10000, 5000):         # the largest tier that is fully scored
        rp = E16_OUT / f"tier_{t}" / "results.json"
        if rp.exists() and len(json.loads(rp.read_text())) >= 18:
            tier = t; break
    if tier is None:
        note("T2 skipped: no fully scored exp16 tier yet - rerun later"); return
    note(f"T2 uses the {tier}-image tier")
    sdir = E16_OUT / f"tier_{tier}" / "scores"
    nz = {}
    for f in sorted((E13_OUT / "GMAS_0.1" / "csv").glob("chunk_*.csv")):
        for r in e13.csv_rows(f):
            if (r.get("nzAC") or "").strip().isdigit():
                nz[r["cover"].strip()] = int(r["nzAC"])
    edges = np.quantile(list(nz.values()), [0.2, 0.4, 0.6, 0.8])
    q_of = {s: int(np.searchsorted(edges, v)) for s, v in nz.items()}
    out = []
    note("T2 cover selection - P_E within each nzAC quintile (Q1 = smoothest images)")
    note("   " + "cell".ljust(16) + "".join(f"Q{i + 1}".rjust(9) for i in range(5))
         + "   errorless Q1..Q5")
    for cell in ("GMAS_0.05", "GMAS_0.1", "ROAST_0.05", "ROAST_0.1", "SCRIPT_0.05",
                 "SCRIPT_0.1", "MINICER_0.05", "J-UNIWARD_0.2", "J-UNIWARD_0.3"):
        f = sdir / f"{cell}.csv"
        if not f.exists():
            continue
        sc = [[] for _ in range(5)]; ss = [[] for _ in range(5)]
        with f.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if r["image"] in q_of:
                    (ss if r["label"] == "stego" else sc)[q_of[r["image"]]].append(float(r["score"]))
        pe = [e16.p_error(np.array(sc[i]), np.array(ss[i])) for i in range(5)]
        okc = [0] * 5; tot = [0] * 5
        for g in sorted((E13_OUT / cell / "csv").glob("chunk_*.csv")):
            for r in e13.csv_rows(g):
                s = (r.get("cover") or "").strip()
                b = r.get("bit_error_rate") or ""
                if s in q_of and e13.row_ok(r) and b != "":
                    tot[q_of[s]] += 1; okc[q_of[s]] += float(b) == 0
        rate = [okc[i] / tot[i] if tot[i] else float("nan") for i in range(5)]
        out.append(dict(cell=cell, p_e_by_quintile=pe, errorless_by_quintile=rate))
        note("   " + cell.ljust(16) + "".join(f"{v:9.4f}" for v in pe) + "   "
             + " ".join(f"{v:.2f}" for v in rate))
    note("   (same detector for every quintile: the warden is NOT retrained on the "
         "selected covers, so this is an optimistic view for the sender)")
    (root / "cover_selection.json").write_text(json.dumps(
        dict(nzAC_quintile_edges=[float(e) for e in edges], cells=out), indent=2),
        encoding="utf-8")


def report(root, note) -> None:
    res = {r["cell"]: r for r in json.loads((root / "results.json").read_text())}
    note("T1 fixed footprint - near-empty message, single-image P_E (chance ~0.475):")
    for m in ("GMAS", "ROAST", "SCRIPT"):
        r = res.get(f"{m}-zero")
        if r:
            note(f"   {m:<8} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}   "
                 f"PSNR {r['mean_psnr']:.1f} dB")
    note("T3 SCRIPT strength on its own channel (QF65 -> QF65):")
    note("   R   payload     P_E      errorless        PSNR")
    for p in (0.05, 0.1, 0.2):
        for R, name in ((0, f"SCRIPT-R0_{p}"), (5, f"SCRIPT-R5_{p}"), (10, f"SCRIPT_{p}")):
            r = res.get(name)
            if not r:
                continue
            rob = (f"{r['errorless']}/{r['embedded']}".rjust(12) + f"{r['mean_psnr']:10.1f} dB"
                   if r.get("embedded") else "   (see exp13)".rjust(12))
            note(f"   {R:<3} {p:<8} {r['p_e_mean']:8.4f}   {rob}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-est", type=int, default=200)
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    if args.pilot:
        args.count, args.chunk, args.workers = 8, 8, 2
    root = paths.ensure_results("exp19_followups_pilot" if args.pilot else "exp19_followups")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    covers = sorted(COVERS.resolve().glob("*.jpg"),
                    key=lambda p: (len(p.stem), p.stem))[:args.count]
    if not (args.pilot or args.no_wait):
        note("waiting for exp18 to finish...")
        wait_for_exp18(note)
    note(f"START follow-ups: {len(CELLS)} new cells on {len(covers)} covers")
    embed(root, covers, args.chunk, args.workers, note)
    if args.pilot:
        for c in CELLS:
            rr = cell_rows(root, c["name"], args.count)
            ok = [r for r in rr if e13.row_ok(r)]
            bits = [r.get("raw_msg_bits") or r.get("msg_bits") for r in rr][:4]
            ber = [r.get("bit_error_rate") for r in ok][:4]
            psnr = [r.get("psnr_db") for r in ok][:3]
            note(f"pilot {c['name']:<16} {len(ok)}/{len(rr)} embedded  bits {bits}  "
                 f"BER {ber}  PSNR {psnr}")
        return 0
    features(root, args.count, note)
    score(root, args.count, args.seeds, args.n_est, note)
    report(root, note)
    cover_selection(root, note)
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
