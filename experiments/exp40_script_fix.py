"""Can the fixed component of SCRIPT's detectability be removed?

F28: where the cover JPEG and an exact DCT of the precover round a coefficient
differently, the released cost code prices a change at exactly zero. SCRIPT at
strength 0 makes about 147 such changes per image at any payload, and they
account for most of its measured detectability (alone: P_E 0.43; the rest of
the stego image: 0.45 / 0.43 at 0.05 / 0.2, chance 0.47). That "rest" was an
analysis, not a decodable stego image. This experiment builds real ones.

Three repairs of the cost (baselines/patched/SCRIPT/costGISS.m, `si_fix`):
    1  no side-information discount where the two transforms disagree
    2  no change allowed there
    3  the discount factor floored at 0.1 everywhere
Everything else is SCRIPT at strength 0 as in the main tables.

Stages (each resumable):
  check   the shard with si_fix = 0 still reproduces exp21's stego files
  pilot   the three repairs, first 2,000 images, 0.05 and 0.2 bpnzac; delivery
          on the QF65 -> QF65 channel and DCTR + FLD (1,000 / 1,000 pairs, the
          setting of exp34, where the released code reads 0.397 / 0.378)
  choose  the repair with the highest mean P_E among those that deliver at
          least 99% of images; stop if none beats the released code by 0.01
  final   that repair at 0.05, 0.1, 0.2 on the first 5,000 images (10,000 with
          --tier 10000), DCTR as in Tables 1 and 3, and a dataset for SRNet

    python experiments/exp40_script_fix.py --smoke      # 8 images, no detector
    python experiments/exp40_script_fix.py
    python experiments/exp40_script_fix.py --tier 10000
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dctr_sets as ds  # noqa: E402
import exp13_fullscale as e13  # noqa: E402
import exp16_dctr_fullscale as e16  # noqa: E402
import exp21_script_low_fullscale as e21  # noqa: E402

COVERS = paths.DATA / f"BOSSbase_QF{e13.COVER_QF}_full"
E16_OUT = paths.RESULTS / "exp16_dctr_fullscale"
E21_OUT = paths.RESULTS / "exp21_script_low_fullscale"
VARIANTS = {1: "no discount where the transforms disagree", 2: "no change there (wet)",
            3: "discount factor floored at 0.1"}
PILOT_N, PILOT_P, FINAL_P = 2000, (0.05, 0.2), (0.05, 0.1, 0.2)
RELEASED_AT_2000 = {0.05: 0.3970, 0.2: 0.3781}      # exp34, same images and protocol

_prev_call = e13.matlab_call


def _call(cell, manifest, out_dir, csv_path, seed_base):
    if "fix" not in cell:
        return _prev_call(cell, manifest, out_dir, csv_path, seed_base)
    r, q = cell["R"], e13.COVER_QF
    return (f"script_embed_shard('{e13.REPO.as_posix()}','{manifest.as_posix()}',"
            f"'{out_dir.as_posix()}','{csv_path.as_posix()}',{cell['payload']},{q},{q},"
            f"{seed_base},{r},{r},0,0,{cell['fix']})")


e13.matlab_call = _call


def cell(v: int, p: float) -> dict:
    return dict(name=f"SCRIPT-fix{v}_{p}", kind="script", payload=p, R=0, fix=v)


def delivery(root: pathlib.Path, name: str, n: int) -> dict:
    rows = [r for f in sorted((root / name / "csv").glob("chunk_*.csv"))
            for r in e13.csv_rows(f) if (r.get("cover") or "").strip().isdigit()
            and int(r["cover"]) <= n]
    ok = [r for r in rows if e13.row_ok(r)]
    clean = sum(1 for r in ok if (r.get("bit_error_rate") or "") != ""
                and float(r["bit_error_rate"]) == 0)
    psnr = sorted(float(r["psnr_db"]) for r in ok if (r.get("psnr_db") or "") not in ("", "Inf"))
    return dict(images=len(rows), embedded=len(ok), error_free=clean,
                rate=clean / len(rows) if rows else 0.0,
                median_psnr_db=psnr[len(psnr) // 2] if psnr else None)


def stego_files(root: pathlib.Path, name: str, n: int) -> list[tuple[str, pathlib.Path]]:
    ok = e21.ok_set(root, name)
    return [(str(i), root / name / "stego" / f"{i}.jpg") for i in range(1, n + 1)
            if str(i) in ok]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 images per cell, no detector")
    ap.add_argument("--tier", type=int, default=5000, help="images for the final stage")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    root = paths.ensure_results("exp40_script_fix_smoke" if args.smoke else "exp40_script_fix")
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    def covers(n):
        return [COVERS / f"{i}.jpg" for i in range(1, n + 1)]

    note(f"START SCRIPT cost repair ({'smoke' if args.smoke else 'full'})")

    # ---- check: si_fix = 0 through the edited shard is still the released SCRIPT
    chk = dict(name="SCRIPT-R0check_0.05", kind="script", payload=0.05, R=0, fix=0)
    e21.embed(root, [chk], covers(8), 8, args.workers, note)
    same = sum(hashlib.md5((root / chk["name"] / "stego" / f"{i}.jpg").read_bytes()).hexdigest()
               == hashlib.md5((E21_OUT / "SCRIPT-R0_0.05" / "stego" / f"{i}.jpg").read_bytes()).hexdigest()
               for i in range(1, 9))
    note(f"check: shard with si_fix = 0 reproduces exp21's stego files in {same} of 8")
    if same != 8:
        note("STOP - the edited shard changed the released behaviour")
        return 1

    # ---- pilot
    n, chunk = (8, 8) if args.smoke else (PILOT_N, args.chunk)
    pilot = [cell(v, p) for p in PILOT_P for v in VARIANTS]
    e21.embed(root, pilot, covers(n), chunk, args.workers, note)
    note(f"{'cell':<20}{'images':>8}{'embedded':>10}{'error-free':>12}{'PSNR':>8}")
    deliv = {}
    for c in pilot:
        d = deliv[c["name"]] = delivery(root, c["name"], n)
        note(f"{c['name']:<20}{d['images']:>8}{d['embedded']:>10}{d['error_free']:>12}"
             f"{(d['median_psnr_db'] or 0):>8.1f}")
    if args.smoke:
        note("END (smoke)")
        return 0

    fdir = root / "features"
    Xcov, cov_stems = e16.load_set(E16_OUT, "cover", PILOT_N, ds.FCHUNK)
    for c in pilot:
        ds.extract(fdir, c["name"], stego_files(root, c["name"], PILOT_N), e13.COVER_QF, note)
    res = ds.score(root / "pilot_2000", Xcov, cov_stems,
                   [(c["name"], dict(variant=c["fix"], payload_bpnzac=c["payload"],
                                     n_images=PILOT_N),
                     lambda c=c: ds.load(fdir, c["name"], PILOT_N // ds.FCHUNK)) for c in pilot],
                   note)
    pe = {r["cell"]: r["p_e_mean"] for r in res}
    note("")
    note(f"{'repair':<46}" + "".join(f"{'P_E ' + str(p):>10}{'delivered':>11}" for p in PILOT_P))
    note(f"{'released code (exp34, exp21)':<46}"
         + "".join(f"{RELEASED_AT_2000[p]:>10.4f}{'99.9%':>11}" for p in PILOT_P))
    score_of = {}
    for v, text in VARIANTS.items():
        names = [cell(v, p)["name"] for p in PILOT_P]
        note(f"{str(v) + '  ' + text:<46}"
             + "".join(f"{pe[nm]:>10.4f}{100 * deliv[nm]['rate']:>10.1f}%" for nm in names))
        if all(deliv[nm]["rate"] >= 0.99 for nm in names):
            score_of[v] = sum(pe[nm] for nm in names) / len(names)
    base = sum(RELEASED_AT_2000.values()) / len(RELEASED_AT_2000)
    best = max(score_of, key=score_of.get) if score_of else None
    decision = dict(mean_p_e_released=base, mean_p_e_by_repair=score_of, chosen=best,
                    proceeds=bool(best is not None and score_of[best] - base >= 0.01))
    (root / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    if not decision["proceeds"]:
        note(f"DECISION: no repair that keeps delivery beats the released code by 0.01 "
             f"(released {base:.4f}, repairs {score_of}) - the final stage is not run")
        note("END")
        return 0
    note(f"DECISION: repair {best} ({VARIANTS[best]}): mean P_E {score_of[best]:.4f} against "
         f"{base:.4f} for the released code - going on to {args.tier} images")

    # ---- final: the chosen repair at three payloads
    final = [cell(best, p) for p in FINAL_P]
    for tier in sorted({5000, args.tier}):
        e21.embed(root, final, covers(tier), args.chunk, args.workers, note)
        Xcov, cov_stems = e16.load_set(E16_OUT, "cover", tier, ds.FCHUNK)
        for c in final:
            ds.extract(fdir, c["name"], stego_files(root, c["name"], tier), e13.COVER_QF, note)
        res = ds.score(root / f"tier_{tier}", Xcov, cov_stems,
                       [(c["name"], dict(variant=best, payload_bpnzac=c["payload"], n_images=tier),
                         lambda c=c, tier=tier: ds.load(fdir, c["name"], tier // ds.FCHUNK))
                        for c in final], note)
        pe = {r["cell"]: r for r in res}
        ref = {r["cell"]: r["p_e_mean"] for r in json.loads(
            (E21_OUT / f"tier_{tier}" / "results.json").read_text())}
        note("")
        note(f"{tier} images, DCTR + FLD    repaired   released   gain     delivered (repaired)")
        for c in final:
            d = delivery(root, c["name"], tier)
            a, b = pe[c["name"]]["p_e_mean"], ref[f"SCRIPT-R0_{c['payload']}"]
            note(f"  {c['payload']:<22}{a:>9.4f}{b:>11.4f}{a - b:>8.4f}{100 * d['rate']:>12.2f}%")

    # ---- dataset for SRNet (first 5,000 images; generic cell names)
    out = paths.ensure_results("exp40_script_fix_5000")
    dest = out / "stego_SCRIPT-fix_5000.bin"
    if not dest.exists():
        tmp = out / "stego_SCRIPT-fix_5000.zip"
        manifest = dict(
            description=f"SCRIPT at strength 0 with the cost repair {best} "
                        f"({VARIANTS[best]}); covers included.",
            cover_dir="cover", cover_count=5000, cells=[],
            image_spec=dict(format="JPEG", size="512x512", channels=1,
                            cover_quality_factor=e13.COVER_QF),
            pairing="stego/<cell>/N.jpg is the same source image as cover/N.jpg.",
            source="experiments/exp40_script_fix.py, seeds = global image index")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for i in range(1, 5001):
                z.write(COVERS / f"{i}.jpg", f"dataset/cover/{i}.jpg")
            for c in final:
                files = stego_files(root, c["name"], 5000)
                for s, f in files:
                    z.write(f, f"dataset/stego/SCRIPT-fix_{c['payload']}/{s}.jpg")
                manifest["cells"].append(dict(
                    cell=f"SCRIPT-fix_{c['payload']}", arm="B",
                    method=f"SCRIPT (R=0, cost repair {best})", payload_bpnzac=c["payload"],
                    channel="QF65->QF65", stego_dir=f"stego/SCRIPT-fix_{c['payload']}",
                    count=len(files)))
            z.writestr("dataset/MANIFEST.json", json.dumps(manifest, indent=2))
        with dest.open("wb") as fo, tmp.open("rb") as fi:
            fo.write(b"STEGO-EVAL packed dataset; open with zipfile.".ljust(64, b" "))
            shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)
        tmp.unlink()
        note(f"wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
