"""Higher payloads of the repaired SCRIPT, so that a network can be given a fair
chance to detect it.

exp40: with the discount floor, SCRIPT at strength 0 reads P_E 0.490 / 0.483 /
0.459 at 0.05 / 0.1 / 0.2 bpnzac under DCTR (null 0.484). The SRNet runs on it
(M4_srnet_SF01, SF2) started their curriculum at 0.2 bpnzac, and the training
loss sat at ln 2 - the network learned nothing. That is what security looks
like, and it is also what a network that fails to start looks like (EfficientNet
on J-UNIWARD, SRNet on SCRIPT at 256x256). DCTR does see a little at 0.2, so
the null result from SRNet cannot be taken as a measurement.

The standard answer is a longer curriculum: train where the signal is clear and
carry the weights down. This embeds the repaired SCRIPT at 0.3, 0.4 and 0.6
bpnzac on the first 5,000 images, measures them with DCTR (to see where the
repair becomes detectable at all), and packs one dataset holding 0.6 ... 0.05
for the chain  0.6 -> 0.4 -> 0.3 -> 0.2 -> 0.1 -> 0.05.

    python experiments/exp42_script_fix_curriculum.py --smoke
    python experiments/exp42_script_fix_curriculum.py
"""
from __future__ import annotations

import argparse
import datetime as dt
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
import exp40_script_fix as e40  # noqa: E402  (installs the si_fix call)

HIGH = (0.3, 0.4, 0.6)
CHAIN = (0.6, 0.4, 0.3, 0.2, 0.1, 0.05)
N = 5000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="8 images per cell, no detector")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    main_root = paths.RESULTS / "exp40_script_fix"
    best = json.loads((main_root / "decision.json").read_text())["chosen"]
    root = paths.ensure_results("exp42_smoke") if args.smoke else main_root
    runlog = (root / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    n, chunk = (8, 8) if args.smoke else (N, args.chunk)
    cells = [e40.cell(best, p) for p in HIGH]
    note(f"START higher payloads of repair {best}: {HIGH} bpnzac, {n} images")
    e21.embed(root, cells, [e40.COVERS / f"{i}.jpg" for i in range(1, n + 1)], chunk,
              args.workers, note)
    note(f"{'cell':<20}{'images':>8}{'embedded':>10}{'error-free':>12}{'PSNR':>8}")
    for c in cells:
        d = e40.delivery(root, c["name"], n)
        note(f"{c['name']:<20}{d['images']:>8}{d['embedded']:>10}{d['error_free']:>12}"
             f"{(d['median_psnr_db'] or 0):>8.1f}")
    if args.smoke:
        note("END (smoke)")
        return 0

    fdir = root / "features"
    Xcov, cov_stems = e16.load_set(e40.E16_OUT, "cover", N, ds.FCHUNK)
    for c in cells:
        ds.extract(fdir, c["name"], e40.stego_files(root, c["name"], N), e13.COVER_QF, note)
    res = ds.score(root / "tier_5000_high", Xcov, cov_stems,
                   [(c["name"], dict(variant=best, payload_bpnzac=c["payload"], n_images=N),
                     lambda c=c: ds.load(fdir, c["name"], N // ds.FCHUNK)) for c in cells], note)
    low = {r["cell"]: r["p_e_mean"] for r in json.loads(
        (root / "tier_5000" / "results.json").read_text())}
    ju = {r["cell"]: r["p_e_mean"] for r in json.loads(
        (e40.E16_OUT / "tier_5000" / "results.json").read_text())}
    note("")
    note("repaired SCRIPT, 5,000 images, DCTR + FLD (null 0.484)      J-UNIWARD")
    for p in sorted(set(HIGH) | {0.05, 0.1, 0.2}):
        v = low.get(f"SCRIPT-fix{best}_{p}") or next(
            (r["p_e_mean"] for r in res if r["cell"] == f"SCRIPT-fix{best}_{p}"), float("nan"))
        note(f"  {p:<8}{v:>10.4f}{ju.get(f'J-UNIWARD_{p}', float('nan')):>40.4f}")

    out = paths.ensure_results("exp42_script_fix_curriculum_5000")
    dest = out / "stego_SCRIPT-fix-curriculum_5000.bin"
    if not dest.exists():
        tmp = dest.with_suffix(".zip")
        manifest = dict(
            description=f"SCRIPT at strength 0 with cost repair {best} "
                        f"({e40.VARIANTS[best]}), payloads {CHAIN}; covers included.",
            cover_dir="cover", cover_count=N, cells=[],
            image_spec=dict(format="JPEG", size="512x512", channels=1,
                            cover_quality_factor=e13.COVER_QF),
            pairing="stego/<cell>/N.jpg is the same source image as cover/N.jpg.",
            source="experiments/exp40_script_fix.py and exp42_script_fix_curriculum.py")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for i in range(1, N + 1):
                z.write(e40.COVERS / f"{i}.jpg", f"dataset/cover/{i}.jpg")
            for p in CHAIN:
                files = e40.stego_files(root, f"SCRIPT-fix{best}_{p}", N)
                for s, f in files:
                    z.write(f, f"dataset/stego/SCRIPT-fix_{p}/{s}.jpg")
                manifest["cells"].append(dict(
                    cell=f"SCRIPT-fix_{p}", arm="B", method=f"SCRIPT (R=0, cost repair {best})",
                    payload_bpnzac=p, channel="QF65->QF65", stego_dir=f"stego/SCRIPT-fix_{p}",
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
