"""Dataset for SRNet on the like-for-like reference of the repaired SCRIPT.

Table 9 (the repaired SCRIPT against its own embedder without the coefficient
selection) is DCTR only, and SRNet is the stronger detector of the repaired
scheme. The review council asked for SRNet on the reference. This embeds the one
payload the reference lacks for the chain of SFH (0.3 bpnzac; exp43 has 0.05,
0.1, 0.2, 0.4, 0.6), packs the six cells with the covers for Kaggle, and then
scores 0.3 with DCTR to complete the table.

    python experiments/exp54_si_fix_curriculum.py
"""
from __future__ import annotations

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
import exp40_script_fix as e40  # noqa: E402
import exp43_si_reference_fix as e43  # noqa: E402

N = 5000
CHAIN = (0.6, 0.4, 0.3, 0.2, 0.1, 0.05)
ROOT = paths.RESULTS / "exp43_si_reference_fix"


def main() -> int:
    best = json.loads((e43.E40_ROOT / "decision.json").read_text())["chosen"]
    runlog = (ROOT / "run.log").open("a", encoding="utf-8")

    def note(msg: str) -> None:
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True); runlog.write(line + "\n"); runlog.flush()

    c = e43.cell(best, 0.3)
    note("START exp54: reference at 0.3 bpnzac and the SRNet dataset")
    e21.embed(ROOT, [c], [e40.COVERS / f"{i}.jpg" for i in range(1, N + 1)], 250, 4, note)
    d = e40.delivery(ROOT, c["name"], N)
    note(f"{c['name']}: {d['embedded']} embedded, {d['error_free']} error-free of {d['images']}")

    out = paths.ensure_results("exp54_si_fix_curriculum_5000")
    dest = out / "stego_SI-fix-curriculum_5000.bin"
    if not dest.exists():
        tmp = out / "pack.zip"
        manifest = dict(
            description=f"SCRIPT's embedder with cost repair {best} ({e40.VARIANTS[best]}) and "
                        f"NO coefficient selection (R1 = R2 = -1), payloads {CHAIN}; covers included.",
            cover_dir="cover", cover_count=N, cells=[],
            image_spec=dict(format="JPEG", size="512x512", channels=1,
                            cover_quality_factor=e13.COVER_QF),
            pairing="stego/<cell>/N.jpg is the same source image as cover/N.jpg.",
            source="experiments/exp43_si_reference_fix.py and exp54_si_fix_curriculum.py")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for i in range(1, N + 1):
                z.write(e40.COVERS / f"{i}.jpg", f"dataset/cover/{i}.jpg")
            for p in CHAIN:
                files = e40.stego_files(ROOT, f"SI-fix{best}_{p}", N)
                if len(files) < 0.95 * N:
                    raise SystemExit(f"SI-fix{best}_{p}: only {len(files)} images")
                for s, f in files:
                    z.write(f, f"dataset/stego/SI-fix_{p}/{s}.jpg")
                manifest["cells"].append(dict(
                    cell=f"SI-fix_{p}", arm="B", method=f"SCRIPT embedder, no selection (cost repair {best})",
                    payload_bpnzac=p, channel="QF65->QF65", stego_dir=f"stego/SI-fix_{p}",
                    count=len(files)))
                note(f"  packed SI-fix_{p}: {len(files)} images")
            z.writestr("dataset/MANIFEST.json", json.dumps(manifest, indent=2))
        part = out / "pack.bin.part"
        with part.open("wb") as fo, tmp.open("rb") as fi:
            fo.write(b"STEGO-EVAL packed dataset; open with zipfile.".ljust(64, b" "))
            shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)
        tmp.unlink()
        part.replace(dest)
        note(f"wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB)")

    fdir = ROOT / "features"
    Xcov, cov_stems = e16.load_set(e40.E16_OUT, "cover", N, ds.FCHUNK)
    ds.extract(fdir, c["name"], e40.stego_files(ROOT, c["name"], N), e13.COVER_QF, note)
    res = ds.score(ROOT / "tier3_5000", Xcov, cov_stems,
                   [(c["name"], dict(variant=best, payload_bpnzac=0.3, n_images=N),
                     lambda: ds.load(fdir, c["name"], N // ds.FCHUNK))], note)
    spe = e43.script_pe(best).get(f"SCRIPT-fix{best}_0.3", float("nan"))
    ds_ = e40.delivery(e43.E40_ROOT, f"SCRIPT-fix{best}_0.3", N)
    r = res[0]
    note(f"0.3 bpnzac: no selection {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}, SCRIPT {spe:.4f}, "
         f"difference {r['p_e_mean'] - spe:.4f}; delivered {100 * d['rate']:.2f}% against "
         f"{100 * ds_['rate']:.2f}%")
    note("END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
