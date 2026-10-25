"""Data for three figures that no table of the paper already shows.

1. Where SCRIPT's zero-cost coefficients sit, by DCT mode, for both kinds:
   kind A, where the compressor and an exact DCT disagree (|e| > 0.5), and
   kind B, exact rounding ties (|e| = 0.5). exp45 kept the per-mode rate of
   kind A only; this keeps both, as counts per image.
2. ROC curves of DCTR + FLD on the released and the repaired SCRIPT at strength
   0, on 10,000 images (five paired splits, averaged at fixed false-alarm rates).
3. The paired-permutation null distributions of exp44 themselves, not only
   their mean and spread, for the cells the text reads against chance.

No new embedding or training; reads covers, precovers and saved score files.

    python experiments/exp55_figure_data.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp34_script_ablation as e34  # noqa: E402
import exp44_paired_permutation as e44  # noqa: E402

R = paths.RESULTS
PGM = paths.DATA / "BOSSbase_1.01"
N_MODES = 300
B = 2000
FPR = np.linspace(0, 1, 401)


def modes() -> dict:
    import jpeglib
    from PIL import Image
    from scipy.fft import dctn

    a = np.zeros((8, 8))
    b = np.zeros((8, 8))
    for i in range(1, N_MODES + 1):
        im = jpeglib.read_dct(str(e34.COVERS / f"{i}.jpg"))
        c, qt = np.asarray(im.Y).astype(float), np.asarray(im.qt[0]).astype(float)
        x = np.asarray(Image.open(PGM / f"{i}.pgm"), float) - 128
        h, w = x.shape
        blk = x.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
        real = dctn(blk, axes=(2, 3), norm="ortho") / qt
        e = np.abs(real - c)
        a += (e > 0.5 + 1e-9).sum((0, 1))
        b += (np.abs(e - 0.5) < 1e-9).sum((0, 1))
    return dict(images=N_MODES, kind_A_per_image=(a / N_MODES).tolist(),
                kind_B_per_image=(b / N_MODES).tolist(),
                kind_A_total=float(a.sum() / N_MODES), kind_B_total=float(b.sum() / N_MODES))


def roc(cover: np.ndarray, stego: np.ndarray) -> np.ndarray:
    """True-positive rate at the false-alarm rates FPR (stego expected higher)."""
    th = np.unique(np.concatenate([cover, stego]))[::-1]
    fa = np.r_[0, [(cover >= t).mean() for t in th]]
    tp = np.r_[0, [(stego >= t).mean() for t in th]]
    return np.interp(FPR, fa, tp)


def rocs() -> dict:
    files = {"released": (R / "exp21_script_low_fullscale" / "tier_10000" / "scores", "SCRIPT-R0_{}.csv"),
             "repaired": (R / "exp40_script_fix" / "tier_10000" / "scores", "SCRIPT-fix3_{}.csv")}
    out = {"fpr": FPR.tolist()}
    for name, (d, pat) in files.items():
        for p in ("0.05", "0.1", "0.2"):
            per = e44.read(d / pat.format(p))
            curves = [roc(c, s) for c, s in per.values()]
            pes = [e44.p_e(c, s) for c, s in per.values()]
            out[f"{name}_{p}"] = dict(tpr=np.mean(curves, 0).tolist(), p_e=float(np.mean(pes)),
                                      tpr_at_1pct=float(np.mean([np.interp(0.01, FPR, t) for t in curves])))
            print(f"ROC {name} {p}: P_E {np.mean(pes):.4f}, caught at 1% false alarm "
                  f"{100 * out[f'{name}_{p}']['tpr_at_1pct']:.1f}%")
    return out


def nulls() -> dict:
    rng = np.random.default_rng(20261021)
    cells = [
        ("J-UNIWARD 0.05, DCTR, 10,000 images",
         [R / "exp16_dctr_fullscale" / "tier_10000" / "scores" / "J-UNIWARD_0.05.csv"]),
        ("MINICER step 1 0.05, DCTR, 2,000 images",
         [R / "exp50_minicer_step_sweep" / "sweep_2000" / "scores" / "MINICER-s1_0.05.csv"]),
        ("SCRIPT repaired 0.05, DCTR, 10,000 images",
         [R / "exp40_script_fix" / "tier_10000" / "scores" / "SCRIPT-fix3_0.05.csv"]),
        ("SCRIPT repaired 0.05, SRNet, 5,000 images",
         [R / f"m4_srnet_{s}" / "srnet_results" / "scores" / "SCRIPT-fix_0.05.csv"
          for s in ("SFH01", "SFH2", "SF01", "SF2")]),
        ("No-selection reference repaired 0.05, DCTR, 5,000 images",
         [R / "exp43_si_reference_fix" / "tier2_5000" / "scores" / "SI-fix3_0.05.csv"]),
    ]
    out = {}
    for cell, fs in cells:
        runs = []
        for f in fs:
            for seed, (c, s) in sorted(e44.read(f).items()):
                null = np.empty(B)
                for k in range(B):
                    sw = rng.random(len(c)) < 0.5
                    null[k] = e44.p_e(np.where(sw, s, c), np.where(sw, c, s))
                obs = e44.p_e(c, s)
                runs.append(dict(source=f.parent.parent.parent.name if "srnet" in str(f) else f.parent.parent.name,
                                 seed=seed, pairs=len(c), p_e=obs, null=null.round(5).tolist(),
                                 p_value=float((1 + (null <= obs).sum()) / (B + 1))))
                print(f"{cell:<58} {seed} P_E {obs:.4f} null {null.mean():.4f} p {runs[-1]['p_value']:.4f}",
                      flush=True)
        out[cell] = runs
    return out


def main() -> int:
    out = paths.ensure_results("exp55_figure_data")
    res = dict(modes=modes(), roc=rocs(), nulls=nulls())
    m = res["modes"]
    print(f"kind A {m['kind_A_total']:.1f} and kind B {m['kind_B_total']:.1f} per image")
    (out / "results.json").write_text(json.dumps(res), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
