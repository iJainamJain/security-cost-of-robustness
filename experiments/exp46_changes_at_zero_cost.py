"""Which zero-cost coefficients does SCRIPT change, under each cost variant?

exp45: two kinds of coefficient have a side-informed discount of zero in the
released cost (factor 1 - 2|e| = 0):

  A  |e| > 0.5: the cover's compressor and an exact DCT round differently
     (about 330 per image, spread over low-frequency modes, |e| - 0.5 < 0.02)
  B  |e| = 0.5 exactly: the exact DCT lands on the rounding boundary, which
     happens only in modes (0,0), (0,4), (4,0), (4,4) (about 84 per image) -
     the known pathology of side-informed costs

exp35 and F28 counted only A. Repairs 1 and 2 of exp40 act on A only; repair 3
(a floor on the factor) acts on both. This counts the changes each variant
makes at A, at B and elsewhere, on the first 300 images at 0.05 and 0.2 bpnzac.
Counts only; no embedding, no detector.

    python experiments/exp46_changes_at_zero_cost.py
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

PGM = paths.DATA / "BOSSbase_1.01"
N = 300
R = paths.RESULTS
VARIANTS = {
    "released": lambda p: R / "exp21_script_low_fullscale" / f"SCRIPT-R0_{p}" / "stego",
    "repair 1 (no discount at A)": lambda p: R / "exp40_script_fix" / f"SCRIPT-fix1_{p}" / "stego",
    "repair 2 (no change at A)": lambda p: R / "exp40_script_fix" / f"SCRIPT-fix2_{p}" / "stego",
    "repair 3 (floor 0.1)": lambda p: R / "exp40_script_fix" / f"SCRIPT-fix3_{p}" / "stego",
}


def main() -> int:
    import jpeglib
    from PIL import Image
    from scipy.fft import dctn

    acc = {(v, p): [] for v in VARIANTS for p in (0.05, 0.2)}
    sizes = []
    for i in range(1, N + 1):
        im = jpeglib.read_dct(str(e34.COVERS / f"{i}.jpg"))
        c, qt = np.asarray(im.Y).astype(np.int32), np.asarray(im.qt[0]).astype(float)
        x = np.asarray(Image.open(PGM / f"{i}.pgm"), float) - 128
        h, w = x.shape
        b = x.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
        real = dctn(b, axes=(2, 3), norm="ortho") / qt
        ae = np.abs(real - c)
        B = np.abs(ae - 0.5) < 1e-9
        A = (ae > 0.5) & ~B
        dc = np.zeros_like(A)
        dc[:, :, 0, 0] = True
        sizes.append([int(A.sum()), int(B.sum()), int((B & dc).sum())])
        for v, where in VARIANTS.items():
            for p in (0.05, 0.2):
                f = where(p) / f"{i}.jpg"
                if not f.exists():
                    continue
                s = np.asarray(jpeglib.read_dct(str(f)).Y).astype(np.int32)
                ch = s != c
                acc[(v, p)].append([int(ch.sum()), int((ch & A).sum()), int((ch & B).sum()),
                                    int((ch & B & dc).sum()), int((ch & ~A & ~B).sum())])
    sz = np.mean(sizes, 0)
    print(f"per image: {sz[0]:.0f} coefficients of kind A, {sz[1]:.0f} of kind B "
          f"({sz[2]:.0f} of them DC)")
    print(f"{'variant':<30}{'payload':>8}{'images':>8}{'changes':>9}{'at A':>8}{'at B':>8}"
          f"{'(B, DC)':>9}{'elsewhere':>11}")
    out = []
    for (v, p), rows in acc.items():
        if not rows:
            continue
        m = np.mean(rows, 0)
        out.append(dict(variant=v, payload=p, images=len(rows), changes=float(m[0]), at_A=float(m[1]),
                        at_B=float(m[2]), at_B_dc=float(m[3]), elsewhere=float(m[4])))
        print(f"{v:<30}{p:>8}{len(rows):>8}{m[0]:>9.0f}{m[1]:>8.1f}{m[2]:>8.1f}{m[3]:>9.1f}{m[4]:>11.0f}")
    d = paths.ensure_results("exp46_changes_at_zero_cost")
    (d / "results.json").write_text(json.dumps(dict(
        kind_A_per_image=float(sz[0]), kind_B_per_image=float(sz[1]), kind_B_dc_per_image=float(sz[2]),
        rows=out), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
