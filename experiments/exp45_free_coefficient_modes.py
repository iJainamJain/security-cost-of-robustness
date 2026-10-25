"""Where are SCRIPT's zero-cost coefficients, and how far past the boundary?

F28: about 320 coefficients per image have |e| > 0.5 (e = exact DCT of the
precover / q - cover coefficient), where the released cost is exactly zero.
Two questions a reviewer asked:

  1. Is this the known side-information problem at e = 1/2 in DCT modes (0,0),
     (0,4), (4,0), (4,4), where the DCT of integer pixels lands exactly on a
     rounding boundary? Then the zero-cost coefficients would sit in those four
     modes.
  2. Or is it the compressor's integer transform disagreeing with an exact DCT
     near a boundary, in any mode? Then they would spread over modes roughly as
     the coefficients near a boundary do, and |e| would exceed 0.5 only slightly.

Counts only, first 300 images, no detector.

    python experiments/exp45_free_coefficient_modes.py
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
SPECIAL = [(0, 0), (0, 4), (4, 0), (4, 4)]


def main() -> int:
    import jpeglib
    from PIL import Image
    from scipy.fft import dctn

    per_mode = np.zeros((8, 8))
    exact_half = np.zeros((8, 8))
    total = np.zeros((8, 8))
    excess = []
    n_free = []
    for i in range(1, N + 1):
        im = jpeglib.read_dct(str(e34.COVERS / f"{i}.jpg"))
        c, qt = np.asarray(im.Y).astype(float), np.asarray(im.qt[0]).astype(float)
        x = np.asarray(Image.open(PGM / f"{i}.pgm"), float) - 128
        h, w = x.shape
        b = x.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)
        real = dctn(b, axes=(2, 3), norm="ortho") / qt
        e = real - c
        free = np.abs(e) > 0.5
        per_mode += free.sum((0, 1))
        exact_half += (np.abs(np.abs(real - np.floor(real)) - 0.5) < 1e-9).sum((0, 1))
        total += np.ones_like(e).sum((0, 1))
        excess.append(np.abs(e)[free] - 0.5)
        n_free.append(int(free.sum()))
    excess = np.concatenate(excess)
    share = per_mode / per_mode.sum()
    spec = sum(per_mode[m] for m in SPECIAL)
    out = paths.ensure_results("exp45_free_coefficient_modes")
    res = dict(
        images=N, free_per_image_mean=float(np.mean(n_free)),
        share_in_modes_00_04_40_44=float(spec / per_mode.sum()),
        share_dc=float(per_mode[0, 0] / per_mode.sum()),
        exact_half_per_image=float(exact_half.sum() / N),
        exact_half_in_special_modes_per_image=float(sum(exact_half[m] for m in SPECIAL) / N),
        free_rate_per_mode=(per_mode / total).tolist(), share_per_mode=share.tolist(),
        excess_quantiles={q: float(np.quantile(excess, q)) for q in (0.5, 0.9, 0.99, 1.0)},
        share_excess_below_0_05=float((excess < 0.05).mean()))
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"zero-cost coefficients per image: {res['free_per_image_mean']:.0f}")
    print(f"share of them in modes (0,0),(0,4),(4,0),(4,4): {100 * res['share_in_modes_00_04_40_44']:.1f}% "
          f"(4 of 64 modes = 6.3%); DC alone {100 * res['share_dc']:.1f}%")
    print(f"coefficients whose exact DCT / q lies exactly on a boundary: "
          f"{res['exact_half_per_image']:.1f} per image "
          f"({res['exact_half_in_special_modes_per_image']:.1f} in the four special modes)")
    print("how far past the boundary |e| - 0.5 is: median %.4f, 90%% %.4f, 99%% %.4f, max %.4f; "
          "%.0f%% below 0.05" % (*[res["excess_quantiles"][q] for q in (0.5, 0.9, 0.99, 1.0)],
                                 100 * res["share_excess_below_0_05"]))
    print("share of zero-cost coefficients by mode (%), rows = vertical frequency:")
    for r in share:
        print("  " + " ".join(f"{100 * v:5.1f}" for v in r))
    print("rate per coefficient by mode (per 10,000):")
    for r in per_mode / total:
        print("  " + " ".join(f"{1e4 * v:5.1f}" for v in r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
