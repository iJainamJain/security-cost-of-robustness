"""DCTR features (Holub & Fridrich, 8000-dim) - a Python port of DCTR.m.

Ported line for line from the DDE Lab reference
(`baselines/features/DCTR_matlab_v1.1/matlab/DCTR.m`) so the extraction can run
where MATLAB cannot. A feature extractor that is subtly wrong is very hard to
notice and poisons every P_E built on it, so this port is only to be trusted
because `experiments/exp15_dctr_port_check.py` shows it reproduces the MATLAB
features on images we already extracted with the reference code.

Three places where a naive port goes wrong, all handled below:
  * MATLAB `round` rounds half away from zero; numpy/torch round half to even.
  * `conv2(..., 'valid')` is a true convolution - the kernel is flipped.
  * MATLAB indexes from 1, so shift (i, j) is offset (i-1, j-1) here.

Runs on CPU or GPU; float64 throughout, because the reference is double.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

T = 4
DIM = 8000


def quant_step(qf: int) -> float:
    if qf < 50:
        return min(8.0 * (50.0 / qf), 100.0)
    return max(8.0 * (2.0 - (qf / 50.0)), 0.2)


def _dct_matrix() -> np.ndarray:
    """A[x, u] = 0.5 * c(u) * cos((2x+1) u pi / 16): the matrix DCTR.m calls A."""
    k, l = np.meshgrid(np.arange(8), np.arange(8))
    a = 0.5 * np.cos(((2.0 * k + 1.0) * l * math.pi) / 16.0)
    a[0, :] = a[0, :] / math.sqrt(2.0)
    return a.T


def _merged_coordinates() -> list[list[tuple[int, int]]]:
    """The 25 groups of symmetric phase shifts, as 0-based offsets."""
    out = []
    for i in range(1, 6):
        for j in range(1, 6):
            cand = {(i, j), (i, 10 - j), (10 - i, j), (10 - i, 10 - j)}
            out.append(sorted((r - 1, c - 1) for r, c in cand if r < 9 and c < 9))
    return out


class DCTR:
    def __init__(self, qf: int, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.q = quant_step(qf)
        a = _dct_matrix()
        self.A = torch.tensor(a, dtype=torch.float64, device=self.device)
        # conv2 flips the kernel: tap a of the image meets element 7-a.
        self.Af = torch.flip(self.A, dims=[0])
        self._row_offset = (torch.arange(64 * 64, dtype=torch.int32,
                                         device=self.device)[:, None] * (T + 2))
        # 64 bases in DCTR.m's order: modeIndex = (mode_r-1)*8 + mode_c.
        bases = np.stack([np.outer(a[:, r], a[:, c])
                          for r in range(8) for c in range(8)])
        # conv2 'valid' flips the kernel; F.conv2d correlates. Flip to match.
        k = torch.tensor(bases[:, ::-1, ::-1].copy(), dtype=torch.float64,
                         device=self.device)
        self.kernels = k.unsqueeze(1)                       # (64, 1, 8, 8)
        # group[r, c] = which of the 25 merged groups phase (r, c) feeds.
        g = torch.zeros(8, 8, 25, dtype=torch.float64, device=self.device)
        for m, coords in enumerate(_merged_coordinates()):
            for r, c in coords:
                g[r, c, m] = 1.0
        self.group = g.reshape(64, 25)

    def spatial(self, coef_blocks: np.ndarray, qt: np.ndarray) -> torch.Tensor:
        """Decompress without rounding or clipping, minus the +128 offset.

        coef_blocks: (bh, bw, 8, 8) quantised DCT coefficients.
        Returns I_spatial - 128, which is what DCTR.m convolves.
        """
        d = torch.tensor(np.asarray(coef_blocks, dtype=np.float64)
                         * np.asarray(qt, dtype=np.float64),
                         device=self.device)
        x = self.A @ d @ self.A.T                           # blockwise idct2
        bh, bw = x.shape[:2]
        return x.permute(0, 2, 1, 3).reshape(bh * 8, bw * 8)

    @torch.no_grad()
    def features(self, coef_blocks: np.ndarray, qt: np.ndarray) -> np.ndarray:
        img = self.spatial(coef_blocks, qt)
        # The 64 DCT bases are separable (base = A[:, r] outer A[:, c]), so the
        # 2-D convolution is two 1-D ones: along y with the 8 row vectors, then
        # along x with the 8 column vectors. Same mathematics as
        # conv2(I, base, 'valid') at about a seventh of the arithmetic, and the
        # result comes out as (y, x, r, c) - already in DCTR.m's mode order
        # r*8 + c, with no copy needed. The summation ORDER differs from a
        # direct 2-D convolution, so a value landing exactly on a rounding
        # boundary can fall the other way; exp15 re-checks P_E against the
        # MATLAB reference for this reason.
        t1 = torch.tensordot(img.unfold(0, 8, 1), self.Af, dims=([2], [0]))  # (H-7, W, r)
        r = torch.tensordot(t1.unfold(1, 8, 1), self.Af, dims=([3], [0]))    # (H-7, W-7, r, c)
        h, w = r.shape[:2]

        # abs(round(R/q)) then threshold, in place and straight to uint8: the
        # same operations in the same order as floor(|R| / q + 0.5),
        # clamp(max=T), without four temporaries.
        r = r.abs_().div_(self.q).add_(0.5).floor_().clamp_(max=T).to(torch.uint8)

        # counts[mode, phase(8x8), value]: the histogram of
        # R(r_shift:8:end, c_shift:8:end) for all 64 phases at once. Pad to a
        # multiple of 8 with a sentinel no bin matches, regroup so each row is
        # one (mode, phase), and histogram all rows with a single bincount.
        # Integer counting: exactly what slicing phase by phase gives.
        ph, pw = -h % 8, -w % 8
        rp = F.pad(r.view(h, w, 64), (0, 0, 0, pw, 0, ph), value=T + 1)
        x = (rp.view((h + ph) // 8, 8, (w + pw) // 8, 8, 64)
               .permute(4, 1, 3, 0, 2).reshape(64 * 64, -1))
        key = x.to(torch.int32) + self._row_offset
        counts = (torch.bincount(key.view(-1), minlength=64 * 64 * (T + 2))
                  .view(64 * 64, T + 2)[:, :T + 1].reshape(64, 64, T + 1).double())
        merged = torch.einsum("mpv,pg->mgv", counts, self.group)   # (64, 25, 5)
        merged = merged / merged.sum(-1, keepdim=True)
        return merged.reshape(-1).cpu().numpy()


def read_jpeg(path) -> tuple[np.ndarray, np.ndarray]:
    """(coefficient blocks, quant table) of the luminance channel."""
    import jpeglib

    im = jpeglib.read_dct(str(path))
    return np.asarray(im.Y), np.asarray(im.qt[0])


def extract(paths, qf: int, device="cpu", progress: int = 0) -> np.ndarray:
    ex = DCTR(qf, device)
    out = np.empty((len(paths), DIM), dtype=np.float64)
    for i, p in enumerate(paths):
        out[i] = ex.features(*read_jpeg(p))
        if progress and (i + 1) % progress == 0:
            print(f"    {i + 1}/{len(paths)}", flush=True)
    return out
