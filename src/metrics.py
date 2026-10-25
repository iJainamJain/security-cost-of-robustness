"""Evaluation metrics.

Two families, and conflating them is the conceptual error this project exists to
correct (RESEARCH_DOSSIER.md §II.3):

* **Perceptual** — PSNR, SSIM. How visible the embedding is to a human.
* **Security** — P_E under a trained detector. How detectable it is to a warden.

They are close to uncorrelated in the regime that matters: StegoTool's LSB mode
scores 57 dB PSNR and is read perfectly by a 2001 detector. Report perceptual
metrics as secondary and always label them as such.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "psnr", "ssim", "bpp", "bpnzac",
    "p_error", "min_p_error", "md_at_fa",
    "pool_mean", "pool_max", "pooled_p_error",
]


# --------------------------------------------------------------------------
# Perceptual (secondary — never report these as security)
# --------------------------------------------------------------------------

def psnr(cover: np.ndarray, stego: np.ndarray, peak: float = 255.0) -> float:
    mse = np.mean((cover.astype(np.float64) - stego.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(peak ** 2 / mse))


def ssim(cover: np.ndarray, stego: np.ndarray, peak: float = 255.0) -> float:
    """Global SSIM with the standard 11x11 Gaussian window (sigma=1.5).

    Implemented directly to avoid a scikit-image dependency on the Kaggle side.
    Operates on a single channel; average across channels for colour.
    """
    a = cover.astype(np.float64)
    b = stego.astype(np.float64)
    if a.ndim == 3:
        return float(np.mean([ssim(a[..., i], b[..., i], peak) for i in range(a.shape[2])]))

    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2

    r = 5
    x = np.arange(-r, r + 1)
    g = np.exp(-(x ** 2) / (2 * 1.5 ** 2))
    g /= g.sum()

    def blur(m: np.ndarray) -> np.ndarray:
        m = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 0, m)
        return np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 1, m)

    mu_a, mu_b = blur(a), blur(b)
    saa = blur(a * a) - mu_a ** 2
    sbb = blur(b * b) - mu_b ** 2
    sab = blur(a * b) - mu_a * mu_b

    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)
    return float(np.mean(num / den))


# --------------------------------------------------------------------------
# Capacity
# --------------------------------------------------------------------------

def bpp(payload_bytes: int, height: int, width: int) -> float:
    """Bits per pixel — the spatial-domain capacity unit."""
    return payload_bytes * 8.0 / (height * width)


def bpnzac(payload_bytes: int, n_nonzero_ac: int) -> float:
    """Bits per non-zero AC coefficient — the JPEG-domain capacity unit.

    This is the unit the robust-steganography literature reports (0.05-0.5), so
    any comparison against GMAS/ROAST/Cheng et al. must use it rather than bpp.
    """
    if n_nonzero_ac <= 0:
        raise ValueError("n_nonzero_ac must be positive")
    return payload_bytes * 8.0 / n_nonzero_ac


# --------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------

def p_error(p_fa: float, p_md: float) -> float:
    """Total classification error under equal priors."""
    return 0.5 * (p_fa + p_md)


def min_p_error(scores_cover: Sequence[float], scores_stego: Sequence[float]) -> float:
    """P_E = min over thresholds of 1/2(P_FA + P_MD).

    This is the field's standard security metric. Higher is more secure; 0.5 is
    a detector doing no better than chance. Published robust-steganography
    methods land at 0.05-0.09 against SRNet, i.e. almost completely broken.

    ``scores_*`` are detector outputs where larger means "more likely stego".
    """
    c = np.asarray(scores_cover, dtype=np.float64)
    s = np.asarray(scores_stego, dtype=np.float64)
    if c.size == 0 or s.size == 0:
        raise ValueError("need both cover and stego scores")

    # Sweep every midpoint between observed scores; the optimum is always there.
    thresholds = np.unique(np.concatenate([c, s]))
    best = 1.0
    for t in thresholds:
        p_fa = float(np.mean(c >= t))   # cover called stego
        p_md = float(np.mean(s < t))    # stego called cover
        best = min(best, 0.5 * (p_fa + p_md))
    return best


def md_at_fa(scores_cover: Sequence[float], scores_stego: Sequence[float],
             target_fa: float = 0.01) -> float:
    """Missed-detection rate at a fixed false-alarm rate.

    P_E assumes equal priors, which flatters the steganographer in a deployment
    where covers vastly outnumber stegos. A low-FA operating point is the more
    realistic warden, so report both.
    """
    c = np.asarray(scores_cover, dtype=np.float64)
    s = np.asarray(scores_stego, dtype=np.float64)
    t = float(np.quantile(c, 1.0 - target_fa))
    return float(np.mean(s < t))


# --------------------------------------------------------------------------
# Pooled steganalysis (batch setting — dossier Part V, L5)
# --------------------------------------------------------------------------

def pool_mean(scores: np.ndarray) -> np.ndarray:
    """Average per-object evidence over each batch. Rows are batches."""
    return np.asarray(scores, dtype=np.float64).mean(axis=1)


def pool_max(scores: np.ndarray) -> np.ndarray:
    """Flag a batch on its single most suspicious object.

    Ker's result: against max-pooling the steganographer should *concentrate*
    payload, whereas against sum/mean-pooling they should spread. RRS's sqrt(N)
    gain therefore is not unconditional, and both poolers must be reported.
    """
    return np.asarray(scores, dtype=np.float64).max(axis=1)


def pooled_p_error(cover_batches: np.ndarray, stego_batches: np.ndarray,
                   pooler=pool_mean) -> float:
    """P_E of a warden who sees whole batches rather than single objects.

    ``*_batches`` are (n_batches, batch_size) arrays of per-object scores.
    """
    return min_p_error(pooler(cover_batches), pooler(stego_batches))
