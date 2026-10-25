"""RS steganalysis (Fridrich, Goljan & Du 2001) — deliberately order-independent.

Used here to test the claim in PROJECT_UNDERSTANDING.md that password-seeded
pixel shuffling defeats statistical LSB detectors. RS analysis groups pixels
spatially and never assumes a sequential embedding path, so if the claim were
true the estimator would still have to read ~0 on a shuffled-LSB stego image.
"""
import numpy as np


def _flip1(a):
    """F1: 0<->1, 2<->3, ... (LSB flip)."""
    return a ^ 1


def _flipm1(a):
    """F-1: -1<->0, 1<->2, ... i.e. F1(x+1)-1."""
    a = a.astype(np.int16)
    return (((a + 1) ^ 1) - 1).clip(0, 255).astype(np.uint8)


def _rs_counts(chan, mask):
    """Return (R, S) fractions for a 1x4 group mask over the channel."""
    h, w = chan.shape
    w4 = (w // 4) * 4
    g = chan[:, :w4].reshape(h, w4 // 4, 4).astype(np.int16)

    flipped = g.copy()
    m = np.asarray(mask)
    for i in range(4):
        if m[i] == 1:
            flipped[:, :, i] = _flip1(g[:, :, i].astype(np.uint8))
        elif m[i] == -1:
            flipped[:, :, i] = _flipm1(g[:, :, i].astype(np.uint8))

    f_orig = np.abs(np.diff(g, axis=2)).sum(axis=2)
    f_flip = np.abs(np.diff(flipped, axis=2)).sum(axis=2)

    total = f_orig.size
    R = np.count_nonzero(f_flip > f_orig) / total
    S = np.count_nonzero(f_flip < f_orig) / total
    return R, S


def rs_estimate(chan, mask=(0, 1, 1, 0)):
    """Estimate LSB-replacement embedding rate p in [0,1] for one channel."""
    mask = np.asarray(mask)
    nmask = -mask

    R_m, S_m = _rs_counts(chan, mask)
    R_nm, S_nm = _rs_counts(chan, nmask)

    chan1 = _flip1(chan)  # corresponds to p = 1
    R_m1, S_m1 = _rs_counts(chan1, mask)
    R_nm1, S_nm1 = _rs_counts(chan1, nmask)

    d0 = R_m - S_m
    dm0 = R_nm - S_nm
    d1 = R_m1 - S_m1
    dm1 = R_nm1 - S_nm1

    a = 2 * (d1 + d0)
    b = dm0 - dm1 - d1 - 3 * d0
    c = d0 - dm0

    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return 0.0
        x = -c / b
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return float("nan")
        r = np.sqrt(disc)
        x1, x2 = (-b + r) / (2 * a), (-b - r) / (2 * a)
        x = x1 if abs(x1) <= abs(x2) else x2

    if abs(x - 0.5) < 1e-12:
        return float("nan")
    return float(x / (x - 0.5))


def rs_estimate_rgb(arr):
    """Mean RS estimate across R,G,B planes."""
    if arr.ndim == 2:
        return rs_estimate(arr)
    return float(np.mean([rs_estimate(arr[:, :, i]) for i in range(arr.shape[2])]))
