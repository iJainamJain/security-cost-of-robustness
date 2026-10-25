"""Is a P_E near chance a detection? A paired permutation test on saved scores.

Our chance levels (the cover-against-cover null of DCTR, and 0.491 for the
networks) are for two independent samples. Every real test set is paired: cover
i and its own stego. With a weak signal the two scores of a pair are strongly
correlated, the two empirical distributions track each other, and the
no-signal P_E is higher and much less variable than the independent-sample
value. Reading a paired P_E against an unpaired null is conservative for a
claim of detection and wrong for a claim of non-detection.

The exact test needs no retraining. Under "the detector cannot tell stego from
cover", swapping the two labels inside a pair leaves the joint distribution of
the test scores unchanged. So for each saved score file we recompute P_E under
random within-pair swaps, which gives the null distribution of P_E for that
trained detector on that test set, and a one-sided p-value for the observed
value. (The detector was trained on other images, so the test is valid
conditionally on it.)

No new embedding or training; reads the score files written by the DCTR and
SRNet runs.

    python experiments/exp44_paired_permutation.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

B = 2000
R = paths.RESULTS


def p_e(cover: np.ndarray, stego: np.ndarray) -> float:
    """min over thresholds of (P_FA + P_MD) / 2, stego expected to score higher."""
    s = np.concatenate([cover, stego])
    lab = np.concatenate([np.zeros(len(cover)), np.ones(len(stego))])
    order = np.argsort(s, kind="mergesort")
    s, lab = s[order], lab[order]
    below_c = np.cumsum(1 - lab) / len(cover)      # covers at or below the threshold
    below_s = np.cumsum(lab) / len(stego)          # stego at or below it (missed)
    last = np.r_[s[1:] != s[:-1], True]            # thresholds only between distinct scores
    err = 0.5 * ((1 - below_c[last]) + below_s[last])
    return float(min(err.min(), 0.5))


def read(path: pathlib.Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    by = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            by.setdefault(r["seed"], {}).setdefault(r["image"], {})[r["label"]] = float(r["score"])
    out = {}
    for seed, im in by.items():
        pairs = [(v["cover"], v["stego"]) for v in im.values() if len(v) == 2]
        a = np.array(pairs)
        out[seed] = (a[:, 0], a[:, 1])
    return out


def test(cover, stego, rng) -> dict:
    obs = p_e(cover, stego)
    null = np.empty(B)
    for b in range(B):
        swap = rng.random(len(cover)) < 0.5
        null[b] = p_e(np.where(swap, stego, cover), np.where(swap, cover, stego))
    return dict(pairs=len(cover), p_e=obs, null_mean=float(null.mean()), null_sd=float(null.std()),
                p_value=float((1 + (null <= obs).sum()) / (B + 1)),
                z=float((null.mean() - obs) / null.std()) if null.std() > 0 else float("nan"))


def main() -> int:
    out = paths.ensure_results("exp44_paired_permutation")
    rng = np.random.default_rng(20261011)
    sets = []                                     # (label, detector, score file)
    fix = R / "exp40_script_fix"
    for p in (0.05, 0.1, 0.2):
        sets.append((f"SCRIPT repaired {p}", "DCTR 5,000", fix / "tier_5000" / "scores" / f"SCRIPT-fix3_{p}.csv"))
        sets.append((f"SCRIPT repaired {p}", "DCTR 10,000", fix / "tier_10000" / "scores" / f"SCRIPT-fix3_{p}.csv"))
        for ses in ("SFH01", "SFH2"):
            sets.append((f"SCRIPT repaired {p}", "SRNet, chain from 0.6",
                         R / f"m4_srnet_{ses}" / "srnet_results" / "scores" / f"SCRIPT-fix_{p}.csv"))
        for ses in ("SF01", "SF2"):
            sets.append((f"SCRIPT repaired {p}", "SRNet, chain from 0.2",
                         R / f"m4_srnet_{ses}" / "srnet_results" / "scores" / f"SCRIPT-fix_{p}.csv"))
    for p in (0.05, 0.1, 0.2, 0.4, 0.6):
        sets.append((f"reference repaired {p}", "DCTR 5,000",
                     R / "exp43_si_reference_fix" / ("tier1_5000" if p >= 0.2 else "tier2_5000")
                     / "scores" / f"SI-fix3_{p}.csv"))
    for p in (0.05, 0.1):
        sets.append((f"J-UNIWARD {p}", "DCTR 5,000",
                     R / "exp16_dctr_fullscale" / "tier_5000" / "scores" / f"J-UNIWARD_{p}.csv"))
        sets.append((f"J-UNIWARD {p}", "DCTR 10,000",
                     R / "exp16_dctr_fullscale" / "tier_10000" / "scores" / f"J-UNIWARD_{p}.csv"))
    rows = []
    print(f"{'cell':<26}{'detector':<24}{'seed':>5}{'pairs':>7}{'P_E':>8}{'paired null':>13}"
          f"{'sd':>8}{'z':>7}{'p':>8}")
    for cell, det, f in sets:
        if not f.exists():
            print(f"{cell:<26}{det:<24}  (no score file: {f.name})")
            continue
        for seed, (c, s) in sorted(read(f).items()):
            r = dict(cell=cell, detector=det, seed=seed, **test(c, s, rng))
            rows.append(r)
            print(f"{cell:<26}{det:<24}{seed:>5}{r['pairs']:>7}{r['p_e']:>8.4f}{r['null_mean']:>13.4f}"
                  f"{r['null_sd']:>8.4f}{r['z']:>7.1f}{r['p_value']:>8.4f}", flush=True)
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print()
    print(f"{'cell':<26}{'detector':<24}{'runs':>5}{'mean P_E':>10}{'mean paired null':>18}"
          f"{'runs with p<0.01':>18}")
    keys = []
    for r in rows:
        if (r["cell"], r["detector"]) not in keys:
            keys.append((r["cell"], r["detector"]))
    for k in keys:
        g = [r for r in rows if (r["cell"], r["detector"]) == k]
        print(f"{k[0]:<26}{k[1]:<24}{len(g):>5}{np.mean([r['p_e'] for r in g]):>10.4f}"
              f"{np.mean([r['null_mean'] for r in g]):>18.4f}"
              f"{sum(r['p_value'] < 0.01 for r in g):>12} of {len(g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
