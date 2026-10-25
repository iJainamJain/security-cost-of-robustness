"""Paper 2: is the CNN a stronger batch warden than DCTR on spread SCRIPT payloads?

exp21 phase A measured the batch-of-20 table with a DCTR + FLD warden. This
repeats it on the same 5,000 images with the EfficientNet-B0 scores that came
back from Kaggle (M4_script_low_cnn for 0.01 and 0.02, the 5,000-image grid for
0.05 and 0.1; the CNN was never trained at 0.2). No training here: it pools
per-image test scores that already exist.

The two detectors do not see the same amount of training data: DCTR trains on
2,500 pairs (50/50 split), the CNN on 2,000 (40/10/50, epoch chosen on
validation). Both are tested on 2,500 pairs.

    python experiments/exp22_cnn_warden_pooling.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp17_batch_pooling as e17  # noqa: E402

R = paths.RESULTS
CNN = {0.01: R / "m4_script_low_cnn/scale_results/cnn/scores/SCRIPT_0.01.csv",
       0.02: R / "m4_script_low_cnn/scale_results/cnn/scores/SCRIPT_0.02.csv",
       0.05: R / "m4_5k_b/scale_results/cnn/scores/SCRIPT_0.05.csv",
       0.1: R / "m4_5k_b/scale_results/cnn/scores/SCRIPT_0.1.csv"}
DCTR = {0.01: R / "exp21_script_low_fullscale/tier_5000/scores/SCRIPT_0.01.csv",
        0.02: R / "exp21_script_low_fullscale/tier_5000/scores/SCRIPT_0.02.csv",
        0.05: R / "exp16_dctr_fullscale/tier_5000/scores/SCRIPT_0.05.csv",
        0.1: R / "exp16_dctr_fullscale/tier_5000/scores/SCRIPT_0.1.csv",
        0.2: R / "exp16_dctr_fullscale/tier_5000/scores/SCRIPT_0.2.csv"}
# Curriculum-trained CNN (M4_chain_cnn): 0.05 from scratch, then 0.02, then 0.01,
# four seeds, one split per seed shared by the three stages.
_CH = R / "m4_chain_cnn/chain_results/scores"
CHAIN = {0.01: _CH / "SCRIPT_0.01.csv", 0.02: _CH / "SCRIPT_0.02.csv",
         0.05: _CH / "SCRIPT_0.05.csv"}
B, N_BATCH = 20, 3000
STRATEGIES = ((1, 0.2), (2, 0.1), (4, 0.05), (10, 0.02), (20, 0.01))


def table(files: dict, rng) -> dict:
    out = {}
    for k, p in STRATEGIES:
        if p not in files or not files[p].exists():
            continue
        seeds = e17.load(files[p])
        single = [e17.p_error(c, s) for c, s in seeds.values()]
        per_seed = [e17.pooled(*e17.batches(c, s, B, k, N_BATCH, rng), k)
                    for c, s in seeds.values()]
        avg = {r: float(np.mean([x[r] for x in per_seed])) for r in per_seed[0]}
        out[p] = dict(k=k, payload=p, seeds=len(seeds), single=float(np.mean(single)),
                      **avg, warden_best=min(avg.values()))
    return out


def main() -> int:
    rng = np.random.default_rng(0)
    res = {"DCTR+FLD": table(DCTR, rng), "EfficientNet-B0": table(CNN, rng),
           "EffNet curriculum": table(CHAIN, rng)}
    # chance level for the pooled statistic: cover batches against cover batches
    c, _ = next(iter(e17.load(DCTR[0.01]).values()))
    null = [min(e17.pooled(*e17.batches(c, c, B, 20, N_BATCH, rng), 20).values())
            for _ in range(20)]
    res["batch_chance"] = dict(mean=float(np.mean(null)), std=float(np.std(null)))

    print(f"SCRIPT R=10, 5,000 images, batch of {B}, total payload 0.2")
    print(f"batch-level chance (best of three rules, cover vs cover): "
          f"{np.mean(null):.4f} +/- {np.std(null):.4f}\n")
    print(f"{'strategy':<12}{'detector':<18}{'single':>8}{'mean':>8}{'max':>8}"
          f"{'top-k':>8}{'best':>8}")
    for k, p in STRATEGIES:
        for det in ("DCTR+FLD", "EfficientNet-B0", "EffNet curriculum"):
            r = res[det].get(p)
            if r:
                print(f"{f'{k} x {p}':<12}{det:<18}{r['single']:>8.4f}{r['mean']:>8.4f}"
                      f"{r['max']:>8.4f}{r['top-k']:>8.4f}{r['warden_best']:>8.4f}")
    out = paths.ensure_results("exp22_cnn_warden_pooling")
    (out / "cnn_warden_pooling.json").write_text(
        json.dumps({d: (list(v.values()) if d != "batch_chance" else v)
                    for d, v in res.items()}, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'cnn_warden_pooling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
