"""Batch security with SRNet as the warden: SCRIPT at strength 0, and J-UNIWARD.

exp21 measured the batch-of-20 table for SCRIPT at R = 0 with a DCTR warden and
found that concentrating the payload beats spreading it. This repeats it with
the SRNet scores of the curriculum grid (sessions E and D2F for SCRIPT at
R = 0; A and B for J-UNIWARD) and with the tuned DCTR ensemble (exp26), on the
same 5,000 images. No training here: it pools per-image test scores that
already exist, exactly as exp22 does.

    python experiments/exp31_srnet_warden_pooling.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp17_batch_pooling as e17  # noqa: E402

R = paths.RESULTS
B, N_BATCH = 20, 3000
STRATEGIES = ((1, 0.2), (2, 0.1), (4, 0.05))
METHODS = {"SCRIPT, R = 0": "SCRIPT-R0", "J-UNIWARD": "J-UNIWARD"}


def srnet_scores(cell: str, out: pathlib.Path) -> pathlib.Path | None:
    """Merge one cell's score files over the 512x512 sessions (one seed each)."""
    rows, seeds = [], set()
    for f in sorted(R.glob(f"m4_srnet_*/srnet_results/scores/{cell}.csv")):
        if not re.fullmatch(r"m4_srnet_[A-F][0-9A-Z]*", f.parts[-4]):
            continue
        with f.open(newline="") as fh:
            for r in csv.DictReader(fh):
                if (f.parts[-4], r["seed"]) not in seeds and r["seed"] in {s for _, s in seeds}:
                    continue                      # a seed already taken from another session
                seeds.add((f.parts[-4], r["seed"]))
                rows.append(r)
    if not rows:
        return None
    dest = out / "scores" / f"{cell}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["seed", "image", "label", "score"])
        w.writeheader()
        w.writerows(rows)
    return dest


def table(files: dict, rng) -> list[dict]:
    out = []
    for k, p in STRATEGIES:
        f = files.get(p)
        if f is None or not f.exists():
            continue
        seeds = e17.load(f)
        single = [e17.p_error(c, s) for c, s in seeds.values()]
        per_seed = [e17.pooled(*e17.batches(c, s, B, k, N_BATCH, rng), k)
                    for c, s in seeds.values()]
        avg = {r: float(np.mean([x[r] for x in per_seed])) for r in per_seed[0]}
        out.append(dict(k=k, payload=p, seeds=len(seeds), single=float(np.mean(single)),
                        **avg, warden_best=min(avg.values()),
                        best_rule=min(avg, key=avg.get)))
    return out


def main() -> int:
    out = paths.ensure_results("exp31_srnet_warden_pooling")
    rng = np.random.default_rng(0)
    res = {}
    for label, prefix in METHODS.items():
        tuned = {p: R / f"exp26_tuned_ensemble/scores/{prefix}_{p}.csv" for _, p in STRATEGIES}
        srnet = {p: srnet_scores(f"{prefix}_{p}", out) for _, p in STRATEGIES}
        res[label] = {"DCTR + FLD, tuned": table(tuned, rng), "SRNet": table(srnet, rng)}
    c, _ = next(iter(e17.load(R / "exp26_tuned_ensemble/scores/J-UNIWARD_0.05.csv").values()))
    null = [min(e17.pooled(*e17.batches(c, c, B, 4, N_BATCH, rng), 4).values())
            for _ in range(20)]
    res["batch_chance"] = dict(mean=float(np.mean(null)), std=float(np.std(null)))

    print(f"5,000 images, batch of {B}, total payload 0.2; batch-level chance "
          f"{np.mean(null):.4f} +/- {np.std(null):.4f}\n")
    print(f"{'method':<16}{'strategy':<10}{'detector':<20}{'seeds':>6}{'single':>8}{'mean':>8}"
          f"{'max':>8}{'top-k':>8}{'best':>8}")
    for label in METHODS:
        for i, (k, p) in enumerate(STRATEGIES):
            for det, rows in res[label].items():
                r = next((x for x in rows if x["payload"] == p), None)
                if r:
                    print(f"{label:<16}{f'{k} x {p}':<10}{det:<20}{r['seeds']:>6}"
                          f"{r['single']:>8.4f}{r['mean']:>8.4f}{r['max']:>8.4f}"
                          f"{r['top-k']:>8.4f}{r['warden_best']:>8.4f}")
    (out / "srnet_warden_pooling.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'srnet_warden_pooling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
