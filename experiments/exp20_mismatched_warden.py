"""Stress test for exp18's good news: a warden who does NOT train at the
sender's tiny payload.

exp18 found that spreading 0.2 bpnzac over 20 covers at 0.01 each lifts batch
P_E to 0.30 (GMAS) and 0.43 (SCRIPT). But the detector there was trained on
0.01-payload images, where the signal is so weak that training itself may have
failed - in which case the sender looks safe only because the warden was inept.

A sharper warden trains where the signal is learnable (0.05 or 0.1 bpnzac) and
applies that detector to the low-payload images, then pools over the batch.
If that detector separates 0.01-payload stego from cover, the spreading gain
shrinks or vanishes. This measures it.

    python experiments/exp20_mismatched_warden.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp16_dctr_fullscale as e16  # noqa: E402
import exp17_batch_pooling as e17  # noqa: E402

E16_OUT = paths.RESULTS / "exp16_dctr_fullscale"
E18_OUT = paths.RESULTS / "exp18_low_payload"
COUNT, SEEDS, B, N_BATCH = 2000, 5, 20, 3000
TARGETS = {0.01: 20, 0.02: 10}          # per-cover payload -> covers used (total 0.2)
TRAIN_AT = (0.05, 0.1)


def load(method: str, p: float):
    if p in TARGETS:
        z = np.load(E18_OUT / "features" / f"{method}_{p}.npz")
        return z["feats"], [str(s) for s in z["stems"]]
    return e16.load_set(E16_OUT, f"{method}_{p}", COUNT, 500)


def main() -> int:
    out = paths.ensure_results("exp20_mismatched_warden")
    Xcov, cov_stems = e16.load_set(E16_OUT, "cover", COUNT, 500)
    matched = {r["cell"]: r for r in json.loads((E18_OUT / "results.json").read_text())}
    pooled_matched = {(r["method"], r["payload"]): r["warden_best"]
                      for r in json.loads((E18_OUT / "batch_pooling.json").read_text())}
    rng = np.random.default_rng(0)
    rows = []
    for method in ("SCRIPT", "GMAS", "ROAST"):
        sets = {p: load(method, p) for p in list(TARGETS) + list(TRAIN_AT)}
        common = set(cov_stems)
        for _, stems in sets.values():
            common &= set(stems)
        stems = sorted(common, key=int)
        cpos = {s: i for i, s in enumerate(cov_stems)}
        Xc = Xcov[[cpos[s] for s in stems]]
        X = {}
        for p, (feats, st) in sets.items():
            pos = {s: i for i, s in enumerate(st)}
            X[p] = feats[[pos[s] for s in stems]]
        n = len(stems)
        print(f"\n{method}  ({n} images common to all payloads)", flush=True)
        for p_train in TRAIN_AT:
            single = {p: [] for p in TARGETS}
            pooled = {p: [] for p in TARGETS}
            for seed in range(SEEDS):
                idx = np.random.default_rng(seed).permutation(n)
                tr, te = idx[:n // 2], idx[n // 2:]
                m = e16.LeanFLD(800, 200, seed).fit(
                    np.concatenate([Xc[tr], X[p_train][tr]]),
                    np.concatenate([np.zeros(len(tr)), np.ones(len(tr))]))
                sc = m.decision_function(Xc[te])
                for p, k in TARGETS.items():
                    ss = m.decision_function(X[p][te])
                    single[p].append(e16.p_error(sc, ss))
                    res = e17.pooled(*e17.batches(sc, ss, B, k, N_BATCH, rng), k)
                    pooled[p].append(min(res.values()))
                print(f"   trained at {p_train}, seed {seed}: " + "  ".join(
                    f"{p}: single {single[p][-1]:.4f} batch {pooled[p][-1]:.4f}"
                    for p in TARGETS), flush=True)
            for p, k in TARGETS.items():
                rows.append(dict(
                    method=method, target_payload=p, covers=k, trained_at=p_train,
                    single_p_e=float(np.mean(single[p])), batch_p_e=float(np.mean(pooled[p])),
                    matched_single_p_e=matched[f"{method}_{p}"]["p_e_mean"],
                    matched_batch_p_e=pooled_matched.get((method, p))))
    (out / "mismatched_warden.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\nBatch of 20, total payload 0.2. Batch P_E under each warden "
          "(lower = better for the warden):")
    print(f"{'method':<8}{'strategy':<12}{'trained at payload':>20}"
          f"{'trained at 0.05':>17}{'trained at 0.1':>16}{'warden best':>13}")
    for method in ("SCRIPT", "GMAS", "ROAST"):
        for p, k in TARGETS.items():
            r = {x["trained_at"]: x for x in rows
                 if x["method"] == method and x["target_payload"] == p}
            a = r[0.05]["matched_batch_p_e"]
            vals = [a, r[0.05]["batch_p_e"], r[0.1]["batch_p_e"]]
            print(f"{method:<8}{f'{k} x {p}':<12}{a:>20.4f}{vals[1]:>17.4f}"
                  f"{vals[2]:>16.4f}{min(vals):>13.4f}")
    print(f"\nwrote {out / 'mismatched_warden.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
