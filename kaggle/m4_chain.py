"""Curriculum ("chain") training of the CNN warden for very low payloads.

Trained from scratch at 0.01 bpnzac the network learns nothing: the signal is
too weak to find from an ImageNet starting point. The standard remedy in
steganalysis is to train where the signal is learnable and carry the weights
down: here SCRIPT 0.05 -> 0.02 -> 0.01. A warden would do this, so a security
figure measured against a from-scratch network is only an upper bound.

For each seed, ONE paired 40/10/50 split is drawn over the images present in
every stage and used for all of them. A test image is therefore never seen in
training at any payload - not as a cover, not as a stego.

Per stage:
  - the network as it arrives from the previous stage is scored first
    ("transfer": a warden trained at the higher payload, applied unchanged);
  - it is then fine-tuned, and the epoch is chosen on VALIDATION only, with the
    untouched arriving network as one of the candidates;
  - the chosen weights go on to the next stage.

Everything else (model, augmentation, optimiser, P_E) is imported from m4.py so
the two cannot drift apart.

    python m4_chain.py --cover DIR --stage SCRIPT_0.05=DIR --stage SCRIPT_0.02=DIR \
        --stage SCRIPT_0.01=DIR --epochs 40 25 25 --seeds 0 1 --out OUT
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import m4


def split(stems: list[str], seed: int):
    """Same rule as m4.run_cell: 10% validation, 40% train, 50% test, paired."""
    idx = np.random.default_rng(seed).permutation(len(stems))
    half = len(stems) // 2
    n_val = max(2, int(round(0.2 * half)))
    va = [stems[i] for i in idx[:n_val]]
    tr = [stems[i] for i in idx[n_val:half]]
    te = [stems[i] for i in idx[half:]]
    assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va))
    return tr, va, te


def run_seed(seed, cover_dir, stages, epochs, stems, device, bs, lr, workers):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tr, va, te = split(stems, seed)
    use_amp = device.type == "cuda"
    lossf = nn.BCEWithLogitsLoss()
    model = m4.build_model().to(device)
    out, score_rows = [], {}

    for si, (name, stego_dir) in enumerate(stages):
        g = torch.Generator()
        g.manual_seed(seed * 100 + si)
        dl_tr = DataLoader(m4.PairedSet(cover_dir, stego_dir, tr, True), batch_size=bs,
                           shuffle=True, num_workers=workers, drop_last=True, generator=g)
        dl_va = DataLoader(m4.PairedSet(cover_dir, stego_dir, va, False),
                           batch_size=bs, shuffle=False, num_workers=workers)
        dl_te = DataLoader(m4.PairedSet(cover_dir, stego_dir, te, False),
                           batch_size=bs, shuffle=False, num_workers=workers)
        # Fresh optimiser and scaler per stage: the momentum built up at the
        # previous payload is not information about this one.
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val, chosen, best_ep, best_state = 2.0, None, None, None
        transfer = None
        if si > 0:
            vc, vs, _ = m4.score_loader(model, dl_va, device, use_amp)
            best_val = m4.p_error(vc, vs)
            chosen = m4.score_loader(model, dl_te, device, use_amp)
            best_ep = -1
            best_state = copy.deepcopy(model.state_dict())
            transfer = dict(test_p_e=m4.p_error(chosen[0], chosen[1]), val_p_e=best_val,
                            md_at_fa1=m4.md_at_fa(chosen[0], chosen[1]))
            score_rows[name + "__transfer"] = [(seed, s, lbl, v) for s, lbl, v in chosen[2]]
            print(f"    seed {seed} {name}: transfer from {stages[si - 1][0]} "
                  f"P_E {transfer['test_p_e']:.4f} (val {best_val:.4f})", flush=True)

        t0 = time.time()
        for ep in range(epochs[si]):
            model.train()
            for x, y, _ in dl_tr:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = lossf(model(x).squeeze(1), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            vc, vs, _ = m4.score_loader(model, dl_va, device, use_amp)
            val_pe = m4.p_error(vc, vs)
            if val_pe <= best_val:          # validation only; later epoch on ties
                best_val = val_pe
                chosen = m4.score_loader(model, dl_te, device, use_amp)
                best_ep = ep
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)   # the chosen weights go to the next stage
        sc, ss, keep = chosen
        r = dict(seed=seed, stage=name, stage_index=si, epochs=epochs[si],
                 test_p_e=m4.p_error(sc, ss), test_p_e_reversed=m4.p_error(ss, sc),
                 val_p_e=best_val, md_at_fa1=m4.md_at_fa(sc, ss), best_epoch=best_ep,
                 transfer=transfer, minutes=round((time.time() - t0) / 60, 1),
                 n_train_pairs=len(tr), n_val_pairs=len(va), n_test_pairs=len(te))
        out.append(r)
        score_rows[name] = [(seed, s, lbl, v) for s, lbl, v in keep]
        print(f"    seed {seed} {name}: P_E {r['test_p_e']:.4f} (val {best_val:.4f}, "
              f"epoch {best_ep} of {epochs[si]}; -1 = arriving weights)  "
              f"{r['minutes']:.0f} min", flush=True)
    return out, score_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cover", required=True)
    ap.add_argument("--stage", action="append", required=True, metavar="NAME=DIR",
                    help="in training order, highest payload first")
    ap.add_argument("--epochs", type=int, nargs="+", required=True,
                    help="one value per stage, or one value for all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N pairs")
    ap.add_argument("--time-budget-min", type=float, default=None)
    args = ap.parse_args()

    cover_dir = pathlib.Path(args.cover)
    stages = [(s.split("=", 1)[0], pathlib.Path(s.split("=", 1)[1])) for s in args.stage]
    epochs = args.epochs * len(stages) if len(args.epochs) == 1 else args.epochs
    if len(epochs) != len(stages):
        raise SystemExit("--epochs needs one value, or one per stage")
    out = pathlib.Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Only images present in EVERY stage, so one split serves the whole chain.
    common = {p.stem for p in cover_dir.glob("*.jpg")}
    for name, d in stages:
        have = {p.stem for p in d.glob("*.jpg")}
        print(f"  {name}: {len(have)} stego images")
        common &= have
    stems = sorted(common, key=lambda s: (len(s), s))
    if args.limit:
        stems = stems[:args.limit]
    chance, sd = m4.null_expectation(len(stems) - len(stems) // 2)
    print(f"chain {' -> '.join(n for n, _ in stages)}: {len(stems)} pairs in every "
          f"stage, epochs {epochs}, seeds {args.seeds}, chance {chance:.4f} +/- {sd:.4f}",
          flush=True)

    res_path = out / "seed_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    t_begin, longest = time.time(), 0.0
    for seed in args.seeds:
        if any(r["seed"] == seed for r in results):
            print(f"skip seed {seed} (already done)")
            continue
        if args.time_budget_min is not None:
            used = (time.time() - t_begin) / 60
            if used + longest * 1.15 > args.time_budget_min:
                print(f"TIME BUDGET: {used:.0f} min used; stopping before seed {seed}")
                break
        t0 = time.time()
        rows, score_rows = run_seed(seed, cover_dir, stages, epochs, stems, device,
                                    args.batch_size, args.lr, args.workers)
        longest = max(longest, (time.time() - t0) / 60)
        # Checkpoint after every seed: a session timeout costs one seed, not all.
        for name, rr in score_rows.items():
            f = out / "scores" / f"{name}.csv"
            new = not f.exists()
            with f.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["seed", "image", "label", "score"])
                w.writerows(rr)
        results += rows
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
