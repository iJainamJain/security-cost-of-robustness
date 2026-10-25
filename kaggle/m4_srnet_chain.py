"""SRNet with curriculum training, driven by a plan of (chain, seed) tasks.

The speed test (M4_srnet_speed) showed SRNet from scratch is strong but slow to
converge: on J-UNIWARD 0.2 it was still improving at epoch 38 of 40, and at
lower payloads a from-scratch network may not start learning at all in the
epochs we can afford. The standard practice is a curriculum: train at the
highest payload of a method, then carry the weights down, one payload at a
time. That is both what a warden would do and cheaper than training every cell
from scratch.

A task is one method's chain for one seed, e.g. J-UNIWARD 0.3 -> 0.2 -> 0.1 ->
0.05. One paired 40/10/50 split is drawn per seed over the images present in
every stage of the chain and reused for all of them, so a test image is never
seen in training at any payload. Per stage: the arriving network is scored
unchanged ("transfer"), then fine-tuned; the epoch is chosen on validation with
the arriving weights as a candidate; the chosen weights go to the next stage.

Model, pair-constraint loader and optimiser settings are those of m4_srnet.py;
split and P_E are those of m4.py. The learning rate drops tenfold after 80% of
the epochs in the first stage and after 60% in fine-tuning stages.

    python m4_srnet_chain.py --plan plan.json --out OUT --time-budget-min 600

plan.json: [{"data": ROOT, "chain": ["GMAS_0.1", "GMAS_0.05"], "epochs": [50, 20],
             "seed": 0}, ...]
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
import m4_srnet
from m4_chain import split

DEFAULT_MIN_PER_EPOCH = 3.9      # measured on a T4 at 2,000 training pairs


def train_stage(model, cover_dir, stego_dir, tr, va, te, seed, si, device, epochs,
                pairs, lr, workers, deadline, reserve_epochs, tag, mpe_plan=None):
    g = torch.Generator()
    g.manual_seed(seed * 100 + si)
    dl_tr = DataLoader(m4_srnet.PairSet(cover_dir, stego_dir, tr), batch_size=pairs,
                       shuffle=True, num_workers=workers, drop_last=True, generator=g)
    dl_va = DataLoader(m4.PairedSet(cover_dir, stego_dir, va, False),
                       batch_size=2 * pairs, shuffle=False, num_workers=workers)
    dl_te = DataLoader(m4.PairedSet(cover_dir, stego_dir, te, False),
                       batch_size=2 * pairs, shuffle=False, num_workers=workers)
    kernels = [p for p in model.parameters() if p.ndim > 1]
    others = [p for p in model.parameters() if p.ndim <= 1]
    # Fresh optimiser per stage: momentum from the previous payload is not
    # information about this one.
    opt = torch.optim.Adamax([dict(params=kernels, weight_decay=2e-4),
                              dict(params=others, weight_decay=0.0)], lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    y = torch.tensor([0.0, 1.0], device=device).repeat(pairs)   # cover, stego, ...

    best_val, chosen, best_ep, best_state = 2.0, None, None, None
    transfer, transfer_rows = None, []
    if si > 0:
        vc, vs, _ = m4.score_loader(model, dl_va, device, use_amp)
        best_val = m4.p_error(vc, vs)
        chosen = m4.score_loader(model, dl_te, device, use_amp)
        best_ep, best_state = -1, copy.deepcopy(model.state_dict())
        transfer = dict(test_p_e=m4.p_error(chosen[0], chosen[1]), val_p_e=best_val,
                        md_at_fa1=m4.md_at_fa(chosen[0], chosen[1]))
        transfer_rows = [(seed, s, lbl, v) for s, lbl, v in chosen[2]]
        print(f"      {tag}: transfer P_E {transfer['test_p_e']:.4f} "
              f"(val {best_val:.4f})", flush=True)

    frac = 0.8 if si == 0 else 0.6
    n_ep, drop_at, ran, t0 = epochs, int(frac * epochs), 0, time.time()
    for ep in range(epochs):
        if ep >= n_ep:
            break
        if ep == drop_at:
            for grp in opt.param_groups:
                grp["lr"] = lr * 0.1
        t_ep, losses = time.time(), []
        model.train()
        for x in dl_tr:
            x = x.to(device, non_blocking=True).flatten(0, 1)   # (2*pairs, 1, H, W)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = lossf(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        vc, vs, _ = m4.score_loader(model, dl_va, device, use_amp)
        val_pe = m4.p_error(vc, vs)
        if val_pe <= best_val:              # validation only; later epoch on ties
            best_val = val_pe
            chosen = m4.score_loader(model, dl_te, device, use_amp)
            best_ep, best_state = ep, copy.deepcopy(model.state_dict())
        ran = ep + 1
        ep_min = (time.time() - t_ep) / 60
        print(f"      {tag} ep {ep:2d}: loss {np.mean(losses):.4f}  val P_E {val_pe:.4f}  "
              f"{ep_min:.1f} min", flush=True)
        # Never be killed by the session cap: if this stage plus the stages
        # still to come cannot fit, shorten this one and say so.
        if ep == 0 and deadline is not None:
            # Epoch 0 is the slowest (data-loader warm-up, 4.3-4.7 min against 3.5
            # later). Sizing the stage from it cut the first stage of the SFH
            # sessions to 15-25 of 50 epochs and left 170 minutes unused, so the
            # estimate is capped at 1.1 x the planning figure when one is given.
            per = min(ep_min, 1.1 * mpe_plan) if mpe_plan else ep_min
            fit = int((deadline - time.time()) * 0.9 / (per * 60)) - reserve_epochs
            if fit < n_ep - 1:
                n_ep = max(2, 1 + fit)
                drop_at = int(frac * n_ep)
                print(f"      !! only {n_ep} of {epochs} epochs fit; training for {n_ep}",
                      flush=True)

    model.load_state_dict(best_state)       # the chosen weights go to the next stage
    sc, ss, keep = chosen
    minutes = (time.time() - t0) / 60
    r = dict(seed=seed, stage_index=si, test_p_e=m4.p_error(sc, ss),
             test_p_e_reversed=m4.p_error(ss, sc), val_p_e=best_val,
             md_at_fa1=m4.md_at_fa(sc, ss), best_epoch=best_ep, epochs_run=ran,
             epochs_asked=epochs, transfer=transfer, minutes=round(minutes, 1),
             min_per_epoch=round(minutes / max(ran, 1), 2),
             n_train_pairs=len(tr), n_val_pairs=len(va), n_test_pairs=len(te),
             epoch_selection="validation")
    return r, [(seed, s, lbl, v) for s, lbl, v in keep], transfer_rows


def run_task(task, device, args, deadline):
    data = m4.resolve_data_root(pathlib.Path(task["data"]))
    manifest = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    by_name = {c["cell"]: c for c in manifest["cells"]}
    cover_dir = data / manifest["cover_dir"]
    chain, epochs, seed = task["chain"], task["epochs"], task["seed"]
    common = {p.stem for p in cover_dir.glob("*.jpg")}
    for name in chain:
        common &= {p.stem for p in (data / by_name[name]["stego_dir"]).glob("*.jpg")}
    stems = sorted(common, key=lambda s: (len(s), s))
    if args.limit:
        stems = stems[:args.limit]
    print(f"\nCHAIN {' -> '.join(chain)}  seed {seed}  ({len(stems)} pairs in every "
          f"stage, epochs {epochs})", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tr, va, te = split(stems, seed)
    model = m4_srnet.SRNet().to(device)
    out_rows, scores = [], {}
    for si, name in enumerate(chain):
        c = by_name[name]
        r, rows, trows = train_stage(
            model, cover_dir, data / c["stego_dir"], tr, va, te, seed, si, device,
            epochs[si], args.pairs, args.lr, args.workers, deadline,
            sum(epochs[si + 1:]), f"{name} seed {seed}", args.min_per_epoch)
        r.update(cell=name, method=c["method"], payload_bpnzac=c["payload_bpnzac"],
                 arm=c.get("arm"), detector="SRNet", chain=" -> ".join(chain),
                 p_e_chance_level=float(m4.null_expectation(r["n_test_pairs"])[0]))
        print(f"    {name} seed {seed}: P_E {r['test_p_e']:.4f} (val {r['val_p_e']:.4f}, "
              f"epoch {r['best_epoch']} of {r['epochs_run']}; -1 = arriving weights)  "
              f"{r['minutes']:.0f} min", flush=True)
        out_rows.append(r)
        scores[name] = rows
        if trows:
            scores[name + "__transfer"] = trows
    return out_rows, scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="JSON list of tasks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=8, help="cover-stego pairs per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N pairs")
    ap.add_argument("--min-per-epoch", type=float, default=DEFAULT_MIN_PER_EPOCH,
                    help="expected minutes per epoch, used only to decide whether a "
                         "task fits in the time left (3.9 at 512x512 on a T4)")
    args = ap.parse_args()

    plan = json.loads(pathlib.Path(args.plan).read_text(encoding="utf-8"))
    out = pathlib.Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | SRNet parameters: "
          f"{sum(p.numel() for p in m4_srnet.SRNet().parameters()) / 1e6:.2f} M | "
          f"{len(plan)} tasks")
    t_begin = time.time()
    deadline = (t_begin + args.time_budget_min * 60) if args.time_budget_min else None

    res_path = out / "seed_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    mpe = args.min_per_epoch
    for task in plan:
        label = " -> ".join(task["chain"])
        if any(r["chain"] == label and r["seed"] == task["seed"] for r in results):
            print(f"skip {label} seed {task['seed']} (already done)")
            continue
        need = sum(task["epochs"]) * mpe * 1.05
        if deadline is not None and time.time() + need * 60 > deadline:
            print(f"TIME BUDGET: {label} seed {task['seed']} needs about {need:.0f} min, "
                  f"{(deadline - time.time()) / 60:.0f} left; skipping it")
            continue
        rows, scores = run_task(task, device, args, deadline)
        mpe = max(mpe, max(r["min_per_epoch"] for r in rows))
        # Checkpoint after every task: a session timeout costs one task, not all.
        for name, rr in scores.items():
            f = out / "scores" / f"{name}.csv"
            new = not f.exists()
            with f.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["seed", "image", "label", "score"])
                w.writerows(rr)
        results += rows
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
