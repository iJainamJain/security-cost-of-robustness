"""SRNet as a second CNN warden.

SRNet (Boroumand, Chen, Fridrich, IEEE TIFS 2019) is the reference deep
steganalyser: no pooling in the first seven layers, so the stego residual is not
averaged away, and no pretrained weights - it is trained from scratch on the
steganalysis task. That makes it a useful check on EfficientNet-B0, whose
starting point is ImageNet.

Kept from the paper: the architecture, the PAIR CONSTRAINT (a cover and its own
stego always sit in the same batch, with the same flip/rotation), Adamax at
1e-3 dropped tenfold for the last fifth of training, L2 2e-4 on kernels only.
Different from the paper, and to be stated wherever a number from this is used:
  - 512x512 inputs decoded to 8-bit pixels and scaled to [0,1] (the paper uses
    256x256 and unrounded pixel values); same input as our EfficientNet;
  - far fewer iterations (tens of epochs on 2,000 pairs, not 500k iterations);
  - mixed precision.

Split, epoch selection and P_E are exactly those of m4.py: one paired 40/10/50
split per seed, epoch chosen on validation only, test scored at that epoch.

    python m4_srnet.py --data ROOT --cells GMAS_0.05 --seeds 0 --out OUT
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import shutil
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import m4
from m4_chain import split

DETECTOR = "SRNet"


# ------------------------------------------------------------------ model ---
def conv_bn(cin: int, cout: int, k: int = 3, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout)]


class Type2(nn.Module):
    """Residual block at full resolution (layers 3-7)."""

    def __init__(self, c: int):
        super().__init__()
        self.body = nn.Sequential(*conv_bn(c, c), nn.ReLU(inplace=True), *conv_bn(c, c))

    def forward(self, x):
        return x + self.body(x)


class Type3(nn.Module):
    """Residual block that halves the resolution (layers 8-11)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(*conv_bn(cin, cout), nn.ReLU(inplace=True),
                                  *conv_bn(cout, cout), nn.AvgPool2d(3, 2, 1))
        self.skip = nn.Sequential(*conv_bn(cin, cout, k=1, stride=2))

    def forward(self, x):
        return self.skip(x) + self.body(x)


class SRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn(1, 64), nn.ReLU(inplace=True),            # layer 1
            *conv_bn(64, 16), nn.ReLU(inplace=True),           # layer 2
            *[Type2(16) for _ in range(5)],                    # layers 3-7
            Type3(16, 16), Type3(16, 64), Type3(64, 128), Type3(128, 256),  # 8-11
            *conv_bn(256, 512), nn.ReLU(inplace=True), *conv_bn(512, 512),  # layer 12
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(512, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        nn.init.normal_(self.fc.weight, 0.0, 0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(self.features(x))


# ------------------------------------------------------------------- data ---
class PairSet(Dataset):
    """One item = a cover and its own stego, with the SAME augmentation."""

    def __init__(self, cover_dir, stego_dir, stems):
        self.cover_dir = pathlib.Path(cover_dir)
        self.stego_dir = pathlib.Path(stego_dir)
        self.stems = list(stems)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, i):
        stem = self.stems[i]
        x = np.stack([
            np.asarray(Image.open(self.cover_dir / f"{stem}.jpg"), dtype=np.float32),
            np.asarray(Image.open(self.stego_dir / f"{stem}.jpg"), dtype=np.float32),
        ]) / 255.0                                   # (2, H, W): cover, stego
        if random.random() < 0.5:
            x = x[:, :, ::-1]
        k = random.randint(0, 3)
        if k:
            x = np.rot90(x, k, axes=(1, 2))
        return torch.from_numpy(np.ascontiguousarray(x)[:, None])   # (2, 1, H, W)


# --------------------------------------------------------------- one seed ---
def run_seed(cover_dir, stego_dir, stems, seed, device, epochs, pairs, lr, workers,
             deadline=None, tag=""):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tr, va, te = split(stems, seed)
    g = torch.Generator()
    g.manual_seed(seed)
    dl_tr = DataLoader(PairSet(cover_dir, stego_dir, tr), batch_size=pairs, shuffle=True,
                       num_workers=workers, drop_last=True, generator=g)
    dl_va = DataLoader(m4.PairedSet(cover_dir, stego_dir, va, False),
                       batch_size=2 * pairs, shuffle=False, num_workers=workers)
    dl_te = DataLoader(m4.PairedSet(cover_dir, stego_dir, te, False),
                       batch_size=2 * pairs, shuffle=False, num_workers=workers)

    model = SRNet().to(device)
    kernels = [p for p in model.parameters() if p.ndim > 1]
    others = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.Adamax([dict(params=kernels, weight_decay=2e-4),
                              dict(params=others, weight_decay=0.0)], lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    y = torch.tensor([0.0, 1.0], device=device).repeat(pairs)   # cover, stego, ...

    n_ep, drop_at = epochs, int(0.8 * epochs)
    best_val, chosen, best_ep, ran = 2.0, None, -1, 0
    t_seed = time.time()
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
            losses.append(float(loss))
        vc, vs, _ = m4.score_loader(model, dl_va, device, use_amp)
        val_pe = m4.p_error(vc, vs)
        if val_pe <= best_val:              # validation only; later epoch on ties
            best_val = val_pe
            chosen = m4.score_loader(model, dl_te, device, use_amp)
            best_ep = ep
        ran = ep + 1
        ep_min = (time.time() - t_ep) / 60
        print(f"      {tag}seed {seed} ep {ep:2d}: loss {np.mean(losses):.4f}  "
              f"val P_E {val_pe:.4f}  {ep_min:.1f} min", flush=True)
        # A session that hits Kaggle's cap loses everything. If the epochs asked
        # for cannot fit, train for fewer and say so, rather than be killed.
        if ep == 0 and deadline is not None:
            fit = int((deadline - time.time()) * 0.9 / (ep_min * 60))
            if fit < n_ep - 1:
                n_ep = max(2, 1 + fit)
                drop_at = int(0.8 * n_ep)
                print(f"      !! only {n_ep} of {epochs} epochs fit in the time left; "
                      f"training for {n_ep}", flush=True)

    sc, ss, keep = chosen
    return dict(seed=seed, test_p_e=m4.p_error(sc, ss), test_p_e_reversed=m4.p_error(ss, sc),
                val_p_e=best_val, md_at_fa1=m4.md_at_fa(sc, ss), best_epoch=best_ep,
                epochs_run=ran, epochs_asked=epochs,
                minutes=round((time.time() - t_seed) / 60, 1),
                min_per_epoch=round((time.time() - t_seed) / 60 / max(ran, 1), 2),
                n_train_pairs=len(tr), n_val_pairs=len(va), n_test_pairs=len(te),
                epoch_selection="validation"), [(seed, s, lbl, v) for s, lbl, v in keep]


def paired_stems(cover_dir, stego_dir) -> list[str]:
    stems = sorted((p.stem for p in stego_dir.glob("*.jpg")), key=lambda s: (len(s), s))
    return [s for s in stems if (cover_dir / f"{s}.jpg").exists()]


def null_test(cover_dir, out, device, args) -> bool:
    """Covers against other covers, through the same pair loader. Must read chance."""
    print("\nNULL TEST (cover vs cover) - expect chance-level P_E", flush=True)
    stems = sorted((p.stem for p in cover_dir.glob("*.jpg")), key=lambda s: (len(s), s))
    if args.limit:
        stems = stems[:2 * args.limit]
    # Random groups: BOSSbase is ordered by source, so halves by file number differ.
    stems = [stems[i] for i in np.random.default_rng(20261002).permutation(len(stems))]
    half = len(stems) // 2
    a, b = stems[:half], stems[half:2 * half]
    with tempfile.TemporaryDirectory() as td:
        fake = pathlib.Path(td)
        for sa, sb in zip(a, b):
            shutil.copyfile(cover_dir / f"{sb}.jpg", fake / f"{sa}.jpg")
        r, _ = run_seed(cover_dir, fake, a, 0, device, max(5, args.epochs // 4),
                        args.pairs, args.lr, args.workers, tag="null ")
    mean, sd = m4.null_expectation(r["n_test_pairs"])
    floor = mean - 2.5 * sd
    r.update(null_expected=mean, null_floor=floor, detector=DETECTOR)
    ok = r["test_p_e"] >= floor
    print(f"  NULL TEST P_E = {r['test_p_e']:.4f}   (chance at {r['n_test_pairs']} test "
          f"pairs = {mean:.4f}, pass if >= {floor:.4f})")
    print("  " + ("PASS - pipeline finds no signal where none exists" if ok else
                  "FAIL - PIPELINE IS LEAKING. Stop and debug."))
    (out / "null_test.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (has MANIFEST.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pairs", type=int, default=8, help="cover-stego pairs per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--null-test", action="store_true")
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N pairs")
    args = ap.parse_args()

    data = m4.resolve_data_root(pathlib.Path(args.data))
    out = pathlib.Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | SRNet parameters: "
          f"{sum(p.numel() for p in SRNet().parameters()) / 1e6:.2f} M")
    manifest = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    cover_dir = data / manifest["cover_dir"]
    t_begin = time.time()
    deadline = (t_begin + args.time_budget_min * 60) if args.time_budget_min else None

    if args.null_test and not null_test(cover_dir, out, device, args):
        return 1

    cells = [c for c in manifest["cells"] if c["cell"] in args.cells]
    missing = set(args.cells) - {c["cell"] for c in cells}
    if missing:
        print(f"unknown cell names: {sorted(missing)}")
        return 1

    res_path = out / "seed_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    longest = 0.0
    # Seeds outermost: if time runs out, every cell has its early seeds.
    for seed in args.seeds:
        for c in cells:
            if any(r["cell"] == c["cell"] and r["seed"] == seed for r in results):
                print(f"skip {c['cell']} seed {seed} (already done)")
                continue
            if deadline is not None and time.time() + longest * 60 * 1.1 > deadline and longest:
                print(f"TIME BUDGET: stopping before {c['cell']} seed {seed}")
                res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                return 0
            stego_dir = data / c["stego_dir"]
            stems = paired_stems(cover_dir, stego_dir)
            if args.limit:
                stems = stems[:args.limit]
            print(f"\n{c['cell']} seed {seed}  ({len(stems)} pairs)", flush=True)
            r, rows = run_seed(cover_dir, stego_dir, stems, seed, device, args.epochs,
                               args.pairs, args.lr, args.workers, deadline)
            longest = max(longest, r["minutes"])
            r.update(cell=c["cell"], method=c["method"], payload_bpnzac=c["payload_bpnzac"],
                     arm=c["arm"], detector=DETECTOR,
                     p_e_chance_level=float(m4.null_expectation(r["n_test_pairs"])[0]))
            print(f"    seed {seed}: P_E {r['test_p_e']:.4f} (val {r['val_p_e']:.4f}, epoch "
                  f"{r['best_epoch']} of {r['epochs_run']} run)  {r['minutes']:.0f} min, "
                  f"{r['min_per_epoch']:.2f} min/epoch", flush=True)
            f = out / "scores" / f"{c['cell']}.csv"
            new = not f.exists()
            with f.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["seed", "image", "label", "score"])
                w.writerows(rows)
            results.append(r)
            res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
