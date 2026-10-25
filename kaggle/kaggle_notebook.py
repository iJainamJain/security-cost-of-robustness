"""M4 CNN steganalysis - reference implementation.

Paste into a Kaggle notebook (GPU on) or run as a script. Written to be correct
first and fast second; every choice that looks unusual is deliberate and is
explained in 02_PITFALLS.md.

Usage (Kaggle: attach the dataset first):

    python kaggle_notebook.py --data /kaggle/input/stego-eval \
        --out /kaggle/working/m4_results --null-test
    python kaggle_notebook.py --data /kaggle/input/stego-eval \
        --out /kaggle/working/m4_results --cells GMAS_0.1
    python kaggle_notebook.py --data /kaggle/input/stego-eval \
        --out /kaggle/working/m4_results --all

Requires: torch, timm, numpy, pillow.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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

SEEDS = [0, 1, 2, 3, 4]
# Early stopping on VALIDATION only: stop a seed after this many epochs with no
# strict improvement in validation P_E. None = always run every epoch.
PATIENCE = None


# ---------------------------------------------------------------- metrics ---
def p_error(sc: np.ndarray, ss: np.ndarray) -> float:
    """min over thresholds of 0.5*(P_FA + P_MD).

    Higher score must mean 'more likely stego'.

    WARNING - inverted scores do NOT show up as a value near 1.0. Because this
    is a min over thresholds, the degenerate threshold (call everything cover)
    always achieves 0.5, so backwards wiring collapses to ~0.5: it reads as
    'perfectly secure' and is indistinguishable from a passing null test. The
    only way to see it is to score the other direction as well - run_cell does,
    and reports p_e_reversed. If reversed is clearly LOWER, the scores are
    upside down.
    """
    best = 1.0
    for t in np.unique(np.concatenate([sc, ss])):
        best = min(best, 0.5 * ((sc >= t).mean() + (ss < t).mean()))
    return float(best)


def null_expectation(n_test_pairs: int) -> tuple[float, float]:
    """(mean, sd) of P_E when there is NO signal, at this test-set size.

    P_E = 0.5 * (1 - D+), where D+ is the one-sided two-sample
    Kolmogorov-Smirnov statistic. D+ is not zero under the null - it shrinks
    like 1/sqrt(n) - so a perfect pipeline reads BELOW 0.5: about 0.453 at 75
    test pairs, 0.466 at 150, 0.476 at 300. "Chance" is this number, not 0.50,
    and a P_E at or above it means "not detected at this sample size".

    Constants checked by simulation, 4000 trials per n.
    """
    n = max(int(n_test_pairs), 1)
    mean = 0.5 * (1.0 - math.sqrt(math.pi / 8.0) / math.sqrt(n / 2.0))
    return mean, 0.232 / math.sqrt(n)


def md_at_fa(sc: np.ndarray, ss: np.ndarray, target_fa: float = 0.01) -> float:
    """Missed detection at a fixed false-alarm rate - a realistic warden."""
    return float((ss < np.quantile(sc, 1 - target_fa)).mean())


# ---------------------------------------------------------------- dataset ---
class PairedSet(Dataset):
    """Cover/stego pairs for one cell.

    Loads JPEGs to float32 in [0,1] at native 512x512 and NEVER re-encodes.

    Augmentation is limited to flips and 90-degree rotations: these are the only
    transforms that permute pixels without resampling, so they preserve the
    embedding signal. Anything involving interpolation, cropping or
    re-compression destroys what we are trying to detect.
    """

    def __init__(self, cover_dir, stego_dir, stems, train: bool):
        self.cover_dir = pathlib.Path(cover_dir)
        self.stego_dir = pathlib.Path(stego_dir)
        self.stems = list(stems)
        self.train = train

    def __len__(self) -> int:
        return len(self.stems) * 2

    def __getitem__(self, i):
        stem = self.stems[i // 2]
        is_stego = i % 2
        d = self.stego_dir if is_stego else self.cover_dir
        x = np.asarray(Image.open(d / f"{stem}.jpg"), dtype=np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                x = np.fliplr(x)
            if random.random() < 0.5:
                x = np.flipud(x)
            k = random.randint(0, 3)
            if k:
                x = np.rot90(x, k)
        x = np.ascontiguousarray(x)[None]  # (1, H, W)
        return torch.from_numpy(x), torch.tensor(float(is_stego)), stem


# ------------------------------------------------------------------ model ---
def build_model() -> nn.Module:
    """EfficientNet-B0, ImageNet weights, single-channel input.

    No resize: the network is fully convolutional and accepts 512x512 directly.
    Downsampling to 224 would average away the residual the detector needs.
    `in_chans=1` lets timm adapt the stem rather than us faking a 3-channel
    image. Inputs are scaled to [0,1] only - no ImageNet mean/std, which is
    meaningless on a replicated grayscale channel.
    """
    import timm

    return timm.create_model("efficientnet_b0", pretrained=True,
                             num_classes=1, in_chans=1)


# ---------------------------------------------------------------- scoring ---
def score_loader(model, dl, device, use_amp):
    """Score every image in a loader -> (cover scores, stego scores, rows)."""
    model.eval()
    sc, ss, keep = [], [], []
    with torch.no_grad():
        for x, y, stem in dl:
            with torch.cuda.amp.autocast(enabled=use_amp):
                o = model(x.to(device)).squeeze(1).float().cpu().numpy()
            for oi, yi, si in zip(o, y.numpy(), stem):
                (ss if yi > 0.5 else sc).append(oi)
                keep.append((si, "stego" if yi > 0.5 else "cover", float(oi)))
    return np.asarray(sc), np.asarray(ss), keep


# --------------------------------------------------------------- one cell ---
def run_cell(cover_dir, stego_dir, stems, device, epochs, bs, lr, out_scores):
    per_seed_pe, per_seed_md, best_epochs = [], [], []
    per_seed_rev, per_seed_val = [], []
    epochs_run = []
    rows = []
    n_tr = n_va = n_te = 0

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # PAIRED split: the index decides, so cover i and stego i stay together.
        # Splitting the two classes independently would put a cover in training
        # and its own stego in test - the network then recognises the photograph
        # instead of the embedding. See 02_PITFALLS.md section 1.
        #
        # Three-way: 40% train / 10% validation / 50% test. The validation
        # slice exists only to choose the epoch. Choosing it on the test set
        # instead - the common shortcut - takes a min over ~40 noisy estimates
        # and drags P_E down by ~0.07 with no signal present at all, which both
        # fails the null test on a correct pipeline and understates the
        # security of every genuinely hard-to-detect method.
        idx = np.random.default_rng(seed).permutation(len(stems))
        half = len(stems) // 2
        n_val = max(2, int(round(0.2 * half)))
        va = [stems[i] for i in idx[:n_val]]
        tr = [stems[i] for i in idx[n_val:half]]
        te = [stems[i] for i in idx[half:]]
        assert not (set(tr) & set(te)), "train/test stems overlap"
        assert not (set(va) & set(te)), "val/test stems overlap"
        assert not (set(tr) & set(va)), "train/val stems overlap"
        n_tr, n_va, n_te = len(tr), len(va), len(te)

        g = torch.Generator()
        g.manual_seed(seed)
        dl_tr = DataLoader(PairedSet(cover_dir, stego_dir, tr, True),
                           batch_size=bs, shuffle=True, num_workers=2,
                           drop_last=True, generator=g)
        dl_va = DataLoader(PairedSet(cover_dir, stego_dir, va, False),
                           batch_size=bs, shuffle=False, num_workers=2)
        dl_te = DataLoader(PairedSet(cover_dir, stego_dir, te, False),
                           batch_size=bs, shuffle=False, num_workers=2)

        model = build_model().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        lossf = nn.BCEWithLogitsLoss()
        use_amp = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val, chosen, best_ep = 2.0, None, -1
        best_strict, since_better, ran = 2.0, 0, 0
        for ep in range(epochs):
            ran = ep + 1
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

            # Epoch selection looks at VALIDATION only. The test set is
            # scored at the chosen epoch and never influences the choice.
            # `<=` keeps the later epoch on ties: the validation slice is
            # small, ties are common, and a longer-trained model is the safer
            # tie-break.
            vc, vs, _ = score_loader(model, dl_va, device, use_amp)
            val_pe = p_error(vc, vs)
            if val_pe <= best_val:
                best_val = val_pe
                chosen = score_loader(model, dl_te, device, use_amp)
                best_ep = ep
            # Patience counts STRICT improvements: with `<=` alone, a cell at
            # chance ties forever and would never stop.
            if val_pe < best_strict:
                best_strict, since_better = val_pe, 0
            else:
                since_better += 1
            if PATIENCE is not None and since_better >= PATIENCE:
                break

        sc, ss, keep = chosen
        test_pe = p_error(sc, ss)
        rev_pe = p_error(ss, sc)
        per_seed_pe.append(test_pe)
        per_seed_rev.append(rev_pe)
        per_seed_val.append(best_val)
        per_seed_md.append(md_at_fa(sc, ss))
        best_epochs.append(best_ep)
        epochs_run.append(ran)
        rows += [(seed, s, lbl, v) for s, lbl, v in keep]
        print(f"    seed {seed}: P_E {test_pe:.4f} "
              f"(val {best_val:.4f}, epoch {best_ep} of {ran} run)", flush=True)

    # Inversion guard. For a detectable cell with correctly oriented scores,
    # the forward P_E is low and the reversed one sits at ~0.5. If it is the
    # other way round the scores are upside down - and the forward number
    # would be reporting ~0.5, i.e. "secure", for a method that is not.
    #
    # Both conditions are needed. At chance level the two directions are just
    # noise around the same value, so "reversed < forward" alone fires at
    # random - including on the null test, where it would be a false alarm on
    # the one check that must be trusted. Requiring the reversed direction to
    # sit significantly BELOW chance means it fires only on a real signal
    # pointing the wrong way.
    chance, chance_sd = null_expectation(n_te)
    rev_mean, fwd_mean = float(np.mean(per_seed_rev)), float(np.mean(per_seed_pe))
    if rev_mean < chance - 3.0 * chance_sd and rev_mean < fwd_mean - 0.05:
        print("  !! SCORES LOOK INVERTED: reversed P_E "
              f"{np.mean(per_seed_rev):.4f} is lower than forward "
              f"{np.mean(per_seed_pe):.4f}. Higher score must mean 'stego'.",
              flush=True)

    out_scores.parent.mkdir(parents=True, exist_ok=True)
    with out_scores.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "image", "label", "score"])
        w.writerows(rows)

    return dict(
        p_e_mean=float(np.mean(per_seed_pe)),
        p_e_std=float(np.std(per_seed_pe)),
        p_e_per_seed=[float(v) for v in per_seed_pe],
        md_at_fa1_mean=float(np.mean(per_seed_md)),
        md_at_fa1_std=float(np.std(per_seed_md)),
        best_epoch_per_seed=best_epochs,
        epochs_run_per_seed=epochs_run,
        patience=PATIENCE,
        p_e_reversed_mean=float(np.mean(per_seed_rev)),
        val_p_e_per_seed=[float(v) for v in per_seed_val],
        # What P_E reads at this test size with NO signal. Not 0.50.
        p_e_chance_level=float(null_expectation(n_te)[0]),
        epoch_selection="validation",
        n_train_pairs=n_tr, n_val_pairs=n_va, n_test_pairs=n_te,
    )


def null_test(cover_dir, out, device, args) -> bool:
    """Covers labelled against other covers. Must return chance-level P_E.

    "Chance" is NOT 0.50 - see null_expectation. The gate is one-sided: a leak
    can only push P_E down, so we only test the lower bound.

    This is the check that catches label leakage, ordering artifacts and
    train/test contamination. Nothing downstream is trustworthy until it passes.
    """
    print("\nNULL TEST (cover vs cover) - expect chance-level P_E")
    stems = sorted((p.stem for p in cover_dir.glob("*.jpg")),
                   key=lambda s: (len(s), s))
    # The two groups MUST be drawn at random. BOSSbase is ordered by source
    # (camera), so "first half vs second half by file number" compares two
    # different kinds of image, and any competent detector separates them:
    # at 2,000 covers that read P_E 0.29 here and 0.31 under DCTR, with no leak
    # anywhere. Random groups of the same covers give 0.48.
    stems = [stems[i] for i in
             np.random.default_rng(20261002).permutation(len(stems))]
    half = len(stems) // 2
    a, b = stems[:half], stems[half:2 * half]

    # Build a fake 'stego' folder whose file N.jpg is a *different cover*, named
    # to pair with cover N.jpg. There is no embedded signal to find.
    with tempfile.TemporaryDirectory() as td:
        fake = pathlib.Path(td)
        for sa, sb in zip(a, b):
            shutil.copyfile(cover_dir / f"{sb}.jpg", fake / f"{sa}.jpg")
        r = run_cell(cover_dir, fake, a, device, max(5, args.epochs // 4),
                     args.batch_size, args.lr, out / "scores" / "NULL_TEST.csv")

    mean, sd = null_expectation(r["n_test_pairs"])
    # Seeds reuse the same images under different splits, so they are not
    # independent draws; 2.5 sd of a SINGLE seed is a deliberately loose floor.
    # A real leak (the network recognising photographs) lands far below it.
    floor = mean - 2.5 * sd
    r["null_expected"], r["null_floor"] = mean, floor
    print(f"\n  NULL TEST P_E = {r['p_e_mean']:.4f}   "
          f"(chance at {r['n_test_pairs']} test pairs = {mean:.4f}, "
          f"pass if >= {floor:.4f})")
    ok = r["p_e_mean"] >= floor
    print("  " + ("PASS - pipeline finds no signal where none exists" if ok else
                  "FAIL - PIPELINE IS LEAKING. Stop and debug (02_PITFALLS.md 2)"))
    (out / "null_test.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
    return ok


def resolve_data_root(given: pathlib.Path) -> pathlib.Path:
    """Find the folder that actually contains MANIFEST.json.

    Kaggle mounts a dataset at /kaggle/input/<slug>/, and because our zip has a
    top-level `dataset/` folder the real root is usually
    /kaggle/input/<slug>/dataset. Rather than making anyone get that right, we
    search: the given path, then one or two levels down.
    """
    if (given / "MANIFEST.json").exists():
        return given
    for cand in sorted(given.glob("*/MANIFEST.json")) + sorted(given.glob("*/*/MANIFEST.json")):
        print(f"note: using data root {cand.parent} (MANIFEST.json found there)")
        return cand.parent
    raise SystemExit(
        "No MANIFEST.json under " + str(given) + "\n"
        "Point --data at the folder containing MANIFEST.json. On Kaggle that is\n"
        "usually /kaggle/input/<dataset-slug>/dataset\n"
        "Tip: run  !find /kaggle/input -name MANIFEST.json  to locate it.")


# ------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (has MANIFEST.json)")
    ap.add_argument("--out", required=True, help="output dir for m4_results")
    ap.add_argument("--cells", nargs="*", default=None, help="specific cell names")
    ap.add_argument("--all", action="store_true", help="run every cell")
    ap.add_argument("--null-test", action="store_true", help="run the null test first")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="stop cleanly before starting a cell that would not "
                         "finish inside this many minutes from now")
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop a seed after N epochs without a strict "
                         "validation improvement")
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of seeds (0..N-1); 5 for the 300-image grid")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    global SEEDS, PATIENCE
    SEEDS = list(range(args.seeds))
    PATIENCE = args.patience

    data = resolve_data_root(pathlib.Path(args.data))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU. This will take days on CPU.")

    manifest = json.loads((data / "MANIFEST.json").read_text(encoding="utf-8"))
    cover_dir = data / manifest["cover_dir"]

    if args.null_test:
        if not null_test(cover_dir, out, device, args):
            return 1
        if not (args.all or args.cells):
            return 0

    cells = manifest["cells"]
    if args.cells:
        cells = [c for c in cells if c["cell"] in args.cells]
        missing = set(args.cells) - {c["cell"] for c in cells}
        if missing:
            print(f"unknown cell names: {sorted(missing)}")
            return 1
    elif not args.all:
        print("nothing to do: pass --all or --cells NAME")
        return 0

    res_path = out / "results.json"
    results = (json.loads(res_path.read_text(encoding="utf-8"))
               if res_path.exists() else [])
    done = {(r["cell"], r["detector"]) for r in results}

    t_begin, longest = time.time(), 0.0
    for c in cells:
        if (c["cell"], "EfficientNet-B0") in done:
            print(f"skip {c['cell']} (already in results.json)")
            continue
        # A Kaggle session that hits its 12-hour cap is killed and its output
        # is lost. Stopping one cell early keeps everything finished so far;
        # the remaining cells are picked up by a rerun.
        if args.time_budget_min is not None:
            used = (time.time() - t_begin) / 60
            if used + longest * 1.15 > args.time_budget_min:
                print(f"TIME BUDGET: {used:.0f} min used, next cell needs about "
                      f"{longest * 1.15:.0f}; stopping cleanly before {c['cell']}")
                break
        stego_dir = data / c["stego_dir"]
        stems = sorted((p.stem for p in stego_dir.glob("*.jpg")),
                       key=lambda s: (len(s), s))
        # Exact pairing only. SCRIPT cells ship 299 stegos against 300 covers;
        # dropping the unmatched cover keeps the pairing exact. Never pad or
        # duplicate to make the counts line up.
        stems = [s for s in stems if (cover_dir / f"{s}.jpg").exists()]
        print(f"\n{c['cell']}  ({len(stems)} pairs)", flush=True)

        t0 = time.time()
        try:
            r = run_cell(cover_dir, stego_dir, stems, device, args.epochs,
                         args.batch_size, args.lr,
                         out / "scores" / f"{c['cell']}.csv")
            entry = dict(cell=c["cell"], method=c["method"],
                         payload_bpnzac=c["payload_bpnzac"], arm=c["arm"],
                         detector="EfficientNet-B0", status="ok",
                         seeds=SEEDS, epochs=args.epochs,
                         train_minutes=round((time.time() - t0) / 60, 1), **r)
            longest = max(longest, (time.time() - t0) / 60)
            print(f"  -> P_E {entry['p_e_mean']:.4f} +/- {entry['p_e_std']:.4f}"
                  f"   (chance = {entry['p_e_chance_level']:.4f})")
        except Exception as exc:  # noqa: BLE001
            # Record failures as failures. Never substitute a default value -
            # that is exactly the defect we found in four published drivers.
            entry = dict(cell=c["cell"], method=c["method"],
                         payload_bpnzac=c["payload_bpnzac"], arm=c["arm"],
                         detector="EfficientNet-B0", status="failed",
                         error=str(exc)[:300])
            print(f"  FAILED: {exc}")

        results.append(entry)
        # Checkpoint after every cell so a session timeout costs one cell, not all.
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nwrote {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
