"""Why does SCRIPT's detectability at strength 0 hardly move with payload?

At R = 0, DCTR reads P_E 0.394 / 0.382 / 0.373 at 0.05 / 0.1 / 0.2 bpnzac
(F21, F25). A detectability that does not grow with payload suggests a cost
paid once per image. For ROAST that cost is a preprocessing step (F15), and a
near-empty message exposes it. SCRIPT has no such step - generateStegoGISS.m
writes the cover's coefficients back with only the coder's changes - and at
R = 0 it cannot embed a near-empty message at all (F16). So the question is
asked of the stego images we already have:

  stage 1 (this file, no embedding): for every image, how many coefficients
  differ from the cover at each payload, which ones (zero AC, non-zero AC, DC),
  and how that count scales with the message length. Changes proportional to
  message bits mean no fixed mark; a count that does not shrink with the
  message means the coder itself leaves one.

    python experiments/exp32_script_footprint.py            # every 5th image
    python experiments/exp32_script_footprint.py --step 1   # all 10,000
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

R = paths.RESULTS
COVERS = paths.DATA / "BOSSbase_QF65_full"
E13, E21 = R / "exp13_fullscale", R / "exp21_script_low_fullscale"
CELLS = [  # (label, strength, payload, folder)
    ("R0", 0, 0.05, E21 / "SCRIPT-R0_0.05"), ("R0", 0, 0.1, E21 / "SCRIPT-R0_0.1"),
    ("R0", 0, 0.2, E21 / "SCRIPT-R0_0.2"),
    ("R10", 10, 0.01, E21 / "SCRIPT_0.01"), ("R10", 10, 0.02, E21 / "SCRIPT_0.02"),
    ("R10", 10, 0.05, E13 / "SCRIPT_0.05"), ("R10", 10, 0.1, E13 / "SCRIPT_0.1"),
    ("R10", 10, 0.2, E13 / "SCRIPT_0.2"),
]


def read_y(path) -> np.ndarray:
    import jpeglib
    return np.asarray(jpeglib.read_dct(str(path)).Y).astype(np.int32)   # (bh, bw, 8, 8)


def csv_info(folder: pathlib.Path) -> dict[str, dict]:
    out = {}
    for f in sorted((folder / "csv").glob("*.csv")):
        with f.open(newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                if (r.get("status") or "").strip().lower() == "ok":
                    out[r["cover"].strip()] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5, help="use every step-th image")
    ap.add_argument("--n", type=int, default=10000)
    args = ap.parse_args()
    out = paths.ensure_results("exp32_script_footprint")
    stems = [str(i) for i in range(1, args.n + 1, args.step)]
    info = {(lab, p): csv_info(folder) for lab, _, p, folder in CELLS}

    rows, t0 = [], time.time()
    for k, s in enumerate(stems):
        c = read_y(COVERS / f"{s}.jpg")
        dc = np.zeros(c.shape, bool)
        dc[:, :, 0, 0] = True
        nz_ac = int(np.count_nonzero(c[~dc]))
        for lab, strength, p, folder in CELLS:
            f = folder / "stego" / f"{s}.jpg"
            r = info[(lab, p)].get(s)
            if r is None or not f.exists():
                continue
            d = read_y(f) - c
            ch = d != 0
            rows.append(dict(
                image=int(s), setting=lab, payload=p, nz_ac=nz_ac,
                msg_bits=int(float(r["msg_bits"])), usable=int(float(r["usable_coeffs"])),
                changed=int(ch.sum()), changed_gt1=int((np.abs(d) > 1).sum()),
                changed_zero_ac=int((ch & (c == 0) & ~dc).sum()),
                changed_dc=int((ch & dc).sum()),
                made_zero=int((ch & (c != 0) & ((c + d) == 0)).sum()),
                blocks_touched=int(ch.any(axis=(2, 3)).sum())))
        if (k + 1) % 200 == 0:
            print(f"  {k + 1} of {len(stems)} images  ({time.time() - t0:.0f} s)", flush=True)

    with (out / "per_image.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = []
    print(f"\n{'setting':<8}{'payload':>8}{'images':>8}{'bits':>8}{'usable':>9}{'changed':>9}"
          f"{'per bit':>9}{'% at 0':>8}{'% DC':>7}{'% >1':>7}{'blocks':>8}")
    for lab, _, p, _ in CELLS:
        rr = [r for r in rows if r["setting"] == lab and r["payload"] == p]
        if not rr:
            continue
        a = {k: np.array([r[k] for r in rr], float) for k in rr[0] if k != "setting"}
        tot = a["changed"].sum()
        s = dict(setting=lab, payload=p, images=len(rr), mean_bits=a["msg_bits"].mean(),
                 mean_usable=a["usable"].mean(), mean_changed=a["changed"].mean(),
                 median_changed=float(np.median(a["changed"])),
                 changes_per_bit=tot / a["msg_bits"].sum(),
                 share_at_zero_ac=a["changed_zero_ac"].sum() / tot,
                 share_dc=a["changed_dc"].sum() / tot, share_gt1=a["changed_gt1"].sum() / tot,
                 mean_blocks_touched=a["blocks_touched"].mean(),
                 unchanged_images=int((a["changed"] == 0).sum()))
        summary.append(s)
        print(f"{lab:<8}{p:>8}{len(rr):>8}{s['mean_bits']:>8.0f}{s['mean_usable']:>9.0f}"
              f"{s['mean_changed']:>9.1f}{s['changes_per_bit']:>9.3f}"
              f"{100 * s['share_at_zero_ac']:>7.1f}%{100 * s['share_dc']:>6.1f}%"
              f"{100 * s['share_gt1']:>6.1f}%{s['mean_blocks_touched']:>8.0f}")

    # Per image, changed = a + b * bits over the payloads of one setting. A fixed
    # mark shows as an intercept that is a large share of the count at 0.05.
    fits = {}
    for lab in ("R0", "R10"):
        by = {}
        for r in rows:
            if r["setting"] == lab:
                by.setdefault(r["image"], []).append((r["msg_bits"], r["changed"]))
        a_, share = [], []
        for pts in by.values():
            if len(pts) < 3:
                continue
            x, y = np.array(pts, float).T
            b, a = np.polyfit(x, y, 1)
            a_.append(a)
            share.append(a / y.min() if y.min() > 0 else np.nan)
        fits[lab] = dict(images=len(a_), intercept_mean=float(np.mean(a_)),
                         intercept_median=float(np.median(a_)),
                         intercept_share_of_smallest_median=float(np.nanmedian(share)))
        print(f"\n{lab}: changed = a + b * bits per image over its payloads; intercept a: "
              f"mean {np.mean(a_):.1f}, median {np.median(a_):.1f} coefficients "
              f"({100 * np.nanmedian(share):.0f}% of the count at the lowest payload, median)")
    (out / "summary.json").write_text(json.dumps(dict(cells=summary, fits=fits), indent=2),
                                      encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
