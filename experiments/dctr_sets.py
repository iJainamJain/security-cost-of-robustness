"""Shared plumbing for the later DCTR experiments (exp40, exp41): chunked feature
extraction that can be resumed, and scoring a list of stego sets against one
cover set with the detector of the main tables (exp16.run_cell: FLD ensemble,
d_sub 800, L 200, five paired 50/50 splits).

Nothing here is new method; exp37 does the same for one fixed set of cells.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import exp16_dctr_fullscale as e16  # noqa: E402

FCHUNK = 500


def extract(fdir: pathlib.Path, name: str, files: list[tuple[str, pathlib.Path]], qf: int,
            note) -> None:
    """Features of `files` ((stem, path) in order) into fdir/name/chunk_XXX.npz."""
    todo = [k for k in range((len(files) + FCHUNK - 1) // FCHUNK)
            if not (fdir / name / f"chunk_{k:03d}.npz").exists()]
    if not todo:
        return
    import torch

    from dctr import DCTR, DIM, read_jpeg
    torch.set_num_threads(8)
    ex = DCTR(qf, "cpu")
    (fdir / name).mkdir(parents=True, exist_ok=True)
    for k in todo:
        part = files[k * FCHUNK:(k + 1) * FCHUNK]
        t0 = time.time()
        feats = np.empty((len(part), DIM), dtype=np.float32)
        for j, (_, p) in enumerate(part):
            feats[j] = ex.features(*read_jpeg(p))
        dest = fdir / name / f"chunk_{k:03d}.npz"
        tmp = dest.with_suffix(".tmp.npz")
        np.savez(tmp, feats=feats, stems=np.array([s for s, _ in part]))
        tmp.replace(dest)
        note(f"  features {name} chunk {k}: {len(part)} images, "
             f"{(time.time() - t0) / max(len(part), 1):.2f} s/img")


def load(fdir: pathlib.Path, name: str, n_chunks: int | None = None):
    fs = sorted(f for f in (fdir / name).glob("chunk_*.npz") if ".tmp" not in f.name)
    if n_chunks is not None:
        fs = fs[:n_chunks]
    feats, stems = [], []
    for f in fs:
        z = np.load(f)
        feats.append(z["feats"]); stems += [str(s) for s in z["stems"]]
    return np.concatenate(feats), stems


def null_test(out: pathlib.Path, Xcov, cov_stems, note, seeds: int = 5) -> dict:
    f = out / "null_test.json"
    if not f.exists():
        out.mkdir(parents=True, exist_ok=True)
        # Random groups: BOSSbase is ordered by source (FINDINGS F9).
        perm = np.random.default_rng(20261002).permutation(len(cov_stems))
        half = len(cov_stems) // 2
        ia, ib = np.sort(perm[:half]), np.sort(perm[half:2 * half])
        note("  NULL TEST (cover vs cover, random groups)")
        r, rows = e16.run_cell(Xcov[ia], Xcov[ib], [cov_stems[i] for i in ia], seeds, 800,
                               200, note)
        e16.write_scores(out / "scores" / "NULL_TEST.csv", rows)
        f.write_text(json.dumps(r, indent=2), encoding="utf-8")
        note(f"  NULL P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}")
    return json.loads(f.read_text())


def score(out: pathlib.Path, Xcov, cov_stems, sets: list[tuple[str, dict, object]], note,
          seeds: int = 5) -> list[dict]:
    """sets: (name, extra fields for the result row, loader returning (X, stems)).
    Only pairs whose stem is in both the cover and the stego set are used."""
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else []
    done = {r["cell"] for r in results}
    pos = {s: i for i, s in enumerate(cov_stems)}
    for name, extra, loader in sets:
        if name in done:
            continue
        Xs, stems = loader()
        keep = [i for i, s in enumerate(stems) if s in pos]
        stems, Xs = [stems[i] for i in keep], Xs[keep]
        note(f"  {name} ({len(stems)} pairs)")
        r, rows = e16.run_cell(Xcov[[pos[s] for s in stems]], Xs, stems, seeds, 800, 200, note)
        e16.write_scores(out / "scores" / f"{name}.csv", rows)
        results.append(dict(cell=name, detector="DCTR+FLD", **extra, **r))
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        note(f"  -> {name} P_E {r['p_e_mean']:.4f} +/- {r['p_e_std']:.4f}  "
             f"caught at 1% FA {100 * (1 - r['md_at_fa1_mean']):.1f}%")
    return results
