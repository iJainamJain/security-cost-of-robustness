"""M3 Arm A — the security-vs-robustness Pareto front.

The figure Paper 1 exists to produce. For each method at a given payload it
measures two things that are deliberately taken from *different* images:

* **Security** — DCTR + FLD ensemble P_E on the stego image **as transmitted**,
  i.e. before the channel. This is the right threat model: the warden intercepts
  the upload, not the recompressed copy that arrives.
* **Robustness** — extraction bit error rate **after** the channel.

Plotting one against the other is the whole point. Robust methods buy the second
by spending the first, and nobody has drawn the trade-off on a common protocol.

J-UNIWARD is included as the non-robust ceiling: maximum security, zero
robustness. It marks how much each robust method has given away.

    python experiments/exp08_arm_a.py --count 300 --payloads 0.1
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import statistics
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402
from ensemble import EnsembleFLD  # noqa: E402
from metrics import min_p_error  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = paths.BASELINES / "Robust_Steganography"
DIM = 8000

# (label, matlab function, needs_alpha_beta)
METHODS = [
    ("J-UNIWARD", "juniward_embed_shard", False),   # non-robust ceiling
    ("GMAS", "gmas_embed_shard", False),
    ("ROAST-recon", "roast_recon_shard", True),
]


def find_matlab() -> str:
    exe = shutil.which("matlab")
    if exe:
        return exe
    f = sorted(pathlib.Path("C:/Program Files/MATLAB").glob("R*/bin/matlab.exe"))
    if not f:
        raise SystemExit("MATLAB not found")
    return str(f[-1])


def matlab(cmd: str) -> None:
    subprocess.run([find_matlab(), "-batch", cmd], cwd=str(HERE),
                   capture_output=True, text=True)


def load_features(path: pathlib.Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float64).reshape(-1, DIM)


def p_error(Xc: np.ndarray, Xs: np.ndarray, repeats: int = 5,
            d_sub: int = 800, n_est: int = 200) -> tuple[float, float]:
    """Paired-split P_E. Cover i and stego i never straddle the split."""
    out = []
    n = min(len(Xc), len(Xs))
    for seed in range(repeats):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        X = np.vstack([Xc[tr], Xs[tr]])
        y = np.concatenate([np.zeros(len(tr)), np.ones(len(tr))])
        m = EnsembleFLD(d_sub=d_sub, n_estimators=n_est, seed=seed).fit(X, y)
        out.append(min_p_error(m.decision_function(Xc[te]),
                               m.decision_function(Xs[te])))
    return float(np.mean(out)), float(np.std(out))


def write_manifest(paths_list, dest: pathlib.Path) -> pathlib.Path:
    dest.write_text("\n".join(str(p) for p in paths_list), encoding="utf-8")
    return dest


def cache_valid(csv_path: pathlib.Path, expected: int) -> bool:
    """A cached CSV is only reusable if it covers the requested image count.

    Without this, `--count 300` silently reuses a 40-cover CSV from an earlier
    smoke test and reports 40 images' results as 300. Found the hard way.
    """
    if not csv_path.exists():
        return False
    with csv_path.open(newline="", encoding="utf-8") as fh:
        n = sum(1 for _ in csv.DictReader(fh))
    if n < expected:
        print(f"  cache miss: {csv_path.name} has {n} rows, need {expected}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--payloads", type=float, nargs="+", default=[0.1])
    ap.add_argument("--cover-qf", type=int, default=65)
    ap.add_argument("--attack-qf", type=int, default=71)
    args = ap.parse_args()

    out = paths.ensure_results("exp08_arm_a")
    cover_dir = paths.DATA / f"BOSSbase_QF{args.cover_qf}_arma"

    # 1. covers
    if not cover_dir.exists() or len(list(cover_dir.glob("*.jpg"))) < args.count:
        subprocess.run([sys.executable, str(HERE / "prep_covers.py"),
                        "--qf", str(args.cover_qf), "--count", str(args.count),
                        "--tag", "arma"], check=True)
    covers = sorted(cover_dir.glob("*.jpg"),
                    key=lambda p: (len(p.stem), p.stem))[:args.count]
    cov_man = write_manifest(covers, out / "covers.txt")
    print(f"{len(covers)} covers at QF{args.cover_qf}")

    # 2. cover features (once)
    cov_feat = out / "cover.f64"
    if not cov_feat.exists():
        print("extracting cover features...", flush=True)
        matlab(f"dctr_extract('{cov_man.as_posix()}','{cov_feat.as_posix()}',"
               f"{args.cover_qf})")
    Xc = load_features(cov_feat)

    rows = []
    for payload in args.payloads:
        for label, fn, needs_ab in METHODS:
            # VERIFIED 2026-08-08: none of these methods' stego images depend on
            # the channel quality factor. `gmas(cover_Path, rho1_P, rho1_M,
            # tab_m)` takes only the *cover* quantisation table; attack_qf enters
            # solely at the channel-simulation step. Byte-comparing the QF71 and
            # QF95 stego JPEGs confirms they are identical, as are their DCTR
            # features and hence their P_E.
            #
            # So **security is channel-invariant here and robustness is not**.
            # We still key the embedder's CSV by attack_qf because it carries the
            # post-channel BER, which genuinely differs; the feature/P_E half of
            # the work is redundant across channels and could be shared. Kept
            # separate for now because the cost is bounded and the alternative
            # is a cache that silently returns another channel's robustness.
            channel_dep = fn != "juniward_embed_shard"
            slug = (f"{label.replace('-', '_')}_p{payload}"
                    + (f"_q{args.attack_qf}" if channel_dep else ""))
            csv_path = out / f"{slug}.csv"
            sdir = out / slug

            if not cache_valid(csv_path, len(covers)):
                print(f"embedding {label} @ {payload}...", flush=True)
                if fn == "juniward_embed_shard":
                    call = (f"{fn}('{REPO.as_posix()}','{cov_man.as_posix()}',"
                            f"'{sdir.as_posix()}','{csv_path.as_posix()}',{payload},0)")
                elif needs_ab:
                    call = (f"{fn}('{REPO.as_posix()}','{cov_man.as_posix()}',"
                            f"'{sdir.as_posix()}','{csv_path.as_posix()}',{payload},"
                            f"{args.cover_qf},{args.attack_qf},0,0.2,0.7)")
                else:
                    call = (f"{fn}('{REPO.as_posix()}','{cov_man.as_posix()}',"
                            f"'{sdir.as_posix()}','{csv_path.as_posix()}',{payload},"
                            f"{args.cover_qf},{args.attack_qf},0)")
                matlab(call)

            stegos = sorted((sdir / "stego").glob("*.jpg"),
                            key=lambda p: (len(p.stem), p.stem))
            if not stegos:
                print(f"  {label}: no stego output, skipping")
                continue

            feat = out / f"{slug}.f64"
            if not feat.exists():
                print(f"  extracting {label} features...", flush=True)
                man = write_manifest(stegos, out / f"{slug}_manifest.txt")
                matlab(f"dctr_extract('{man.as_posix()}','{feat.as_posix()}',"
                       f"{args.cover_qf})")

            Xs = load_features(feat)
            pe, sd = p_error(Xc[:len(Xs)], Xs)

            # robustness, from the embedder's own CSV
            with csv_path.open(newline="", encoding="utf-8") as fh:
                recs = list(csv.DictReader(fh))
            bers = [float(r["bit_error_rate"]) for r in recs
                    if r.get("bit_error_rate")]
            ber = statistics.mean(bers) if bers else float("nan")
            errorless = sum(1 for b in bers if b == 0)

            rows.append(dict(method=label, payload=payload,
                             attack_qf=args.attack_qf, p_e=pe, p_e_sd=sd,
                             mean_ber=ber, errorless=errorless, n=len(bers)))
            print(f"  {label:<12} P_E {pe:.4f} +/- {sd:.4f}   "
                  f"BER {ber:.5f}   errorless {errorless}/{len(bers)}")

    res = out / f"arm_a_q{args.attack_qf}.json"
    res.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n{'method':<14}{'payload':>8}{'P_E':>9}{'mean BER':>11}{'errorless':>11}")
    print("-" * 53)
    for r in rows:
        print(f"{r['method']:<14}{r['payload']:>8}{r['p_e']:>9.4f}"
              f"{r['mean_ber']:>11.5f}{r['errorless']:>7}/{r['n']}")
    print(f"\nJ-UNIWARD is the non-robust ceiling: highest P_E, no channel "
          f"survival.\nThe drop from it to each robust method is what that "
          f"method paid for robustness.")
    print(f"\nwrote {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
