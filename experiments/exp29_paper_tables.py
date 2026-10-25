"""Headline tables for the paper, built only from results files.

    python experiments/exp29_paper_tables.py
    python experiments/exp29_paper_tables.py --update-draft

Writes results/exp29_paper_tables/paper_tables.json (read by
exp30_paper_figures.py) and paper_tables.md (the tables as they go into the
draft). Rerun whenever a results folder changes - e.g. when a new SRNet session
is unpacked under results/m4_srnet_<X>/ - and the tables follow. With
--update-draft, Table 3 is also rewritten in PAPER1_DRAFT.md, between its
TABLE3 markers (the only table still waiting for results).

Table 1  Security at 10,000 images, DCTR + FLD (exp16, exp21).
Table 2  Robustness over the same 10,000 images (exp13, exp21 per-image CSVs).
Table 5  Second image source (256x256): DCTR (exp25) and SRNet
         (m4_srnet_256_<session>) on BOSSbase-256 and BOWS2-256.
Table 3  Four detector settings at 5,000 images: DCTR fixed and tuned (exp16,
         exp21, exp26), EfficientNet-B0 (m4_5k_a/b, m4_script_r0_cnn) and SRNet
         with curriculum (m4_srnet_<session>).
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import paths  # noqa: E402

RES = paths.RESULTS
N_FULL = 10000

# (label, cell prefix, embedding source folder, channel the method targets)
METHODS = [
    ("J-UNIWARD", "J-UNIWARD", "exp13_fullscale", "none"),
    ("GMAS", "GMAS", "exp13_fullscale", "QF 65 -> 71"),
    ("ROAST", "ROAST", "exp13_fullscale", "QF 65 -> 71"),
    ("MINICER", "MINICER", "exp13_fullscale", "QF 65 -> 65"),
    ("SCRIPT, R = 0", "SCRIPT-R0", "exp21_script_low_fullscale", "QF 65 -> 65"),
    ("SCRIPT, R = 10", "SCRIPT", "exp13_fullscale", "QF 65 -> 65"),
]
PAYLOADS = (0.05, 0.1, 0.2, 0.3)
SOURCES_256 = ("BOSSbase256", "BOWS2_256")


def load(rel):
    return json.loads((RES / rel).read_text(encoding="utf-8"))


def by_cell(rows):
    return {r["cell"]: r for r in rows}


def row_ok(r):
    if "embed_ok" in r:                      # GMAS writes 1/0, the others "ok"
        return (r["embed_ok"] or "").strip() not in ("", "0")
    return (r.get("status") or "").strip().lower() == "ok"


def robustness(folder, cell):
    d = RES / folder / cell / "csv"
    if not d.exists():
        return None
    seen, ok, clean, ber, psnr = set(), 0, 0, [], []
    for f in sorted(d.glob("chunk_*.csv")):
        with f.open(newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                stem = (r.get("cover") or "").strip()
                if not stem.isdigit() or int(stem) > N_FULL or stem in seen:
                    continue
                seen.add(stem)
                if not row_ok(r):
                    continue
                ok += 1
                try:
                    b = float(r["bit_error_rate"])
                    ber.append(b)
                    clean += b == 0
                except (KeyError, ValueError, TypeError):
                    pass
                try:
                    psnr.append(float(r["psnr_db"]))
                except (KeyError, ValueError, TypeError):
                    pass
    return dict(images=len(seen), embedded=ok, error_free=int(clean),
                error_free_rate=clean / len(seen) if seen else None,
                mean_ber=statistics.fmean(ber) if ber else None,
                # Median, and finite values only: an image the embedder leaves
                # unchanged has infinite PSNR and would swamp a mean.
                median_psnr_db=statistics.median(x for x in psnr if math.isfinite(x))
                if psnr else None,
                unchanged_images=sum(1 for x in psnr if not math.isfinite(x)))


def srnet_rows():
    """One row per (cell, seed) over every unpacked 512x512 session."""
    seen = {}
    for f in sorted(RES.glob("m4_srnet_*/srnet_results/seed_results.json")):
        # Sessions A ... F only: not the from-scratch speed test, and not the
        # 256x256 runs, whose cells carry the same names. (glob is not
        # case-sensitive on Windows, hence the explicit pattern.)
        if not re.fullmatch(r"m4_srnet_[A-F][0-9A-Z]*", f.parts[-3]):
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            seen.setdefault((r["cell"], r["seed"]), dict(r, session=f.parts[-3]))
    out = {}
    for (cell, _), r in sorted(seen.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.setdefault(cell, []).append(r)
    return out


def srnet_256():
    """source -> cell -> one row per seed, over the unpacked 256x256 sessions.
    The source is read from the data path each session recorded in its plan."""
    out = {}
    for d in sorted(RES.glob("m4_srnet_256_*/srnet_results")):
        plans = [t for f in sorted(d.glob("plan_gpu*.json"))
                 for t in json.loads(f.read_text(encoding="utf-8"))]
        srcs = {s for t in plans for s in SOURCES_256 if "stego_" + s + "_" in t.get("data", "")}
        if len(srcs) != 1 or not (d / "seed_results.json").exists():
            continue
        cells = out.setdefault(srcs.pop(), {})
        for r in json.loads((d / "seed_results.json").read_text(encoding="utf-8")):
            rows = cells.setdefault(r["cell"], [])
            if all(x["seed"] != r["seed"] for x in rows):
                rows.append(dict(r, session=d.parts[-2]))
    for cells in out.values():
        for rows in cells.values():
            rows.sort(key=lambda r: r["seed"])
    return out


def mean_sd(v):
    return statistics.fmean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)


def main() -> int:
    out = paths.ensure_results("exp29_paper_tables")
    d10 = by_cell(load("exp16_dctr_fullscale/tier_10000/results.json"))
    d10.update(by_cell(load("exp21_script_low_fullscale/tier_10000/results.json")))
    d5 = by_cell(load("exp16_dctr_fullscale/tier_5000/results.json"))
    d5.update(by_cell(load("exp21_script_low_fullscale/tier_5000/results.json")))
    tuned = by_cell(load("exp26_tuned_ensemble/results.json"))
    eff = {}
    for s in ("m4_5k_a", "m4_5k_b", "m4_script_r0_cnn"):
        eff.update(by_cell(load(f"{s}/scale_results/cnn/results.json")))
    sr = srnet_rows()
    nulls = dict(
        dctr_10000=load("exp16_dctr_fullscale/tier_10000/null_test.json"),
        dctr_5000=load("exp16_dctr_fullscale/tier_5000/null_test.json"),
        dctr_tuned_5000=load("exp26_tuned_ensemble/null_test.json"),
        effnet_5000=load("m4_5k_a/scale_results/cnn/null_test.json"))
    nulls = {k: dict(p_e_mean=v["p_e_mean"], p_e_std=v["p_e_std"],
                     n_test_pairs=v["n_test_pairs"]) for k, v in nulls.items()}

    security, robust, detectors = [], [], []
    for label, prefix, folder, channel in METHODS:
        for p in PAYLOADS:
            cell = f"{prefix}_{p}"
            if cell in d10:
                r = d10[cell]
                security.append(dict(
                    method=label, cell=cell, payload=p, p_e=r["p_e_mean"],
                    p_e_sd=r["p_e_std"], caught_at_fa1=1 - r["md_at_fa1_mean"],
                    n_train_pairs=r["n_train_pairs"], n_test_pairs=r["n_test_pairs"]))
            rb = robustness(folder, cell)
            if rb:
                robust.append(dict(method=label, cell=cell, payload=p, channel=channel, **rb))
            if cell not in d5 or (cell not in eff and cell not in sr):
                continue
            row = dict(method=label, cell=cell, payload=p,
                       dctr_fixed=d5[cell]["p_e_mean"], dctr_fixed_sd=d5[cell]["p_e_std"],
                       dctr_fixed_caught=1 - d5[cell]["md_at_fa1_mean"])
            if cell in tuned:
                t = tuned[cell]
                row.update(dctr_tuned=t["p_e_mean"], dctr_tuned_sd=t["p_e_std"],
                           dctr_tuned_caught=1 - t["md_at_fa1_mean"],
                           d_sub=t["d_sub_per_seed"])
            if cell in eff:
                e = eff[cell]
                row.update(effnet=e["p_e_mean"], effnet_sd=e["p_e_std"],
                           effnet_caught=1 - e["md_at_fa1_mean"],
                           effnet_seeds=len(e["seeds"]), effnet_chance=e["p_e_chance_level"])
            if cell in sr:
                m, sd = mean_sd([r["test_p_e"] for r in sr[cell]])
                row.update(srnet=m, srnet_sd=sd, srnet_seeds=len(sr[cell]),
                           srnet_per_seed=[r["test_p_e"] for r in sr[cell]],
                           srnet_caught=1 - statistics.fmean(r["md_at_fa1"] for r in sr[cell]),
                           srnet_chance=sr[cell][0]["p_e_chance_level"])
            # Best warden: lowest mean. Called a tie when the runner-up is within
            # two standard deviations of the noisier of the two.
            cand = [(row[k], row.get(k + "_sd", 0.0), name) for k, name in
                    (("dctr_tuned", "DCTR"), ("effnet", "EfficientNet-B0"), ("srnet", "SRNet"))
                    if k in row]
            if "dctr_tuned" not in row:
                cand.append((row["dctr_fixed"], row["dctr_fixed_sd"], "DCTR"))
            cand.sort()
            row.update(best_p_e=cand[0][0], best_detector=cand[0][2],
                       best_is_tie=(len(cand) > 1 and cand[1][0] - cand[0][0]
                                    < 2 * max(cand[0][1], cand[1][1])),
                       runner_up=cand[1][2] if len(cand) > 1 else None)
            detectors.append(row)

    # Second source: every 256x256 cell, with the 512x512 figures beside it.
    d256 = {s: by_cell(load(f"exp24_second_source/{s}/dctr/results.json")) for s in SOURCES_256}
    s256 = srnet_256()
    det512 = {r["cell"]: r for r in detectors}
    second = []
    for label, prefix, _, _ in METHODS:
        for p in PAYLOADS:
            cell = f"{prefix}_{p}"
            if cell not in d256[SOURCES_256[0]]:
                continue
            row = dict(method=label, cell=cell, payload=p,
                       dctr_512=d5[cell]["p_e_mean"], srnet_512=det512.get(cell, {}).get("srnet"))
            for src in SOURCES_256:
                row[f"dctr_{src}"] = d256[src][cell]["p_e_mean"]
                row[f"dctr_sd_{src}"] = d256[src][cell]["p_e_std"]
                seeds = s256.get(src, {}).get(cell, [])
                if seeds:
                    m, sd = mean_sd([r["test_p_e"] for r in seeds])
                    row.update({f"srnet_{src}": m, f"srnet_sd_{src}": sd,
                                f"srnet_per_seed_{src}": [r["test_p_e"] for r in seeds],
                                f"srnet_seeds_{src}": [r["seed"] for r in seeds]})
            second.append(row)

    tables = dict(nulls=nulls, security_10000=security, robustness_10000=robust,
                  detectors_5000=detectors, second_source_5000=second)
    (out / "paper_tables.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")

    def f(x, sd=None, nd=3):
        if x is None:
            return "-"
        return f"{x:.{nd}f}" + (f" ± {sd:.{nd}f}" if sd is not None else "")

    md = ["# Paper tables (generated by experiments/exp29_paper_tables.py)", ""]
    n = nulls["dctr_10000"]
    md += ["## Table 1 - Security at 10,000 images, DCTR + FLD (d_sub 800, L 200)", "",
           f"P_E, mean ± sd over five paired 5,000 / 5,000 splits. Cover-against-cover "
           f"null: {f(n['p_e_mean'], n['p_e_std'], 4)} ({n['n_test_pairs']} test pairs).", "",
           "| Payload | " + " | ".join(m[0] for m in METHODS) + " |",
           "|---|" + "---|" * len(METHODS)]
    sec = {(r["method"], r["payload"]): r for r in security}
    for p in PAYLOADS:
        md.append(f"| {p} | " + " | ".join(
            f(sec[(m[0], p)]["p_e"], sec[(m[0], p)]["p_e_sd"], 4) if (m[0], p) in sec else "-"
            for m in METHODS) + " |")
    md += ["", "Stego images caught at a 1% false-alarm rate:", "",
           "| Payload | " + " | ".join(m[0] for m in METHODS) + " |",
           "|---|" + "---|" * len(METHODS)]
    for p in PAYLOADS:
        md.append(f"| {p} | " + " | ".join(
            f"{100 * sec[(m[0], p)]['caught_at_fa1']:.1f}%" if (m[0], p) in sec else "-"
            for m in METHODS) + " |")

    md += ["", "## Table 2 - Robustness over the same 10,000 images", "",
           "| Method | Channel | Payload | Embedded | Error-free | Rate | Mean BER | Median PSNR (dB) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in robust:
        if r["channel"] == "none":
            md.append(f"| {r['method']} | none | {r['payload']} | {r['embedded']} | n/a | n/a "
                      f"| n/a | {f(r['median_psnr_db'], nd=1)} |")
        else:
            md.append(f"| {r['method']} | {r['channel']} | {r['payload']} | {r['embedded']} "
                      f"| {r['error_free']} | {100 * r['error_free_rate']:.1f}% "
                      f"| {f(r['mean_ber'], nd=4)} | {f(r['median_psnr_db'], nd=1)} |")

    t3_start = len(md) + 4
    md += ["", "## Table 3 - Detectors at 5,000 images", "",
           f"DCTR: five paired 2,500 / 2,500 splits (null {f(nulls['dctr_5000']['p_e_mean'], nulls['dctr_5000']['p_e_std'])}; "
           f"tuned {f(nulls['dctr_tuned_5000']['p_e_mean'], nulls['dctr_tuned_5000']['p_e_std'])}). "
           f"CNNs: paired 2,000 / 500 / 2,500 split per seed, epoch chosen on validation "
           f"(EfficientNet null {f(nulls['effnet_5000']['p_e_mean'], nulls['effnet_5000']['p_e_std'])}; "
           f"chance at 2,500 test pairs 0.491). Networks: three seeds per cell (the seed counts are in paper_tables.json).", "",
           "| Cell | DCTR, fixed | DCTR, tuned | EfficientNet-B0 | SRNet, curriculum | Best warden |",
           "|---|---|---|---|---|---|"]
    name = lambda r: f"{r['method'].replace(', R = ', ' R=')} {r['payload']}"  # noqa: E731
    for r in detectors:
        best = r["best_detector"] + (f" (tie with {r['runner_up']})" if r["best_is_tie"] else "")
        md.append(
            f"| {name(r)} | {f(r['dctr_fixed'], r['dctr_fixed_sd'])} "
            f"| {f(r.get('dctr_tuned'), r.get('dctr_tuned_sd')) if 'dctr_tuned' in r else '-'} "
            f"| {f(r['effnet'], r['effnet_sd']) if 'effnet' in r else '-'} "
            f"| {f(r['srnet'], r['srnet_sd']) if 'srnet' in r else '-'} "
            f"| {f(r['best_p_e'])}, {best} |")
    t3 = md[t3_start:]
    md += ["", "Stego images caught at a 1% false-alarm rate, same cells:", "",
           "| Cell | DCTR, tuned | EfficientNet-B0 | SRNet |", "|---|---|---|---|"]
    for r in detectors:
        c = lambda k: f"{100 * r[k]:.1f}%" if k in r else "-"   # noqa: E731
        md.append(f"| {name(r)} | "
                  f"{c('dctr_tuned_caught') if 'dctr_tuned' in r else c('dctr_fixed_caught')} "
                  f"| {c('effnet_caught')} | {c('srnet_caught')} |")
    md += ["", "## Table 5 - Second image source, 5,000 images each", ""]
    t5_start = len(md)
    md += ["| Cell | BOSSbase-256: DCTR | SRNet | BOWS2-256: DCTR | SRNet | BOSSbase-512: DCTR | SRNet |",
           "|---|---|---|---|---|---|---|"]
    for r in second:
        def sr5(src):
            if f"srnet_{src}" not in r:
                return "-"
            return f(r[f"srnet_{src}"], r[f"srnet_sd_{src}"])
        md.append(f"| {name(r)} | {f(r['dctr_BOSSbase256'])} | {sr5('BOSSbase256')} "
                  f"| {f(r['dctr_BOWS2_256'])} | {sr5('BOWS2_256')} "
                  f"| {f(r['dctr_512'])} | {f(r['srnet_512'])} |")
    t5 = md[t5_start:]
    (out / "paper_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    if "--update-draft" in sys.argv:
        draft = paths.ROOT / "PAPER1_DRAFT.md"
        s = draft.read_text(encoding="utf-8")
        for tag, lines in (("TABLE3", t3), ("TABLE5", t5)):
            if f"<!-- {tag}:BEGIN" not in s:
                continue
            a, b = s.index(f"<!-- {tag}:BEGIN"), s.index(f"<!-- {tag}:END -->")
            a = s.index("\n", a) + 1
            s = s[:a] + "\n" + "\n".join(lines).strip("\n") + "\n\n" + s[b:]
            print(f"{tag} rewritten in {draft.name}")
        draft.write_text(s, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
