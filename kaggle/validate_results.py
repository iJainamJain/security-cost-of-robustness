"""Validate an m4_results/ folder before sending it back.

Run this yourself; it is faster than a round-trip with us. It checks structure,
plausibility, and consistency against our DCTR baseline.

    python code/validate_results.py --results m4_results/ --baseline 04_OUR_BASELINE.md

Exit code 0 = clean, 1 = problems found. WARNINGs are judgement calls; ERRORs
mean the results are not usable as-is.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

EXPECTED_CELLS = [
    "J-UNIWARD_0.05", "J-UNIWARD_0.1", "J-UNIWARD_0.2", "J-UNIWARD_0.3",
    "GMAS_0.05", "GMAS_0.1", "GMAS_0.2", "GMAS_0.3",
    "ROAST_0.05", "ROAST_0.1", "ROAST_0.2", "ROAST_0.3",
    "SCRIPT_0.05", "SCRIPT_0.1", "SCRIPT_0.2",
    "MINICER_0.05", "MINICER_0.1", "MINICER_0.2",
    "StegoTool_0.01",
]

REQUIRED_FIELDS = ["cell", "method", "payload_bpnzac", "arm", "detector", "status"]
OK_FIELDS = ["p_e_mean", "p_e_std", "p_e_per_seed",
             "md_at_fa1_mean", "md_at_fa1_std"]

errors: list[str] = []
warnings: list[str] = []
notes: list[str] = []


def err(m: str) -> None:
    errors.append(m)


def warn(m: str) -> None:
    warnings.append(m)


def load_baseline(path: pathlib.Path) -> dict[str, float]:
    """Parse the DCTR column out of 04_OUR_BASELINE.md."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*`([^`]+)`\s*\|.*?\|\s*\*\*([0-9.]+)\*\*", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def check_results(results: list[dict], baseline: dict[str, float]) -> None:
    by_cell: dict[str, list[dict]] = {}
    for r in results:
        for f in REQUIRED_FIELDS:
            if f not in r:
                err(f"entry missing required field '{f}': {r.get('cell', '?')}")
        by_cell.setdefault(r.get("cell", "?"), []).append(r)

    eff = {c: [r for r in v if r.get("detector") == "EfficientNet-B0"]
           for c, v in by_cell.items()}

    for cell in EXPECTED_CELLS:
        if cell not in by_cell:
            err(f"missing cell: {cell}")
        elif not eff.get(cell):
            err(f"{cell}: no EfficientNet-B0 entry (only {[r['detector'] for r in by_cell[cell]]})")

    for cell in by_cell:
        if cell not in EXPECTED_CELLS:
            warn(f"unexpected cell name '{cell}' (check spelling against MANIFEST.json)")

    seen_pe: dict[float, list[str]] = {}
    for r in results:
        cell = r.get("cell", "?")
        if r.get("status") == "failed":
            warn(f"{cell}: status=failed ({r.get('error', 'no reason given')})")
            continue
        if r.get("status") != "ok":
            err(f"{cell}: status must be 'ok' or 'failed', got {r.get('status')!r}")
            continue

        for f in OK_FIELDS:
            if f not in r:
                err(f"{cell}: missing '{f}'")
        if "p_e_mean" not in r:
            continue

        pe = r["p_e_mean"]
        if not (0.0 <= pe <= 0.5):
            err(f"{cell}: p_e_mean {pe} outside [0, 0.5]. A min over "
                f"thresholds cannot exceed 0.5, so the metric itself is wrong "
                f"- use p_error() from the reference code verbatim.")

        # Inverted scores read as ~0.5 ("secure"), never as >0.5, so the range
        # check above cannot see them. The reversed-direction P_E can.
        rev, chance = r.get("p_e_reversed_mean"), r.get("p_e_chance_level")
        if rev is None:
            warn(f"{cell}: no p_e_reversed_mean - cannot rule out inverted "
                 f"scores. Rerun with the current reference code.")
        elif chance is not None and rev < chance - 0.06 and rev < pe - 0.05:
            err(f"{cell}: SCORES INVERTED - reversed P_E {rev:.4f} detects "
                f"but forward P_E {pe:.4f} does not. The reported value is "
                f"meaningless; higher score must mean MORE likely stego.")

        if r.get("epoch_selection") != "validation":
            err(f"{cell}: epoch_selection is {r.get('epoch_selection')!r}, "
                f"must be 'validation'. Choosing the epoch on the test set "
                f"biases P_E down by up to 0.07 (02_PITFALLS.md section 6b).")

        if chance is not None and pe >= chance - 0.02:
            notes.append(f"{cell}: P_E {pe:.4f} is at chance level "
                         f"({chance:.4f} for this test size) - not detected.")
        seeds = r.get("p_e_per_seed", [])
        if len(seeds) != 5:
            err(f"{cell}: expected 5 per-seed values, got {len(seeds)}")
        elif abs(sum(seeds) / 5 - pe) > 0.005:
            err(f"{cell}: p_e_mean {pe:.4f} does not match mean of "
                f"p_e_per_seed ({sum(seeds)/5:.4f})")
        if r.get("p_e_std", 0) > 0.15:
            warn(f"{cell}: p_e_std {r['p_e_std']:.3f} is high — unstable training?")

        seen_pe.setdefault(round(pe, 6), []).append(cell)

        base = baseline.get(cell)
        if base is not None and pe > base + 0.02:
            warn(f"{cell}: your P_E {pe:.4f} is HIGHER than our DCTR baseline "
                 f"{base:.4f}. A CNN should detect better, not worse. "
                 f"See 02_PITFALLS.md section 10.")

    for pe, cells in seen_pe.items():
        # Cells at the detection floor legitimately collide near zero, so only
        # flag repeats that are high enough to be a real copy-paste symptom.
        if len(cells) > 2 and pe > 0.01:
            warn(f"identical p_e_mean {pe} across {len(cells)} cells "
                 f"({', '.join(cells[:4])}...) — copy-paste error?")


def check_scores(root: pathlib.Path, results: list[dict]) -> None:
    sdir = root / "scores"
    if not sdir.is_dir():
        err("no scores/ directory — per-image scores are required")
        return
    for r in results:
        if r.get("status") != "ok":
            continue
        cell = r["cell"]
        f = sdir / f"{cell}.csv"
        if not f.exists():
            err(f"{cell}: missing scores/{cell}.csv")
            continue
        with f.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        need = {"seed", "image", "label", "score"}
        if not rows:
            err(f"{cell}: scores CSV is empty")
            continue
        if set(rows[0]) < need:
            err(f"{cell}: scores CSV columns {sorted(rows[0])} missing {sorted(need - set(rows[0]))}")
            continue
        labels = {r_["label"] for r_ in rows}
        if labels - {"cover", "stego"}:
            err(f"{cell}: unexpected labels {labels - {'cover', 'stego'}}")
        seeds = {r_["seed"] for r_ in rows}
        if len(seeds) != 5:
            warn(f"{cell}: scores cover {len(seeds)} seeds, expected 5")
        expected = r.get("n_test_pairs", 0) * 2 * 5
        if expected and abs(len(rows) - expected) > expected * 0.05:
            warn(f"{cell}: {len(rows)} score rows, expected ~{expected} "
                 f"(n_test_pairs x 2 x 5 seeds)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, default=pathlib.Path("m4_results"))
    ap.add_argument("--baseline", type=pathlib.Path,
                    default=pathlib.Path("04_OUR_BASELINE.md"))
    args = ap.parse_args()

    root = args.results
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return 1

    rp = root / "results.json"
    if not rp.exists():
        print(f"ERROR: {rp} not found")
        return 1
    try:
        results = json.loads(rp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: results.json is not valid JSON: {exc}")
        return 1
    if not isinstance(results, list):
        print("ERROR: results.json must be a JSON array of objects")
        return 1

    if not (root / "NOTES.md").exists():
        warn("NOTES.md missing — we need the null-test result and your recipe")
    else:
        txt = (root / "NOTES.md").read_text(encoding="utf-8").lower()
        if "null" not in txt:
            warn("NOTES.md does not mention the null test — please include its P_E")

    nt = root / "null_test.json"
    if not nt.exists():
        err("null_test.json missing - the null test is a gate, not optional")
    else:
        n = json.loads(nt.read_text(encoding="utf-8"))
        v, floor = n.get("p_e_mean"), n.get("null_floor")
        if v is None or floor is None:
            err("null_test.json has no p_e_mean/null_floor - rerun the null "
                "test with the current reference code")
        elif v < floor:
            err(f"NULL TEST FAILED: P_E {v:.4f} is below the floor {floor:.4f}. "
                f"The pipeline is leaking; no result here is usable.")
        else:
            notes.append(f"null test P_E {v:.4f} (chance "
                         f"{n.get('null_expected', float('nan')):.4f}, "
                         f"floor {floor:.4f}) - pass")

    baseline = load_baseline(args.baseline)
    if not baseline:
        warn(f"could not read baseline from {args.baseline}; skipping comparison")

    check_results(results, baseline)
    check_scores(root, results)

    print(f"checked {len(results)} entries in {rp}\n")
    for n in notes:
        print(f"  note     {n}")
    for w in warnings:
        print(f"  WARNING  {w}")
    for e in errors:
        print(f"  ERROR    {e}")
    if not errors and not warnings:
        print("  clean - nothing to flag")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    if errors:
        print("Fix the errors before sending. Explain any warnings in NOTES.md.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
