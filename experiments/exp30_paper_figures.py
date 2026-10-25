"""Figures for the full-scale results, built from exp29's paper_tables.json.

    python experiments/exp29_paper_tables.py     # refresh the tables first
    python experiments/exp30_paper_figures.py

Fig. 1  Security against payload at 10,000 images under DCTR, one panel per
        channel the methods target.
Fig. 2  The 5,000-image cells under three detectors, one panel per method.

exp12_figures.py still regenerates the 300-image figures of the earlier draft.
Colour is never the only encoding: every series has its own marker and line
style, so the figures survive greyscale print.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "exp29_paper_tables" / "paper_tables.json"
OUT = ROOT / "figures"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Method identity: colour, marker, line style, hollow marker. The first three
# are those of exp12_figures.py. The two SCRIPT settings share a colour and
# differ in line style and marker fill - one method, two strengths.
METHOD = {"J-UNIWARD": ("#2a78d6", "o", "-", False),
          "GMAS": ("#eb6834", "s", "--", False),
          "ROAST": ("#1baf7a", "^", "-.", False),
          "MINICER": ("#b0429a", "v", "--", False),
          "SCRIPT, R = 0": ("#222222", "D", "-", False),
          "SCRIPT, R = 10": ("#222222", "D", ":", True)}
LABEL = {"J-UNIWARD": "J-UNIWARD (non-robust)", "ROAST": "ROAST (reconstruction)", "SCRIPT, R = 0": "SCRIPT, strength 0",
         "SCRIPT, R = 10": "SCRIPT, strength 10"}
# Detector identity: the validated triple of exp12_figures.py (CVD dE 16.2).
DETECTOR = {"DCTR + FLD, tuned": ("dctr", "#4a3aa7", "o", "-"),
            "EfficientNet-B0": ("effnet", "#008300", "s", "--"),
            "SRNet, curriculum": ("srnet", "#eda100", "^", "-.")}

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7,
    "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def base(ax, top=0.52):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.04 * top, top)


def series(ax, xs, ys, sd, name, col, mk, ls, hollow=False):
    return ax.errorbar(xs, ys, yerr=sd, color=col, marker=mk, linestyle=ls, linewidth=1.5,
                       markersize=4.5, capsize=2, elinewidth=0.8,
                       markerfacecolor="white" if hollow else col,
                       markeredgecolor=col if hollow else "white",
                       markeredgewidth=1.0 if hollow else 0.6, label=LABEL.get(name, name))


def chance(ax, y, x, text="chance (measured null)"):
    ax.axhline(y, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)))
    ax.text(x, y + 0.006, text, color=MUTED, fontsize=6.5, va="bottom")


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)


# Figure 1, redrawn 2026-10-13 after the review council: SCRIPT is no longer set
# beside J-UNIWARD (SCRIPT is side-informed, J-UNIWARD is not) and strength 10 is
# gone. Panel (b) sets each cost version of SCRIPT against its own embedder
# without the selection; panel (c) is MINICER against its step.
# Per-panel palettes checked with the dataviz validator (CVD and normal vision):
# (a) blue / orange / green, (b) violet / green, (c) one hue.
RES = ROOT / "results"
RELEASED, REPAIRED, MINICER_COL = "#4a3aa7", "#008300", "#b0429a"
PAIRED_CHANCE = {5000: 0.496, 2500: 0.494, 1000: 0.492}   # exp44, by test pairs


def _cells(path, prefix):
    out = {}
    if path.exists():
        for r in json.loads(path.read_text(encoding="utf-8")):
            if r["cell"].startswith(prefix):
                out[float(r["cell"].rsplit("_", 1)[1])] = (r["p_e_mean"], r["p_e_std"])
    return out


def _line(ax, d, col, mk, ls, hollow, label, size=4.5):
    xs = sorted(d)
    return ax.errorbar(xs, [d[x][0] for x in xs], yerr=[d[x][1] for x in xs], color=col,
                       marker=mk, linestyle=ls, linewidth=1.5, markersize=size, capsize=2,
                       elinewidth=0.8, markerfacecolor="white" if hollow else col,
                       markeredgecolor=col if hollow else "white",
                       markeredgewidth=1.0 if hollow else 0.6, label=label)


def fig1(t):
    sec = {}
    for r in t["security_10000"]:
        sec.setdefault(r["method"], []).append(r)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.0, 1.35, 0.85], wspace=0.12))
    # (a) upward recompression, 10,000 images
    ax = axes[0]
    base(ax)
    ax.set_xticks([0.05, 0.1, 0.2, 0.3]); ax.set_xlim(0.03, 0.32)
    ax.set_title("(a) Upward, QF 65 → 71", fontsize=8, color=INK, loc="left")
    chance(ax, PAIRED_CHANCE[5000], 0.17, "chance (paired test)")
    for m in ("J-UNIWARD", "GMAS", "ROAST"):
        rows = sorted(sec[m], key=lambda r: r["payload"])
        col, mk, ls, hollow = METHOD[m]
        series(ax, [r["payload"] for r in rows], [r["p_e"] for r in rows],
               [r["p_e_sd"] for r in rows], m, col, mk, ls, hollow)
    ax.set_xlabel("Payload (bpnzac)")
    ax.set_ylabel(r"Detection error $P_E$ (DCTR + FLD)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.6)

    # (b) same quality factor: SCRIPT and its embedder without the selection, 5,000 images
    ax = axes[1]
    base(ax)
    ax.set_xticks([0.05, 0.1, 0.2, 0.3, 0.4, 0.6]); ax.set_xlim(0.02, 0.63)
    labels = ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3", "0.4", "0.6"])
    labels[1].set_ha("left")
    ax.set_title("(b) Same QF: SCRIPT at strength 0", fontsize=8, color=INK, loc="left")
    chance(ax, PAIRED_CHANCE[2500], 0.33, "chance (paired test)")
    rel = _cells(RES / "exp21_script_low_fullscale/tier_5000/results.json", "SCRIPT-R0_")
    rel_ref = _cells(RES / "exp36_si_reference/tier_5000/results.json", "SI-UERD_")
    rep = {**_cells(RES / "exp40_script_fix/tier_5000/results.json", "SCRIPT-fix3_"),
           **_cells(RES / "exp40_script_fix/tier_5000_high/results.json", "SCRIPT-fix3_")}
    rep_ref = {}
    for sub in ("tier1_5000", "tier2_5000", "tier3_5000"):
        rep_ref.update(_cells(RES / "exp43_si_reference_fix" / sub / "results.json", "SI-fix3_"))
    # The two members of a pair nearly coincide: the reference is drawn first,
    # larger and hollow, and the scheme on top, smaller and filled.
    h = [_line(ax, rel_ref, RELEASED, "D", ":", True, "released cost, no selection", 6.5),
         _line(ax, rel, RELEASED, "D", "-", False, "released cost, with selection (SCRIPT)", 3.6),
         _line(ax, rep_ref, REPAIRED, "o", ":", True, "repaired cost, no selection", 6.5),
         _line(ax, rep, REPAIRED, "o", "-", False, "repaired cost, with selection", 3.6)]
    ax.set_xlabel("Payload (bpnzac)")
    ax.legend([h[1], h[0], h[3], h[2]], [x.get_label() for x in (h[1], h[0], h[3], h[2])],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.6)

    # (c) MINICER against its step, 2,000 images
    ax = axes[2]
    base(ax)
    sweep = {}
    f50 = RES / "exp50_minicer_step_sweep/sweep_2000/results.json"
    if f50.exists():
        for r in json.loads(f50.read_text(encoding="utf-8")):
            sweep.setdefault(r["payload_bpnzac"], {})[r["step"]] = (r["p_e_mean"], r["p_e_std"])
    ax.set_xticks([1, 2, 3, 4, 6]); ax.set_xlim(0.5, 6.5)
    ax.set_title("(c) Same QF: MINICER", fontsize=8, color=INK, loc="left")
    chance(ax, PAIRED_CHANCE[1000], 3.0, "chance")
    for p, ls, hollow in ((0.05, "-", False), (0.1, "--", True)):
        if p in sweep:
            _line(ax, sweep[p], MINICER_COL, "v", ls, hollow, f"{p:g} bpnzac")
    ax.set_xlabel("Step (1 = released)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.6)
    save(fig, "fig1_security_10000")
    (OUT / "fig1_data.json").write_text(json.dumps(dict(
        panel_a={m: [(r["payload"], r["p_e"], r["p_e_sd"]) for r in sec[m]]
                 for m in ("J-UNIWARD", "GMAS", "ROAST")},
        panel_b=dict(released=rel, released_no_selection=rel_ref, repaired=rep,
                     repaired_no_selection=rep_ref),
        panel_c=sweep, paired_chance=PAIRED_CHANCE), indent=1, default=str), encoding="utf-8")


def fig2(t):
    det = {}
    for r in t["detectors_5000"]:
        det.setdefault(r["method"], []).append(r)
    # Two groups with their own vertical scale: on one shared axis the three
    # detectors could not be told apart below 0.05.
    groups = [(["J-UNIWARD", "SCRIPT, R = 0"], 0.52), (["GMAS", "ROAST", "MINICER"], 0.17)]
    fig = plt.figure(figsize=(7.1, 2.5))
    gs = fig.add_gridspec(1, 6, width_ratios=[1.15, 1, 0.16, 0.72, 0.72, 0.72], wspace=0.10)
    cols, handles = [0, 1, 3, 4, 5], {}
    k = 0
    for methods, top in groups:
        for j, m in enumerate(methods):
            ax = fig.add_subplot(gs[0, cols[k]])
            k += 1
            base(ax, top)
            rows = sorted(det[m], key=lambda r: r["payload"])
            xs = [r["payload"] for r in rows]
            ax.set_xticks(xs)
            labels = ax.set_xticklabels([f"{x:g}" for x in xs])
            if len(xs) > 3:                  # 0.05 and 0.1 sit close together
                labels[0].set_ha("right")
                labels[1].set_ha("left")
            pad = 0.35 * (xs[-1] - xs[0])
            ax.set_xlim(xs[0] - pad * 0.5, xs[-1] + pad * 0.5)
            # the three robust panels are narrow: "ROAST (reconstruction)" ran into "MINICER"
            ax.set_title(LABEL.get(m, m).replace(" (non-robust)", "").replace(" (", "\n("),
                         fontsize=8 if top > 0.3 else 7, color=INK, loc="left")
            if top > 0.3:
                ax.axhline(0.491, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)))
                if m != "J-UNIWARD":
                    ax.text(0.15, 0.478, "chance", color=MUTED, fontsize=6.5, va="top",
                            ha="center")
            for name, (key, col, mk, ls) in DETECTOR.items():
                ks = "dctr_tuned" if key == "dctr" else key
                # A cell whose tuned run is not in yet falls back to the fixed
                # setting; exp29's table says which.
                pts = [(r["payload"], r.get(ks, r["dctr_fixed"] if key == "dctr" else None),
                        r.get(ks + "_sd", r["dctr_fixed_sd"] if key == "dctr" else None))
                       for r in rows]
                pts = [p for p in pts if p[1] is not None]
                if pts:
                    handles[name] = ax.errorbar(
                        [p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                        color=col, marker=mk, linestyle=ls, linewidth=1.5, markersize=4.5,
                        capsize=2, elinewidth=0.8, markeredgecolor="white",
                        markeredgewidth=0.6, label=name)
            if j:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(r"Detection error $P_E$" if k == 1 else "")
    fig.text(0.5, -0.04, "Payload (bpnzac)", ha="center", color=INK)
    fig.legend(list(handles.values()), list(handles), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.2), handlelength=2.8)
    save(fig, "fig2_detectors_5000")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    t = json.loads(TABLES.read_text(encoding="utf-8"))
    fig1(t)
    fig2(t)
    for f in sorted(OUT.glob("fig*")):
        print(f"{f.name:<36} {f.stat().st_size / 1024:6.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
