"""Figures 3 and 5-8 of the manuscript (Figures 2 and 4 come from
experiments/exp30_paper_figures.py, Figure 1 is architecture_pipeline.svg).

Revised 2026-10-20 after a check of the 13 October version
(generate_paper_figures_before_fixes_2026-10-20.py):

- Figure 3 is split by channel. J-UNIWARD is plotted where it was measured (its
  untouched delivery on the same-quality-factor channel, about 22%), not at
  100%; the "ideal target" boxes are gone; the repaired SCRIPT and the
  no-selection reference are added at 10,000 images.
- Figure 5 counts payload-independent changes (not changes "at 0 payload"),
  adds the principled rule on standard covers, and moves the legends off the
  bars.
- Figure 6 drops an annotation that pointed at the wrong curve.
- Figure 7 marks the SRNet runs on SCRIPT at 256x256 that did not train.
- Figure 8 no longer presents the released SCRIPT's batch result as a general
  rule.

Every number is copied from a table of the manuscript; the table is named next
to it. Palettes per panel were checked with the dataviz validator (CVD and
normal vision).
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})

# Same identities as exp30_paper_figures.py (Figures 2 and 4).
C = {"J-UNIWARD": "#2a78d6", "GMAS": "#eb6834", "ROAST": "#1baf7a", "MINICER": "#b0429a",
     "released": "#4a3aa7", "repaired": "#008300"}


def _ax(ax, ylim=(-0.02, 0.53)):
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(*ylim)


def _chance(ax, y, x, label="chance (paired test)"):
    ax.axhline(y, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)), zorder=1)
    ax.text(x, y + 0.006, label, color=MUTED, fontsize=6.5, va="bottom")


def _series(ax, x, y, col, mk, ls, label, hollow=False, labels=None, dx=1.2, dy=0.0):
    ax.plot(x, y, color=col, marker=mk, linestyle=ls, linewidth=1.4, markersize=5,
            markerfacecolor="white" if hollow else col, markeredgecolor=col,
            markeredgewidth=1.1, label=label, zorder=4)
    for xi, yi, t in zip(x, y, labels or []):
        ax.text(xi + dx, yi + dy, t, fontsize=5.8, color=MUTED, va="center", zorder=5)


def _save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", stem)


# =====================================================================
# Figure 3: delivery against detection error, one panel per channel
# =====================================================================
def generate_fig4():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True,
                             gridspec_kw=dict(width_ratios=[0.85, 1.15], wspace=0.08))
    # (a) upward channel, 10,000 images: Tables 1 and 2
    ax = axes[0]
    _ax(ax)
    _chance(ax, 0.496, 62)
    p4 = ["0.05", "0.1", "0.2", "0.3"]
    _series(ax, [67.4, 72.1, 72.8, 71.3], [0.1538, 0.0180, 0.0004, 0.0001], C["GMAS"], "s", "--",
            "GMAS", labels=p4[:2], dx=-4.2)
    _series(ax, [98.8, 99.0, 98.7, 97.6], [0.0749, 0.0111, 0.0006, 0.0004], C["ROAST"], "^", "-.",
            "ROAST (reconstruction)", labels=p4[:2], dx=-4.4)
    ax.set_xlim(60, 102)
    ax.set_xlabel("Images delivered without error (%)")
    ax.set_ylabel(r"Detection error $P_{\mathrm{E}}$ (DCTR + FLD)")
    ax.set_title("(a) Upward, QF 65 → 71", fontsize=8, color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)

    # (b) same quality factor: Tables 1, 2, 8 and 10, and the 10,000-image repair run
    ax = axes[1]
    _ax(ax)
    _chance(ax, 0.496, 2)
    p3 = ["0.05", "0.1", "0.2"]
    _series(ax, [22.5, 22.3, 21.4], [0.4697, 0.4222, 0.3147], C["J-UNIWARD"], "o", "-",
            "J-UNIWARD, not robust (delivery: Table 10)", labels=p3, dx=1.5)
    _series(ax, [26.6, 26.6, 26.3], [0.389, 0.377, 0.365], C["released"], "D", ":",
            "SCRIPT's embedder, no selection (Table 8)", hollow=True, labels=None)
    _series(ax, [56.8, 40.0, 19.1], [0.0984, 0.0106, 0.0005], C["MINICER"], "v", "--",
            "MINICER, step 6", labels=p3, dx=1.5)
    _series(ax, [99.95, 99.92, 99.79], [0.3940, 0.3817, 0.3734], C["released"], "D", "-",
            "SCRIPT, released cost", labels=None)
    _series(ax, [99.98, 99.92, 99.84], [0.4913, 0.4822, 0.4541], C["repaired"], "o", "-",
            "SCRIPT, repaired cost", labels=None)
    ax.set_xlim(0, 104)
    ax.set_xlabel("Images delivered without error (%)")
    ax.set_title("(b) Same quality factor, QF 65 → 65", fontsize=8, color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=1.2)
    _save(fig, "fig4_tradeoff_delivery_vs_security")


# =====================================================================
# Figure 5: SCRIPT's payload-independent changes and P_E (Table 7)
# =====================================================================
def generate_fig5():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75),
                             gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.3))
    cats = ["Standard\nreleased", "Standard\nno change\nat zero cost", "Standard\nfloor 0.1",
            "Exact DCT\nreleased", "Exact DCT\nno change\nat ties"]
    x = np.arange(len(cats))
    dis = [147, 0, 0, 0, 0]
    ties = [38, 0, 0, 39, 0]
    ax = axes[0]
    _ax(ax, (0, 225))
    ax.bar(x, dis, width=0.55, color=C["GMAS"], label="where the transforms disagree", zorder=3)
    ax.bar(x, ties, width=0.55, bottom=dis, color=C["MINICER"], label="at exact rounding ties",
           zorder=3)
    for i in range(len(cats)):
        ax.text(i, dis[i] + ties[i] + 4, f"{dis[i] + ties[i]}", ha="center", va="bottom",
                fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=6.2)
    ax.set_ylabel("Changes per image that do\nnot depend on the payload")
    ax.set_title("(a) Free changes", fontsize=8, color=INK, loc="left")
    ax.legend(loc="upper right", fontsize=6.5)

    ax = axes[1]
    _ax(ax, (0.30, 0.52))
    pe05 = [0.397, 0.459, 0.490, 0.455, 0.492]
    pe20 = [0.378, 0.443, 0.459, 0.453, 0.480]
    ax.bar(x - 0.15, pe05, width=0.3, color=C["released"], label="0.05 bpnzac", zorder=3)
    ax.bar(x + 0.15, pe20, width=0.3, color=C["J-UNIWARD"], label="0.2 bpnzac", zorder=3)
    _chance(ax, 0.492, -0.4, "chance (paired test, 0.492)")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=6.2)
    ax.set_ylabel(r"Detection error $P_{\mathrm{E}}$ (DCTR + FLD)")
    ax.set_title("(b) Detection error", fontsize=8, color=INK, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2)
    _save(fig, "fig5_script_compressor_mismatch")


# =====================================================================
# Figure 6: detection error against the size of the experiment (Table 11)
# =====================================================================
def generate_fig6():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True, gridspec_kw=dict(wspace=0.12))
    cells = [("J-UNIWARD 0.1", C["J-UNIWARD"], "o", "-"), ("J-UNIWARD 0.3", C["J-UNIWARD"], "o", "--"),
             ("GMAS 0.05", C["GMAS"], "s", "-"), ("GMAS 0.1", C["GMAS"], "s", "--"),
             ("ROAST 0.05", C["ROAST"], "^", "-"), ("MINICER 0.05", C["MINICER"], "v", "-")]
    dctr = [[0.454, 0.426, 0.422], [0.289, 0.213, 0.208], [0.243, 0.151, 0.154],
            [0.065, 0.021, 0.018], [0.125, 0.071, 0.075], [0.186, 0.109, 0.098]]
    eff = [[0.487, 0.495, 0.497], [0.479, 0.424, 0.224], [0.450, 0.149, 0.134],
           [0.255, 0.041, 0.025], [0.207, 0.040, 0.024], [0.462, 0.083, 0.053]]
    for ax, sizes, vals, title in ((axes[0], [300, 2000, 10000], dctr, "(a) DCTR + FLD"),
                                   (axes[1], [300, 2000, 5000], eff, "(b) EfficientNet-B0")):
        _ax(ax)
        for (name, col, mk, ls), v in zip(cells, vals):
            ax.plot(sizes, v, color=col, marker=mk, linestyle=ls, linewidth=1.3, markersize=4,
                    label=name)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Images in the experiment")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
    axes[0].set_ylabel(r"Detection error $P_{\mathrm{E}}$")
    axes[1].text(5000, 0.505, "J-UNIWARD 0.1: never trains", fontsize=6.2, color=MUTED, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1))
    _save(fig, "fig6_sample_size_scaling")


# =====================================================================
# Figure 7: image size and image source at 0.05 bpnzac (Table 14)
# =====================================================================
def generate_fig7():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True, gridspec_kw=dict(wspace=0.08))
    schemes = ["J-UNIWARD", "GMAS", "ROAST", "MINICER", "SCRIPT\nstrength 0"]
    x = np.arange(len(schemes))
    w = 0.26
    cols = ("#2a78d6", "#4a3aa7", "#1baf7a")
    names = (r"BOSSbase 256$\times$256", r"BOWS2 256$\times$256", r"BOSSbase 512$\times$512")
    data = {"(a) DCTR + FLD": ([0.488, 0.262, 0.136, 0.214, 0.444], [0.490, 0.304, 0.205, 0.242, 0.452],
                               [0.474, 0.157, 0.080, 0.109, 0.390]),
            "(b) SRNet": ([0.451, 0.196, 0.069, 0.112, 0.472], [0.457, 0.220, 0.121, 0.124, 0.494],
                          [0.418, 0.124, 0.037, 0.042, 0.418])}
    for ax, (title, sets) in zip(axes, data.items()):
        _ax(ax)
        for k, (vals, col, nm) in enumerate(zip(sets, cols, names)):
            bars = ax.bar(x + (k - 1) * w, vals, width=w, color=col, label=nm, zorder=3)
            if title.startswith("(b)") and k < 2:
                # SRNet on SCRIPT at 256x256: four of six runs never trained (Section 4.7)
                bars[-1].set_facecolor("white")
                bars[-1].set_edgecolor(col)
                bars[-1].set_hatch("////")
        _chance(ax, 0.494, -0.45, "chance")
        ax.set_xticks(x)
        ax.set_xticklabels(schemes, fontsize=6.6)
        ax.set_title(title + ", 0.05 bpnzac", fontsize=8, color=INK, loc="left")
    axes[0].set_ylabel(r"Detection error $P_{\mathrm{E}}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    _save(fig, "fig7_resolution_dataset_comparison")


# =====================================================================
# Figure 8: batch steganography (Tables A1 and A2)
# =====================================================================
def generate_fig8():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), sharey=True, gridspec_kw=dict(wspace=0.08))
    ax = axes[0]
    _ax(ax)
    k = [1, 2, 4, 10, 20]
    ax.plot(k, [0.037, 0.097, 0.173, 0.287, 0.398], "o-", color=C["released"], lw=1.3, ms=4.5,
            label="DCTR + FLD")
    ax.plot(k, [np.nan, 0.073, 0.103, 0.147, 0.482], "s--", color=C["repaired"], lw=1.3, ms=4.5,
            label="EfficientNet, from scratch")
    ax.plot(k, [np.nan, np.nan, 0.102, 0.110, 0.105], "^-.", color=C["GMAS"], lw=1.3, ms=4.5,
            label="EfficientNet, curriculum")
    ax.set_xticks(k)
    ax.set_xlabel("Images sharing the payload")
    ax.set_ylabel(r"Batch detection error $P_{\mathrm{E}}$")
    ax.set_title("(a) SCRIPT at strength 10", fontsize=8, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=6.4)

    ax = axes[1]
    _ax(ax)
    kk = [1, 2, 4]
    ax.plot(kk, [0.465, 0.438, 0.390], "D-", color=C["released"], lw=1.3, ms=4.5,
            label="released SCRIPT, DCTR (tuned)")
    ax.plot(kk, [0.474, 0.446, 0.414], "D--", color=C["released"], lw=1.3, ms=4.5,
            markerfacecolor="white", label="released SCRIPT, SRNet")
    ax.plot(kk, [0.442, 0.459, 0.470], "o-", color=C["J-UNIWARD"], lw=1.3, ms=4.5,
            label="J-UNIWARD, DCTR (tuned)")
    ax.plot(kk, [0.414, 0.435, 0.439], "o--", color=C["J-UNIWARD"], lw=1.3, ms=4.5,
            markerfacecolor="white", label="J-UNIWARD, SRNet")
    ax.set_xticks(kk)
    ax.set_xticklabels(["1 × 0.2", "2 × 0.1", "4 × 0.05"])
    ax.set_xlabel("Images × payload each (bpnzac)")
    ax.set_title("(b) Strength 0, released cost", fontsize=8, color=INK, loc="left")
    ax.legend(loc="lower left", fontsize=6.4)
    _save(fig, "fig8_batch_pooling_strategies")


# =====================================================================
# Figures 5-7 (added 2026-10-21): data no table shows, from
# experiments/exp55_figure_data.py
# =====================================================================
EXP55 = HERE.parent / "results" / "exp55_figure_data" / "results.json"
BLUE = ["#f4f3f0", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def generate_modes():
    import json
    from matplotlib.colors import LinearSegmentedColormap
    m = json.loads(EXP55.read_text())["modes"]
    a, b = np.array(m["kind_A_per_image"]), np.array(m["kind_B_per_image"])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE)
    vmax = max(a.max(), b.max())
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw=dict(wspace=0.18))
    for ax, v, title in ((axes[0], a, "(a) Transforms disagree, |e| > 0.5\n"
                                      f"{m['kind_A_total']:.0f} per image, about 147 changed"),
                         (axes[1], b, "(b) Exact rounding ties, |e| = 0.5\n"
                                      f"{m['kind_B_total']:.0f} per image, about 38 changed")):
        im = ax.imshow(v, cmap=cmap, vmin=0, vmax=vmax)
        for i in range(8):
            for j in range(8):
                if v[i, j] >= 0.5:
                    ax.text(j, i, f"{v[i, j]:.0f}", ha="center", va="center", fontsize=6.2,
                            color="white" if v[i, j] > 0.45 * vmax else INK)
        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("Horizontal frequency")
        ax.set_title(title, fontsize=7.4, color=INK, loc="left")
    axes[0].set_ylabel("Vertical frequency")
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("Zero-cost coefficients per image", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=0)
    _save(fig, "fig5_zero_cost_modes")


def generate_roc():
    import json
    r = json.loads(EXP55.read_text())["roc"]
    fpr = np.array(r["fpr"])
    pay = [("0.05", "#86b6ef", "-"), ("0.1", "#2a78d6", "--"), ("0.2", "#104281", "-.")]
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharey=True, gridspec_kw=dict(wspace=0.1))
    for ax, name, title in ((axes[0], "released", "(a) Released cost"),
                            (axes[1], "repaired", "(b) Repaired cost (floor 0.1)")):
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)
        ax.text(0.62, 0.55, "no detection", rotation=38, fontsize=6.3, color=MUTED)
        for p, col, ls in pay:
            d = r[f"{name}_{p}"]
            ax.plot(fpr, d["tpr"], color=col, ls=ls, lw=1.5, zorder=3,
                    label=f"{p} bpnzac  ($P_{{\\mathrm{{E}}}}$ {d['p_e']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.6, zorder=0)
        ax.set_xlabel("False-alarm rate (covers flagged)")
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.legend(loc="lower right", fontsize=6.4, handlelength=2.6)
    axes[0].set_ylabel("Detection rate (stego images caught)")
    _save(fig, "fig6_roc_script")


def generate_nulls():
    import json
    nl = json.loads(EXP55.read_text())["nulls"]
    titles = ["(a) J-UNIWARD\nDCTR, 10,000", "(b) MINICER, step 1\nDCTR, 2,000",
              "(c) SCRIPT repaired\nDCTR, 10,000", "(d) SCRIPT repaired\nSRNet, 5,000",
              "(e) No-selection ref.,\nrepaired, DCTR, 5,000"]
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 2.1), gridspec_kw=dict(wspace=0.18))
    for ax, (cell, runs), title in zip(axes, nl.items(), titles):
        null = np.concatenate([x["null"] for x in runs])
        obs = [x["p_e"] for x in runs]
        lo, hi = min(min(obs), null.min()) - 0.002, max(max(obs), null.max()) + 0.001
        step = 1 / (2 * runs[0]["pairs"])            # P_E moves in steps of 1/(2n)
        w = step * np.ceil(0.0004 / step - 1e-9)
        ax.hist(null, bins=np.arange(lo - step / 2, hi, w), color="#9ec5f4", zorder=2)
        for o in obs:
            ax.axvline(o, color=INK, lw=0.9, ymax=0.82, zorder=3)
        k = sum(x["p_value"] < 0.01 for x in runs)
        ax.text(0.02, 0.97, f"p < 0.01 in\n{k} of {len(runs)} runs", transform=ax.transAxes,
                fontsize=6, color=MUTED, va="top")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.tick_params(axis="x", labelsize=6)
        ax.set_title(title, fontsize=7, color=INK, loc="left")
    axes[2].set_xlabel(r"Detection error $P_{\mathrm{E}}$ on the paired test set")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend([Patch(color="#9ec5f4"), Line2D([], [], color=INK, lw=0.9)],
               ["$P_{\\mathrm{E}}$ with labels exchanged at random within pairs (no signal)",
                "observed $P_{\\mathrm{E}}$, one line per split or seed"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    _save(fig, "fig7_paired_null")


# =====================================================================
# Figure 1 (redrawn 2026-10-21): the evaluation pipeline, protocol only
# =====================================================================
def generate_pipeline():
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(7.1, 3.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53)
    ax.axis("off")
    LINE = "#8a8984"

    def box(x0, y0, x1, y1, text, edge=LINE, fill="#f7f6f3", size=6.8):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                    linewidth=0.9, edgecolor=edge, facecolor=fill, zorder=2))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=size,
                color=INK, zorder=3, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), zorder=1,
                    arrowprops=dict(arrowstyle="-|>", color=LINE, lw=0.9, shrinkA=0, shrinkB=0,
                                    mutation_scale=8))

    def note(x, y, text, va):
        ax.text(x, y, text, ha="center", va=va, fontsize=6.2, color=MUTED, style="italic")

    def head(x, y, text):
        ax.text(x, y, text, ha="left", va="bottom", fontsize=6.8, color=MUTED, weight="bold")

    # sender
    head(0.5, 48.5, "SENDER")
    box(0.5, 16, 18, 34, "Covers\nBOSSbase 1.01\n10,000 images\n512×512\nJPEG QF 65")
    box(21.5, 41, 46.5, 47.5, "Message, 0.05–0.3 bpnzac")
    ax.add_patch(FancyBboxPatch((21.5, 3), 25, 34.5, boxstyle="round,pad=0,rounding_size=1.2", lw=0.9,
                                edgecolor=LINE, facecolor="white", zorder=1))
    ax.text(34, 34.6, "Embedding", ha="center", va="center", fontsize=7.2, color=INK, weight="bold")
    box(23.5, 23.5, 44.5, 31.5, "Arm A: GMAS, ROAST\nwith RS(31,15)", edge=C["GMAS"], fill="#fdf0ea")
    box(23.5, 14, 44.5, 22, "Arm B: MINICER, SCRIPT", edge=C["MINICER"], fill="#f7ecf5")
    box(23.5, 4.5, 44.5, 12.5, "Reference: J-UNIWARD\n(not robust)", edge=C["J-UNIWARD"], fill="#eaf2fc")
    arrow(18, 25, 21.5, 25)
    arrow(34, 41, 34, 37.5)
    box(50, 18, 61, 32, "Stego image\nas uploaded")
    arrow(46.5, 25, 50, 25)

    # security branch
    red, green = "#b23a48", "#008300"
    head(67, 48.5, "SECURITY: measured before the channel")
    box(67, 31, 92, 46, "Warden\nDCTR + FLD, 10,000 images\nEfficientNet-B0, SRNet,\n5,000 images",
        edge=red, fill="#fbeef0")
    box(97, 31, 119.5, 46, "Detection error $P_{\\mathrm{E}}$\nagainst the chance level\nof a paired test",
        edge=red, fill="#fbeef0")
    arrow(61, 29, 67, 37)
    note(61.5, 36.5, "intercepted", "bottom")
    arrow(92, 38.5, 97, 38.5)

    # robustness branch
    head(67, 0.2, "ROBUSTNESS: measured after the channel")
    box(67, 5, 85, 21, "Channel\nArm A: QF 65 → 71\nArm B: QF 65 → 65", edge=green, fill="#ebf6ea")
    box(89, 5, 101, 21, "Receiver\nextracts the\nmessage", edge=green, fill="#ebf6ea")
    box(105, 5, 119.5, 21, "Error-free\ndelivery and\nbit error rate", edge=green, fill="#ebf6ea")
    arrow(61, 21, 67, 13)
    note(61.5, 13.5, "recompressed", "top")
    arrow(85, 13, 89, 13)
    arrow(101, 13, 105, 13)
    _save(fig, "fig1_pipeline")


if __name__ == "__main__":
    generate_pipeline()
    generate_modes()
    generate_roc()
    generate_nulls()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    generate_fig8()
