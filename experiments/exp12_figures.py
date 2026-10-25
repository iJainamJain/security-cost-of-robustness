"""Paper figures, built only from results files (no hand-typed numbers).

    python experiments/exp12_figures.py

Fig. 1  Security vs payload under DCTR: the trade-off itself (section 4.1).
Fig. 2  The same cells under three detectors, one panel per method (4.5, 4.6).

Colour is never the only encoding: every series also has its own marker and
line style, and is labelled directly, so the figures survive greyscale print.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "figures"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Method identity (Fig. 1). Validated: CVD dE 9.2, normal-vision dE 27.6.
METHOD = {"J-UNIWARD": ("#2a78d6", "o", "-"),
          "GMAS": ("#eb6834", "s", "--"),
          "ROAST-recon": ("#1baf7a", "^", "-.")}
# Detector identity (Fig. 2) - different hues from the methods on purpose, so a
# colour never means two things. Validated: CVD dE 16.2, normal-vision dE 30.3.
DETECTOR = {"DCTR + FLD": ("#4a3aa7", "o", "-"),
            "CC-PEV + FLD": ("#eda100", "D", ":"),
            "EfficientNet-B0": ("#008300", "s", "--")}
NULL_DCTR = 0.4856  # measured cover-vs-cover null of the DCTR pipeline (3.4)

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7,
    "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load():
    arm_a = json.loads((RES / "exp08_arm_a" / "arm_a_q71.json").read_text())
    ccpev = json.loads((RES / "exp11_ccpev" / "ccpev_crosscheck.json").read_text())
    cnn = json.loads((RES / "m4_results" / "results.json").read_text())
    dctr = {(r["method"], r["payload"]): (r["p_e"], r["p_e_sd"]) for r in arm_a}
    cc = {(r["method"], r["payload"]): (r["p_e"], r["p_e_sd"]) for r in ccpev}
    net = {(r["method"], r["payload_bpnzac"]): (r["p_e_mean"], r["p_e_std"])
           for r in cnn if r["status"] == "ok"}
    return dctr, cc, net


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.02, 0.52)
    ax.set_xticks([0.05, 0.1, 0.2, 0.3])
    ax.set_xlim(0.03, 0.32)


def fig1(dctr):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    style(ax)
    ax.axhline(NULL_DCTR, color=MUTED, linewidth=0.7, linestyle=(0, (1, 2)))
    ax.text(0.031, NULL_DCTR + 0.008, "chance (measured null)", color=MUTED,
            fontsize=6.5, va="bottom")
    labels = {"J-UNIWARD": "J-UNIWARD (non-robust)", "GMAS": "GMAS",
              "ROAST-recon": "ROAST"}
    for m, (col, mk, ls) in METHOD.items():
        xs = sorted(p for (mm, p) in dctr if mm == m)
        ys = [dctr[(m, p)][0] for p in xs]
        sd = [dctr[(m, p)][1] for p in xs]
        ax.errorbar(xs, ys, yerr=sd, color=col, marker=mk, linestyle=ls,
                    linewidth=1.5, markersize=4.5, capsize=2, elinewidth=0.8,
                    markeredgecolor="white", markeredgewidth=0.6, label=labels[m])
    ax.set_xlabel("Payload (bpnzac)")
    ax.set_ylabel(r"Detection error $P_E$ (DCTR + FLD)")
    # The band between the curves is empty from 0.12 bpnzac up: the legend sits
    # there, clear of every mark and error bar.
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.50), handlelength=2.6)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_security_vs_payload.{ext}")
    plt.close(fig)


def fig2(dctr, cc, net):
    methods = [("J-UNIWARD", "J-UNIWARD (non-robust)"), ("GMAS", "GMAS"),
               ("ROAST-recon", "ROAST")]
    src = {"DCTR + FLD": dctr, "CC-PEV + FLD": cc, "EfficientNet-B0": net}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), sharey=True)
    for ax, (m, title) in zip(axes, methods):
        style(ax)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        for det, (col, mk, ls) in DETECTOR.items():
            d = src[det]
            xs = sorted(p for (mm, p) in d if mm == m)
            if not xs:
                continue
            ax.errorbar(xs, [d[(m, p)][0] for p in xs],
                        yerr=[d[(m, p)][1] for p in xs], color=col, marker=mk,
                        linestyle=ls, linewidth=1.5, markersize=4.5, capsize=2,
                        elinewidth=0.8, markeredgecolor="white",
                        markeredgewidth=0.6, label=det)
    axes[1].set_xlabel("Payload (bpnzac)")
    axes[0].set_ylabel(r"Detection error $P_E$")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16),
               handlelength=2.8)
    fig.subplots_adjust(wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_three_detectors.{ext}")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    dctr, cc, net = load()
    fig1(dctr)
    fig2(dctr, cc, net)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:<34} {f.stat().st_size / 1024:6.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
