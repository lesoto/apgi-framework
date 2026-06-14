"""Paper 3 — Figure 3: Cross-Level Neuromodulatory Coupling Pathways (§3).

Single-panel circuit showing fast NE (ACC→LC) and slow cortisol (L3/L4→HPA)
pathways with timescale annotations.

Run:
    python figures/paper3/generate_fig3.py
    python figures/paper3/generate_fig3.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

LEVEL_COLORS = ["#a50f15", "#de2d26", "#fc8d59", "#fdcc8a", "#f0f0f0"]
LEVEL_NAMES = ["L4", "L3", "L2", "L1", "L0"]


def _box(ax, x, y, w, h, label, color, fontsize=8, edge="#333333", lw=1.5):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                   facecolor=color, edgecolor=edge, lw=lw, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, zorder=4, multialignment="center")


def plot(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # ── Level column (left) ───────────────────────────────────────────────
    LVL_X, LVL_W, LVL_H = 0.03, 0.14, 0.10
    level_centres = {}
    for i, (name, color) in enumerate(zip(LEVEL_NAMES, LEVEL_COLORS)):
        y = 0.82 - i * 0.16
        _box(ax, LVL_X, y, LVL_W, LVL_H, name, color, fontsize=9)
        level_centres[name] = (LVL_X + LVL_W / 2, y + LVL_H / 2)

    ax.text(LVL_X + LVL_W / 2, 0.96, "Hierarchy\nLevel", ha="center", fontsize=8,
            fontweight="bold", color="#333333")

    # ── Pathway A: fast NE (ACC → LC → L1) ───────────────────────────────
    NE_COLOR = "#2166ac"
    node_acc = (0.35, 0.80)
    node_lc = (0.55, 0.68)
    node_theta1 = (0.75, 0.48)

    for xy, label, color in [
        (node_acc, "ACC\n(L3 node)", "#cce5ff"),
        (node_lc, "Locus\nCoeruleus", "#d0e4ff"),
        (node_theta1, r"$\theta_1 \downarrow$ (L1)" + "\ngain ↑ (L0)", "#e8f4fd"),
    ]:
        circ = mpatches.Circle(xy, 0.07, facecolor=color, edgecolor=NE_COLOR,
                               lw=1.8, zorder=5)
        ax.add_patch(circ)
        ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=7,
                zorder=6, multialignment="center")

    def _arr(ax, src, dst, color, lw=1.8, label=None):
        ax.annotate("", xy=dst, xytext=src,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=7)
        if label:
            mx, my = (src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2
            ax.text(mx + 0.02, my + 0.02, label, fontsize=7, color=color, ha="center")

    _arr(ax, (node_acc[0] + 0.07, node_acc[1] - 0.04),
             (node_lc[0] - 0.07, node_lc[1] + 0.03),
             NE_COLOR, label="cortico-\ncoeruleus\nglutamate")
    _arr(ax, (node_lc[0] + 0.06, node_lc[1] - 0.04),
             (node_theta1[0] - 0.07, node_theta1[1] + 0.02),
             NE_COLOR, label="phasic NE\n(ms timescale)")

    ax.text(0.55, 0.88, "Pathway A — Fast NE", fontsize=9, fontweight="bold",
            color=NE_COLOR, ha="center")
    ax.text(0.55, 0.84, "(ACC→LC→L1; milliseconds)", fontsize=7.5,
            color=NE_COLOR, ha="center")

    # ── Pathway B: slow cortisol (L3/L4 → PVN → HPA → L1/L0) ───────────
    CRT_COLOR = "#d6604d"
    node_pvn = (0.35, 0.38)
    node_hpa = (0.55, 0.24)
    node_l1l0 = (0.75, 0.14)

    for xy, label, color in [
        (node_pvn, "PVN\n(hypothalamus)", "#fde0d9"),
        (node_hpa, "HPA axis\n(adrenal)", "#fdd49e"),
        (node_l1l0, "Cortisol →\nL1/L0 synaptic\nexcitability", "#fee8c8"),
    ]:
        circ = mpatches.Circle(xy, 0.07, facecolor=color, edgecolor=CRT_COLOR,
                               lw=1.8, zorder=5)
        ax.add_patch(circ)
        ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=7,
                zorder=6, multialignment="center")

    _arr(ax, (level_centres["L3"][0] + LVL_W / 2, level_centres["L3"][1]),
             (node_pvn[0] - 0.07, node_pvn[1] + 0.02),
             CRT_COLOR, label="L3/L4\ngoal state")
    _arr(ax, (node_pvn[0] + 0.07, node_pvn[1] - 0.03),
             (node_hpa[0] - 0.07, node_hpa[1] + 0.03),
             CRT_COLOR, label="CRH cascade")
    _arr(ax, (node_hpa[0] + 0.07, node_hpa[1] - 0.03),
             (node_l1l0[0] - 0.07, node_l1l0[1] + 0.03),
             CRT_COLOR, label="cortisol\n(min–hours)")

    ax.text(0.55, 0.50, "Pathway B — Slow Cortisol", fontsize=9,
            fontweight="bold", color=CRT_COLOR, ha="center")
    ax.text(0.55, 0.46, "(L3/L4→PVN→HPA→L1/L0; minutes–hours)", fontsize=7.5,
            color=CRT_COLOR, ha="center")

    # Caveat
    ax.text(0.50, 0.01,
            "Causal evidence for specific projection strengths is limited; "
            "pathway assignments follow anatomical and pharmacological literature (§3).",
            ha="center", va="bottom", fontsize=6.5, color="#888888", style="italic")

    ax.set_title(
        "Figure 3 — Cross-Level Neuromodulatory Coupling Pathways (Paper 3, §3)\n"
        "Pathway A: Fast NE  |  Pathway B: Slow Cortisol",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig3_neuromodulatory_coupling.pdf")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    plot(show=not args.no_show)


if __name__ == "__main__":
    main()
