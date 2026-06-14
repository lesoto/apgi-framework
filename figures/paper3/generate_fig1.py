"""Paper 3 — Figure 1: Five-Level Russian Doll Hierarchical Architecture (§2.3).

Vertically stacked concentric-ring diagram L0–L4 with timescales,
substrates, neuromodulators, coupling arrows, and substrate brackets.

Run:
    python figures/paper3/generate_fig1.py
    python figures/paper3/generate_fig1.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

LEVELS = [
    {
        "name": "L4",
        "tau": "months–years",
        "substrate": "Neuroendocrine / HPA",
        "neuromod": "cortisol",
        "color": "#a50f15",
        "r": 0.44,
        "apgi": r"$\Pi_4, \theta_4, \tau_{\mathrm{ign},4}$",
    },
    {
        "name": "L3",
        "tau": "minutes–hours",
        "substrate": "Prefrontal–limbic",
        "neuromod": "NE (phasic)",
        "color": "#de2d26",
        "r": 0.34,
        "apgi": r"$\Pi_3, \theta_3, \tau_{\mathrm{ign},3}$",
    },
    {
        "name": "L2",
        "tau": "seconds",
        "substrate": "Cortical reservoir",
        "neuromod": "ACh",
        "color": "#fc8d59",
        "r": 0.24,
        "apgi": r"$\Pi_2, \theta_2, \tau_{\mathrm{ign},2}$",
    },
    {
        "name": "L1",
        "tau": "200–500 ms",
        "substrate": "Cortical reservoir",
        "neuromod": "NE (fast)",
        "color": "#fdcc8a",
        "r": 0.14,
        "apgi": r"$\Pi_1, \theta_1, \tau_{\mathrm{ign},1}$",
    },
    {
        "name": "L0\n(sub-APGI)",
        "tau": "~10–50 ms",
        "substrate": "Reflexive (spinal/brainstem)",
        "neuromod": "—",
        "color": "#f0f0f0",
        "r": 0.06,
        "apgi": "no Π/θ/τ_ign",
    },
]


def plot(show: bool = True) -> None:
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
    ax_rings = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # ── Concentric rings ──────────────────────────────────────────────────
    ax_rings.set_xlim(-0.6, 1.1)
    ax_rings.set_ylim(-0.6, 0.7)
    ax_rings.set_aspect("equal")
    ax_rings.axis("off")

    cx, cy = 0.25, 0.0

    for lv in LEVELS:
        edge = "#888888" if "sub-APGI" in lv["name"] else lv["color"]
        ls = "--" if "sub-APGI" in lv["name"] else "-"
        circ = mpatches.Circle(
            (cx, cy), lv["r"],
            facecolor=lv["color"] if "sub-APGI" not in lv["name"] else "#f0f0f0",
            edgecolor=edge, lw=1.8, linestyle=ls, alpha=0.70, zorder=len(LEVELS) - LEVELS.index(lv),
        )
        ax_rings.add_patch(circ)
        label = lv["name"].split("\n")[0]
        ax_rings.text(
            cx, cy + lv["r"] - 0.03, label,
            ha="center", va="top", fontsize=8, fontweight="bold",
            color="#333333" if lv["color"] != "#fdcc8a" else "#555555",
        )

    # Centre dot
    ax_rings.plot(cx, cy, "o", color="#333333", ms=4, zorder=20)

    # Neuromodulatory carrier per boundary (outermost to innermost: L4/L3, L3/L2, L2/L1, L1/L0)
    CARRIERS = ["cortisol", "NE (phasic)", "NE (fast)", "NE (fast)"]

    # Coupling arrows
    for i in range(len(LEVELS) - 1):
        r_inner = LEVELS[i + 1]["r"]
        r_outer = LEVELS[i]["r"]
        mid_r = (r_inner + r_outer) / 2
        # α_down
        ax_rings.annotate(
            "", xy=(cx + mid_r + 0.02, cy), xytext=(cx + mid_r + 0.04, cy),
            arrowprops=dict(arrowstyle="<-", color="#2166ac", lw=1.3),
        )
        # α_up
        ax_rings.annotate(
            "", xy=(cx - mid_r - 0.02, cy), xytext=(cx - mid_r - 0.04, cy),
            arrowprops=dict(arrowstyle="<-", color="#d6604d", lw=1.3),
        )
        # Carrier label centred between the two ring radii (right side only to avoid clutter)
        ax_rings.text(
            cx + mid_r + 0.03, cy - 0.04,
            CARRIERS[i],
            ha="center", va="top", fontsize=5.5, color="#555555", style="italic",
        )

    # Legend for coupling
    ax_rings.plot([], [], color="#2166ac", lw=2, label=r"$\alpha_{\mathrm{down}}$ (top-down)")
    ax_rings.plot([], [], color="#d6604d", lw=2, label=r"$\alpha_{\mathrm{up}}$ (bottom-up)")
    ax_rings.legend(fontsize=7.5, loc="lower right", framealpha=0.8)

    # Substrate brackets
    brackets = [
        ("Reflexive (L0)", -0.08, -0.08, "#888888"),
        ("Cortical reservoir\n(L1–L2, ≤ ~2 s fading-memory)", 0.10, 0.28, "#fc8d59"),
        ("Neuroendocrine\n(L3–L4)", 0.30, 0.46, "#a50f15"),
    ]
    for label, r0, r1, color in brackets:
        ax_rings.annotate(
            "", xy=(cx + r1 + 0.05, 0.40), xytext=(cx + r0 + 0.05, 0.40),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.5),
        )
        ax_rings.text(cx + (r0 + r1) / 2 + 0.05, 0.44, label, ha="center",
                      fontsize=6.5, color=color, multialignment="center")

    # L0 legend note
    ax_rings.text(cx, -0.55,
                  "L0 is sub-APGI (no Π₀/θ₀/τ_ign_0);\nsupplies sensory drive to L1",
                  ha="center", fontsize=7, color="#888888", style="italic")

    ax_rings.set_title("Russian Doll concentric-ring\nhierarchy", fontsize=9, fontweight="bold")

    # ── Table panel ───────────────────────────────────────────────────────
    ax_table.axis("off")
    col_labels = ["Level", "τ_int", "Substrate", "Neuromod.", "APGI params"]
    table_data = [
        [lv["name"].replace("\n(sub-APGI)", "\n(sub)"), lv["tau"],
         lv["substrate"], lv["neuromod"], lv["apgi"]]
        for lv in LEVELS
    ]
    colors_rows = [["#fde0d9" if "sub" not in lv["name"] else "#f0f0f0"] * 5 for lv in LEVELS]
    # Override with level-specific colour
    for i, lv in enumerate(LEVELS):
        for j in range(5):
            colors_rows[i][j] = lv["color"] + "55"  # add transparency via hex

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.15, 1.7)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold")
        elif row <= len(LEVELS):
            cell.set_facecolor(LEVELS[row - 1]["color"] + "44")
    ax_table.set_title("Level parameters", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Figure 1 — Five-Level Russian Doll Hierarchical Architecture (§2.3)\n"
        "Paper 3",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig1_russian_doll_hierarchy.pdf")
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
