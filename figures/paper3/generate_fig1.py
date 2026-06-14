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
    from matplotlib.patches import FancyArrowPatch

    fig = plt.figure(figsize=(15, 7.5))
    # Ring diagram given ≥50% of the figure width; the table is compressed and
    # defers full detail to Table 1 of the manuscript.
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.18)
    ax_rings = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    # ── Concentric rings ──────────────────────────────────────────────────
    ax_rings.set_xlim(-0.85, 1.25)
    ax_rings.set_ylim(-0.75, 0.85)
    ax_rings.set_aspect("equal")
    ax_rings.axis("off")

    cx, cy = 0.18, -0.02

    for lv in LEVELS:
        is_l0 = "sub-APGI" in lv["name"]
        if is_l0:
            # L0 sub-APGI status made visually unambiguous: distinct white +
            # hatched fill and a dashed grey border (not a tiny dark dot).
            circ = mpatches.Circle(
                (cx, cy),
                lv["r"],
                facecolor="white",
                edgecolor="#777777",
                lw=2.2,
                linestyle="--",
                hatch="////",
                zorder=30,
            )
        else:
            circ = mpatches.Circle(
                (cx, cy),
                lv["r"],
                facecolor=lv["color"],
                edgecolor=lv["color"],
                lw=1.8,
                linestyle="-",
                alpha=0.70,
                zorder=len(LEVELS) - LEVELS.index(lv),
            )
        ax_rings.add_patch(circ)
        label = lv["name"].split("\n")[0]
        ax_rings.text(
            cx,
            cy + lv["r"] - 0.025,
            label,
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=(
                "#777777"
                if is_l0
                else ("#333333" if lv["color"] != "#fdcc8a" else "#555555")
            ),
            zorder=31,
        )

    # ── Neuromodulatory boundary carriers, derived from each outer level's
    #    own neuromodulator so they can never contradict the parameter table
    #    (fixes the prior L2 'NE (fast)' vs ACh inconsistency). ──────────────
    CARRIERS = [
        (
            LEVELS[i]["name"].split("\n")[0],
            LEVELS[i + 1]["name"].split("\n")[0],
            LEVELS[i]["neuromod"],
        )
        for i in range(len(LEVELS) - 1)
    ]

    # ── Coupling: visible curved arrows OUTSIDE the ring circumference ──────
    r_out = LEVELS[0]["r"]
    # Top-down (α_down): curved arrow on the right, pointing inward
    ax_rings.add_patch(
        FancyArrowPatch(
            (cx + r_out + 0.34, cy + 0.20),
            (cx + r_out + 0.02, cy + 0.02),
            connectionstyle="arc3,rad=-0.38",
            arrowstyle="-|>",
            mutation_scale=16,
            color="#2166ac",
            lw=2.2,
            zorder=25,
        )
    )
    ax_rings.text(
        cx + r_out + 0.36,
        cy + 0.24,
        r"$\alpha_{\mathrm{down}}$",
        fontsize=9,
        color="#2166ac",
        fontweight="bold",
    )
    # Bottom-up (α_up): curved arrow on the left, pointing outward
    ax_rings.add_patch(
        FancyArrowPatch(
            (cx - r_out - 0.02, cy - 0.02),
            (cx - r_out - 0.34, cy - 0.20),
            connectionstyle="arc3,rad=-0.38",
            arrowstyle="-|>",
            mutation_scale=16,
            color="#d6604d",
            lw=2.2,
            zorder=25,
        )
    )
    ax_rings.text(
        cx - r_out - 0.36,
        cy - 0.26,
        r"$\alpha_{\mathrm{up}}$",
        fontsize=9,
        color="#d6604d",
        fontweight="bold",
        ha="right",
    )

    # Carrier list (right margin, no overlap with rings)
    ax_rings.text(
        cx + r_out + 0.12,
        cy + 0.66,
        "Cross-level carriers (top-down):",
        fontsize=6.8,
        color="#333333",
        fontweight="bold",
        ha="left",
    )
    for k, (outer, inner, mod) in enumerate(CARRIERS):
        ax_rings.text(
            cx + r_out + 0.12,
            cy + 0.57 - k * 0.085,
            f"{outer}→{inner}:  {mod}",
            fontsize=6.5,
            color="#555555",
            ha="left",
        )

    # ── Substrate zone labels, vertically staggered on the left (no merge) ──
    brackets = [
        ("Neuroendocrine\n(L3–L4)", "#a50f15", cy + 0.42, r_out),
        (
            "Cortical reservoir\n(L1–L2, ≤ ~2 s\nfading-memory)",
            "#fc8d59",
            cy + 0.02,
            LEVELS[2]["r"],
        ),
        ("Reflexive\n(L0)", "#777777", cy - 0.42, LEVELS[4]["r"]),
    ]
    for label, color, ly, r_edge in brackets:
        ax_rings.annotate(
            "",
            xy=(cx - r_edge, cy + (ly - cy) * 0.18),
            xytext=(-0.50, ly),
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0, alpha=0.6),
        )
        ax_rings.text(
            -0.52,
            ly,
            label,
            ha="right",
            va="center",
            fontsize=6.8,
            color=color,
            multialignment="right",
            fontweight="bold",
        )

    # L0 explanatory note
    ax_rings.text(
        cx,
        -0.70,
        "L0 is sub-APGI (no Π₀/θ₀/τ_ign,0); supplies sensory drive to L1",
        ha="center",
        fontsize=7,
        color="#888888",
        style="italic",
    )

    ax_rings.set_title(
        "Russian Doll concentric-ring hierarchy", fontsize=9, fontweight="bold"
    )

    # ── Table panel ───────────────────────────────────────────────────────
    ax_table.axis("off")
    col_labels = ["Level", "τ_int", "Substrate", "Neuromod.", "APGI params"]
    table_data = [
        [
            lv["name"].replace("\n(sub-APGI)", "\n(sub)"),
            lv["tau"],
            lv["substrate"],
            lv["neuromod"],
            lv["apgi"],
        ]
        for lv in LEVELS
    ]
    colors_rows = [
        ["#fde0d9" if "sub" not in lv["name"] else "#f0f0f0"] * 5 for lv in LEVELS
    ]
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
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.45)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", fontweight="bold")
        elif row <= len(LEVELS):
            cell.set_facecolor(LEVELS[row - 1]["color"] + "44")
    ax_table.set_title("Level parameters (summary)", fontsize=9, fontweight="bold")
    ax_table.text(
        0.5,
        0.04,
        "Full parameter values: see Table 1 (manuscript).",
        transform=ax_table.transAxes,
        ha="center",
        fontsize=6.8,
        color="#888888",
        style="italic",
    )

    fig.suptitle(
        "Figure 1 — Five-Level Russian Doll Hierarchical Architecture (§2.3)\n"
        "Paper 3",
        fontsize=11,
        fontweight="bold",
        y=1.01,
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
