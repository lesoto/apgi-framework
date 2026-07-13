"""Paper 1 — Figure 5: Epistemic Tier Architecture and Falsification Mapping.

A three-row stacked matrix mapping the three epistemic tiers to their
empirical protocols, primary neural/behavioural measures, and specific
falsification thresholds, stacked Tier 3 (top) -> Tier 1 (bottom) to match
Paper 4 Fig 1's tier ordering:
    Tier 3 Computational        -> Protocols 2 & 3 -> model-fit falsifiers
    Tier 2 Information-Theoretic -> Protocol 1      -> mutual-information falsifiers
    Tier 1 Thermodynamic        -> Protocol 4      -> calorimetric falsifiers
Vertical downward arrows between tiers are labelled "bridge principles".

This figure was missing from the repository (Paper 1 has fig1-4 + figS1/S2
implemented, but no fig5), even though APGI-Figures.pdf specifies it
immediately after Figure 4 / before Figure S1. This script closes that gap.

Run:
    python figures/paper1/generate_fig5.py
    python figures/paper1/generate_fig5.py --no-show
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

# Tier colours match figures/paper4/generate_fig1.py's tier palette exactly,
# so the two figures' tier-colour coding never drifts apart.
TIER_ROWS = [
    {
        "tier": "Tier 3",
        "name": "Computational",
        "protocol": "Protocols 2 & 3\nPerturbational Complexity &\nParameter Recovery",
        "measure": "Model Fit\nGoodness-of-fit ($R^2$),\nparameter-recovery correlation",
        "falsifier": "Model-Fit Falsifiers\n$R^2 < 0.50$ or parameter\nrecovery $\\rho < 0.50$",
        "color": "#2166ac",
    },
    {
        "tier": "Tier 2",
        "name": "Information-Theoretic",
        "protocol": "Protocol 1\nInformation Decomposition\n(HEP, $\\Phi$, PCI)",
        "measure": "Mutual Information\n$I(S_t; I_t)$ (bits)",
        "falsifier": "Mutual-Information Falsifiers\n$I(S_t; I_t) \\leq 0.05$ bits",
        "color": "#4dac26",
    },
    {
        "tier": "Tier 1",
        "name": "Thermodynamic",
        "protocol": "Protocol 4\nCalorimetry (Stimulus, Rest,\n& Ignition Phases)",
        "measure": r"Heat Release Rate $\dot{Q}$" "\n(J s$^{-1}$)",
        "falsifier": "Calorimetric Falsifiers\n$\\dot{Q} \\leq 0$\n(no excess heat)",
        "color": "#d6604d",
    },
]

COLS = ["Epistemic Tier", "Empirical Protocol(s)", "Primary Neural/\nBehavioural Measure", "Specific Falsification\nThreshold"]
COL_X = [0.02, 0.24, 0.50, 0.76]
COL_W = [0.20, 0.24, 0.24, 0.22]


def _row_box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=8.2, fontweight="normal", textcolor="#222222"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.008",
        facecolor=facecolor, edgecolor=edgecolor, lw=1.2, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, fontweight=fontweight, color=textcolor,
        multialignment="center", zorder=4,
    )


def plot(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_rows = len(TIER_ROWS)
    row_h = 0.22
    header_y = 0.90
    row_gap = 0.06  # vertical gap between rows for the bridge-principle arrows

    # Header row
    for col, x, w in zip(COLS, COL_X, COL_W):
        ax.text(
            x + w / 2, header_y + 0.03, col, ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#333333", multialignment="center",
        )

    row_tops = [header_y - i * (row_h + row_gap) for i in range(n_rows)]

    for i, row in enumerate(TIER_ROWS):
        y_top = row_tops[i] - row_h
        color = row["color"]
        tier_text = f"{row['tier']}\n{row['name']}"
        cells = [tier_text, row["protocol"], row["measure"], row["falsifier"]]
        for j, (x, w, text) in enumerate(zip(COL_X, COL_W, cells)):
            is_tier_col = j == 0
            _row_box(
                ax, x, y_top, w, row_h, text,
                facecolor=color if is_tier_col else color + "22",
                edgecolor=color, fontsize=9.5 if is_tier_col else 8.0,
                fontweight="bold" if is_tier_col else "normal",
                textcolor="white" if is_tier_col else "#222222",
            )

        # Bridge-principle downward arrow to the next tier (Tier 3->2, Tier 2->1).
        if i < n_rows - 1:
            arrow_x = COL_X[0] + COL_W[0] / 2
            y_from = y_top
            y_to = row_tops[i + 1] - row_h
            ax.annotate(
                "", xy=(arrow_x, y_to + row_h + 0.008), xytext=(arrow_x, y_from - 0.008),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8), zorder=2,
            )
            ax.text(
                arrow_x + 0.015, (y_from + y_to + row_h) / 2, "bridge\nprinciples",
                ha="left", va="center", fontsize=6.8, color="#555555",
                style="italic", multialignment="center",
            )

    # Legend: only glyphs that actually appear in the diagram body (tier
    # colour swatches + a single "Measure" entry) -- no "?" placeholder icon
    # and no duplicated "Measure" swatches (CRITICAL correction).
    legend_handles = [
        mpatches.Patch(facecolor=row["color"], edgecolor=row["color"], label=f"{row['tier']} ({row['name']})")
        for row in TIER_ROWS
    ]
    legend_handles.append(
        mpatches.Patch(facecolor="#999999", edgecolor="#999999", alpha=0.35, label="Measure")
    )
    ax.legend(
        handles=legend_handles, loc="lower center", ncol=4, fontsize=7.5,
        frameon=False, bbox_to_anchor=(0.5, -0.06),
    )

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig5_epistemic_tier_falsification_mapping.pdf")
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
