"""Paper 4 — Figure S1: APGI Twelve-Prediction Matrix by Tier (supplementary / Table S4).

Structured visual matrix: P1–P12 grouped by tier (T3 green, T2 amber, T1 red).
Columns: Prediction, Protocol, Effect-size target, Falsification criterion,
Feasibility horizon, Current status.

Run:
    python figures/paper4/generate_figS1.py
    python figures/paper4/generate_figS1.py --no-show
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

TIER_COLORS = {"T3": "#d4edda", "T2": "#fff3cd", "T1": "#f8d7da"}
TIER_EDGE   = {"T3": "#28a745", "T2": "#856404", "T1": "#721c24"}
STATUS_COLORS = {
    "Confirmed":  "#28a745",
    "Partial":    "#2166ac",
    "Pending":    "#856404",
    "Speculative": "#888888",
}

PREDICTIONS = [
    # P1–P4 Tier 3
    ("P1", "T3", "HEP amplitude predicts P3b\ntrial-by-trial (Πⁱ proxy)",
     "EEG cardiac-phase", "r≥0.35", "HEP–P3b r<0.10", "Now", "Partial"),
    ("P2", "T3", "Detection rate higher\nat diastole vs. systole",
     "EEG cardiac-phase", "d'≥0.5", "Null cardiac-phase effect", "Now", "Pending"),
    ("P3", "T3", "Somatic-marker agent out-\nperforms β-lesion under volatility",
     "Agent simulation", "η²≥0.25", "No reward advantage", "Now", "Confirmed"),
    ("P4", "T3", "vmPFC–insula PPI higher in\nanticipation > outcome",
     "fMRI PPI", "β≥0.20", "PPI null in anticipation", "Now", "Pending"),
    # P5–P8 Tier 2
    ("P5", "T2", "High-gamma bimodal at\nfrontoparietal threshold",
     "iEEG", "BF≥10", "Unimodal gamma", "2–5 yr", "Pending"),
    ("P6", "T2", "pIC TMS abolishes\nHEP–PCI coupling",
     "TMS-EEG", "ΔPCI≥0.10", "No TMS dissociation", "2–5 yr", "Pending"),
    ("P7", "T2", "Masking effectiveness\n∝ ρ_crit (not stimulus energy)",
     "Visual masking LNN", "AUC≥0.80", "ρ_crit null predictor", "2–5 yr", "Speculative"),
    ("P8", "T2", "1/f exponent inflections at\nL1–L4 maturation ages",
     "Developmental EEG", "H inflection Δ≥0.05", "Absent/out-of-sequence", "2–5 yr", "Speculative"),
    # P9–P12 Tier 1
    ("P9",  "T1", "κ (ATP/bit) within\n10¹³–10¹⁴ range",
     "³¹P-MRS + iEEG", "95% CI overlaps range", "κ outside range", "5–10 yr", "Speculative"),
    ("P10", "T1", "BOLD-calibrated fMRI\nmatches double-bridge estimate",
     "Calibrated fMRI", "r²≥0.60", "Systematic deviation", "5–10 yr", "Speculative"),
    ("P11", "T1", "Bistability preserved\nacross 94% of parameter space",
     "Monte Carlo", "94.2%", "<80% bistability", "Now", "Confirmed"),
    ("P12", "T1", "Post-block θ elevation\nin DoC patients",
     "DoC iEEG", "Cohen's d≥0.8", "No threshold re-elevation", "5–10 yr", "Pending"),
]

COLS = ["Prediction", "Protocol", "Effect target", "Falsification", "Horizon", "Status"]
COL_WIDTHS = [0.28, 0.14, 0.12, 0.16, 0.08, 0.10]  # fractions of table width


def plot(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    N_ROWS = len(PREDICTIONS) + 1  # +1 header
    row_h = 0.90 / N_ROWS
    x_starts = np.cumsum([0] + COL_WIDTHS[:-1]) + 0.02

    # ── Header row ────────────────────────────────────────────────────────
    for j, (col, x) in enumerate(zip(COLS, x_starts)):
        rect = mpatches.FancyBboxPatch(
            (x, 0.92), COL_WIDTHS[j] - 0.005, row_h,
            boxstyle="square,pad=0.005",
            facecolor="#333333", edgecolor="#333333", lw=0.5, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x + COL_WIDTHS[j] / 2, 0.92 + row_h / 2, col,
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white", zorder=4)

    # ── Tier group labels ─────────────────────────────────────────────────
    tier_row_ranges = {"T3": (0, 3), "T2": (4, 7), "T1": (8, 11)}
    tier_labels = {"T3": "Tier 3\nComputational", "T2": "Tier 2\nInfo-Theoretic",
                   "T1": "Tier 1\nThermodynamic"}

    for i, pred in enumerate(PREDICTIONS):
        pid, tier, text, protocol, eff, falsif, horizon, status = pred
        row_y = 0.91 - (i + 1) * row_h

        bg_color = TIER_COLORS[tier]
        edge_color = TIER_EDGE[tier]

        row_data = [f"{pid}: {text}", protocol, eff, falsif, horizon, status]

        for j, (cell_text, x) in enumerate(zip(row_data, x_starts)):
            cell_bg = bg_color
            if j == 5:  # status column
                cell_bg = STATUS_COLORS.get(status, "#ffffff") + "22"
            rect = mpatches.FancyBboxPatch(
                (x, row_y), COL_WIDTHS[j] - 0.005, row_h,
                boxstyle="square,pad=0.003",
                facecolor=cell_bg,
                edgecolor=edge_color,
                lw=0.6, zorder=3,
            )
            ax.add_patch(rect)
            cell_color = STATUS_COLORS.get(status, "#333333") if j == 5 else "#333333"
            ax.text(x + COL_WIDTHS[j] / 2, row_y + row_h / 2, cell_text,
                    ha="center", va="center", fontsize=6.5, zorder=4,
                    color=cell_color,
                    fontweight="bold" if j == 5 else "normal",
                    multialignment="center")

    # Tier bracket annotations (left margin)
    for tier, (r0, r1) in tier_row_ranges.items():
        y_top = 0.91 - (r0 + 1) * row_h + row_h
        y_bot = 0.91 - (r1 + 1) * row_h
        color = TIER_EDGE[tier]
        ax.annotate("", xy=(0.01, y_bot), xytext=(0.01, y_top),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=2.0))
        ax.text(0.005, (y_top + y_bot) / 2, tier_labels[tier],
                ha="right", va="center", fontsize=7, color=color,
                fontweight="bold", multialignment="center")

    # Status legend
    handles = [mpatches.Patch(color=c, label=s)
               for s, c in STATUS_COLORS.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              title="Validation status", title_fontsize=8, framealpha=0.9,
              bbox_to_anchor=(0.99, 0.01))

    ax.set_title(
        "Figure S1 / Table S4 — APGI Twelve-Prediction Matrix by Tier\n"
        "(Paper 4, supplementary; cross-referenced from §7.4)\n"
        r"Canonical equation: $S_t = \Pi^e |z^e| + \Pi^i_{\mathrm{eff}} |z^i| > \theta_t$",
        fontsize=10, fontweight="bold", y=0.995,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figS1_twelve_prediction_matrix.pdf")
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
