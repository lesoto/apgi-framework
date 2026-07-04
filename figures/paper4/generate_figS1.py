"""Paper 4 — Figure S1: APGI Twelve-Prediction Matrix by Tier (supplementary / Table S4).

Structured visual matrix: P1–P12 grouped by tier (T3 green, T2 amber, T1 red).
Columns: Prediction, Protocol, Effect-size target, Falsification criterion,
Feasibility horizon, Current status.

Note on caption numbering: the spec text (OUP-Paper4-EpistemicArchitecture.txt)
is internally inconsistent about whether this content is captioned "Figure 3"
(§7.4, appearing right after the twelve-predictions narrative) or "Figure S1"
(cross-referenced from §7.4 as Table S4). We keep this script's existing
filename/numbering (figS1_twelve_prediction_matrix) to avoid breaking other
cross-references, and surface both captions in the on-figure title.

The twelve predictions themselves (P1-P12), their tiers, and their
falsification criteria are transcribed verbatim from the spec's §7.4
"Testable Predictions by Tier" narrative (Tier 3 = P1-P4, Tier 2 = P5-P8,
Tier 1 = P9-P12) - NOT from any other hypothesis register (H-series/EP/
Pred-x.y) in the spec, which belongs to a different document's content.

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
TIER_EDGE = {"T3": "#28a745", "T2": "#856404", "T1": "#721c24"}
STATUS_COLORS = {
    "Confirmed": "#28a745",
    "Partial": "#2166ac",
    "Pending": "#856404",
    "Speculative": "#888888",
}

# Verbatim from spec §7.4 "Testable Predictions by Tier":
#   Tier 3 (P1-P4): testable now with TMS-EEG and pharmacology.
#   Tier 2 (P5-P8): require multi-voxel information-theoretic analyses
#     achievable with current neuroimaging.
#   Tier 1 (P9-P12): require metabolic calorimetry paradigms not yet
#     standardized -- the most demanding frontier.
# All twelve derive from the canonical relation Sₜ = Πᵉ|zᵉ| + Πᵢ_eff|zᵢ| > θₜ.
# Status column: none of P1-P12 have been empirically run per the spec text
# (§7.4 explicitly distinguishes these forward-looking predictions from the
# self-audit's "what has [been confirmed]"), so all twelve are marked
# "Pending" -- there is no per-prediction confirmed/partial status given in
# the spec to justify otherwise.
PREDICTIONS = [
    # P1–P4 Tier 3 (testable now)
    (
        "P1",
        "T3",
        "Nicotinic agonist lowers\neffective threshold",
        "Pharmacology + EEG",
        "improved detection,\nearlier P3b",
        "No change",
        "Now",
        "Pending",
    ),
    (
        "P2",
        "T3",
        "Near-threshold stimuli elicit\nbistable percepts, critical slowing",
        "Psychophysics + EEG",
        "bistable regime\nw/ critical slowing",
        "Continuous graded relationship",
        "Now",
        "Pending",
    ),
    (
        "P3",
        "T3",
        "TMS (200–300ms) to frontoparietal\ncortex selectively abolishes report",
        "TMS-EEG",
        "report abolished,\npriming spared",
        "Disrupts both equally, or\naccess persists",
        "Now",
        "Pending",
    ),
    (
        "P4",
        "T3",
        "Sₜ>θₜ predicts access at\nd′≥0.5, single-trial",
        "Single-trial EEG,\nLOPO-CV",
        "d′≥0.5; partial R²≥0.05\nfor Πᵉ and Πⁱ_eff",
        "d′<0.5, or stimulus intensity\nalone matches within Δd′ 0.05",
        "Now",
        "Pending",
    ),
    # P5–P8 Tier 2 (multi-voxel information-theoretic analyses)
    (
        "P5",
        "T2",
        "Attentional cueing raises\nstimulus–response MI by ≥1 bit",
        "Information-theoretic\nneuroimaging",
        "ΔMI≥1 bit",
        "No increase",
        "2–5 yr",
        "Pending",
    ),
    (
        "P6",
        "T2",
        "Conscious bandwidth asymptotes\nnear 40 bits/s across modalities",
        "Cross-modal\ninformation analysis",
        "~40 bits/s\nplateau",
        "Systematic modality variation, or\nsustained >100 bits/s after training",
        "2–5 yr",
        "Pending",
    ),
    (
        "P7",
        "T2",
        "Empirical θₜ approximates the\nNeyman–Pearson optimum",
        "Signal-detection\nanalysis",
        "within 2 SD of\nNP-optimal θₜ",
        "Systematic deviation >2 SD",
        "2–5 yr",
        "Pending",
    ),
    (
        "P8",
        "T2",
        "Interoceptive transfer entropy to\naccess report exceeds exteroceptive",
        "Transfer-entropy\nanalysis",
        "ΔTE≥0.05\nbits/trial",
        "TE_intero ≤ TE_extero",
        "2–5 yr",
        "Pending",
    ),
    # P9–P12 Tier 1 (metabolic calorimetry paradigms)
    (
        "P9",
        "T1",
        "Ignition trials cost more\nfrontoparietal energy",
        "CMRO₂/glucose\nmeasurement",
        "metabolic\ndifferential > 0",
        "No differential after controls",
        "5–10 yr",
        "Pending",
    ),
    (
        "P10",
        "T1",
        "Threshold-gated processing is more\nenergy-efficient than continuous",
        "Matched\nspiking-network sim.",
        "threshold-gated\n< continuous cost",
        "Continuous matches or beats it",
        "5–10 yr",
        "Pending",
    ),
    (
        "P11",
        "T1",
        "Metabolic depletion\nelevates θₜ",
        "Metabolic\ndepletion + EEG",
        "higher detection\nthreshold, reduced P3b",
        "No effect, or depletion\nlowers thresholds",
        "5–10 yr",
        "Pending",
    ),
    (
        "P12",
        "T1",
        "Single-ignition metabolic cost exceeds\nLandauer minimum by ~10¹⁸× (κ≈100 ATP/bit)",
        "BOLD proxy, then\n³¹P-MRS",
        "within 1 order\nof magnitude of κ",
        "Measured per-bit cost departs from κ\nby >1 order of magnitude, or BOLD/\n³¹P-MRS diverge by >10×",
        "5–10 yr",
        "Pending",
    ),
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
            (x, 0.92),
            COL_WIDTHS[j] - 0.005,
            row_h,
            boxstyle="square,pad=0.005",
            facecolor="#333333",
            edgecolor="#333333",
            lw=0.5,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            x + COL_WIDTHS[j] / 2,
            0.92 + row_h / 2,
            col,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    # ── Tier group labels ─────────────────────────────────────────────────
    # Per spec §7.4: Tier 3 = P1-P4 (rows 0-3), Tier 2 = P5-P8 (rows 4-7),
    # Tier 1 = P9-P12 (rows 8-11). Note P11 (metabolic depletion elevates
    # theta_t) is Tier 1 in the spec, NOT Tier 3 as a prior version of this
    # script incorrectly classified it.
    tier_row_ranges = {"T3": (0, 3), "T2": (4, 7), "T1": (8, 11)}
    tier_labels = {
        "T3": "Tier 3\nComputational",
        "T2": "Tier 2\nInfo-Theoretic",
        "T1": "Tier 1\nThermodynamic",
    }

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
                (x, row_y),
                COL_WIDTHS[j] - 0.005,
                row_h,
                boxstyle="square,pad=0.003",
                facecolor=cell_bg,
                edgecolor=edge_color,
                lw=0.6,
                zorder=3,
            )
            ax.add_patch(rect)
            cell_color = STATUS_COLORS.get(status, "#333333") if j == 5 else "#333333"
            ax.text(
                x + COL_WIDTHS[j] / 2,
                row_y + row_h / 2,
                cell_text,
                ha="center",
                va="center",
                fontsize=6.5,
                zorder=4,
                color=cell_color,
                fontweight="bold" if j == 5 else "normal",
                multialignment="center",
            )

    # Tier bracket annotations (left margin)
    for tier, (r0, r1) in tier_row_ranges.items():
        y_top = 0.91 - (r0 + 1) * row_h + row_h
        y_bot = 0.91 - (r1 + 1) * row_h
        color = TIER_EDGE[tier]
        ax.annotate(
            "",
            xy=(0.01, y_bot),
            xytext=(0.01, y_top),
            arrowprops=dict(arrowstyle="<->", color=color, lw=2.0),
        )
        ax.text(
            0.005,
            (y_top + y_bot) / 2,
            tier_labels[tier],
            ha="right",
            va="center",
            fontsize=7,
            color=color,
            fontweight="bold",
            multialignment="center",
        )

    # Status legend
    handles = [mpatches.Patch(color=c, label=s) for s, c in STATUS_COLORS.items()]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8,
        title="Validation status",
        title_fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(0.99, 0.01),
    )

    ax.set_title(
        "Figure S1 (spec also captions this Figure 3) / Table S4 — "
        "APGI Twelve-Prediction Matrix by Tier\n"
        "(Paper 4, §7.4; predictions P1–P12 verbatim from "
        "\"Testable Predictions by Tier\")\n"
        r"Canonical equation: $S_t = \Pi^e |z^e| + \Pi^i_{\mathrm{eff}} |z^i| > \theta_t$",
        fontsize=10,
        fontweight="bold",
        y=0.995,
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
