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
# CRITICAL (spec Fig.3 prompt): every Validation-status cell must read ONLY
# "Theoretically specified" or "Pending" -- no "Confirmed", "Partial", or
# "Speculative" anywhere. Those three legacy labels are removed entirely
# (previously present here as dead/misleading legend entries that never
# actually appeared in the table, which was itself a labelling-accuracy bug).
STATUS_COLORS = {
    "Theoretically specified": "#28a745",
    "Pending": "#856404",
}

# Verbatim from the PDF spec's reference table (Paper 4, Fig.3 "APGI's
# twelve falsifiable predictions by tier"). Columns: Prediction | Measurement
# protocol | Target effect size | Falsification criterion | Feasibility
# horizon | Validation status. Tier 3 (P1-P4) = testable "now" and marked
# "Theoretically specified" (sufficient theoretical grounding exists); Tier 2
# (P5-P8, "2-5 years") and Tier 1 (P9-P12, "5-10 years") = "Pending" (awaits
# empirical measurement). All twelve derive from the canonical relation
# Sₜ = Πᵉ|zᵉ| + Πᵢ_eff|zᵢ| > θₜ.
PREDICTIONS = [
    # P1–P4 Tier 3 (testable now)
    (
        "P1",
        "T3",
        "Nicotinic agonist lowers\neffective threshold",
        "Nicotine or α4β2 agonist vs placebo;\npsychophysics (d′) and EEG (P3b amplitude)",
        "d′ increase ≥ 0.3;\nP3b amplitude increase ≥ 20%",
        "No change in detection/P3b.",
        "now",
        "Theoretically specified",
    ),
    (
        "P2",
        "T3",
        "Near-threshold bistability\nwith critical slowing",
        "Staircase near threshold; trial-to-trial\nRT/EEG; estimate lag-1 autocorrelation (AC1)",
        "AC1 increase ≥ 0.2\nwithin 10% of threshold",
        "A continuous graded relationship.",
        "now",
        "Theoretically specified",
    ),
    (
        "P3",
        "T3",
        "200–300 ms frontoparietal\nTMS abolishes report, spares priming",
        "TMS at 200–300 ms over frontoparietal\ncortex; measure report (d′) and priming",
        "Report d′ drop ≥ 0.8; priming\neffect preserved (≤ 20% reduction)",
        "Both disrupted equally\nor access persists.",
        "now",
        "Theoretically specified",
    ),
    (
        "P4",
        "T3",
        "Sₜ > θₜ predicts access\nat d′ ≥ 0.5",
        "Single-trial decoding of Sₜ and θₜ\n(EEG/iEEG); predict access on held-out trials",
        "AUC ≥ 0.80; accuracy\nimprovement ≥ 10% over chance",
        "d′ < 0.5 (N ≥ 2,000).",
        "now",
        "Theoretically specified",
    ),
    # P5–P8 Tier 2 (multi-voxel information-theoretic analyses)
    (
        "P5",
        "T2",
        "Attentional cueing raises\nmutual information by ≥1 bit",
        "Cue vs no-cue; estimate stimulus–\nbrain mutual information (MI)",
        "ΔMI ≥ 1 bit",
        "No increase.",
        "2–5 years",
        "Pending",
    ),
    (
        "P6",
        "T2",
        "Conscious bandwidth\nasymptotes ~40 bits/s",
        "Rapid RSVP across modalities; estimate\ninformation rate (bits/s) vs stimulation rate",
        "Asymptote = 30–50 bits/s;\nfit saturating curve",
        "Modality variation or\nsustained > 100 bits/s.",
        "2–5 years",
        "Pending",
    ),
    (
        "P7",
        "T2",
        "θₜ approximates the\nNeyman–Pearson optimum",
        "Model-based ROC analysis; compare\nθₜ to NP-optimal criterion",
        "Criterion within 2 SD\nof NP optimum",
        "Deviation > 2 SD.",
        "2–5 years",
        "Pending",
    ),
    (
        "P8",
        "T2",
        "Interoceptive transfer entropy\n> exteroceptive by ≥ 0.05 bits/trial",
        "iEEG/EEG; compute TE from interoceptive vs\nexteroceptive regions to frontoparietal cortex",
        "ΔTE ≥ 0.05 bits/trial",
        "TE_intero ≤ TE_extero.",
        "2–5 years",
        "Pending",
    ),
    # P9–P12 Tier 1 (metabolic calorimetry paradigms)
    (
        "P9",
        "T1",
        "Ignition trials cost more\nfrontoparietal energy (CMRO₂/glucose)",
        "Simultaneous fMRI/PET (CMRO₂ or glucose);\ncompare ignition vs matched sub-threshold trials",
        "ΔCMRO₂ or Δglucose\n≥ 10% after controls",
        "No differential after controls.",
        "5–10 years",
        "Pending",
    ),
    (
        "P10",
        "T1",
        "Threshold-gated processing more\nenergy-efficient than continuous",
        "Adaptive (threshold-gated) vs continuous\nstimulation/task; energy per bit or per correct report",
        "≥ 20% lower\nenergy cost",
        "Continuous matches or beats it.",
        "5–10 years",
        "Pending",
    ),
    (
        "P11",
        "T1",
        "Metabolic depletion\nelevates θₜ",
        "Induce metabolic depletion (hypercapnia/\nhypoxia or TMS); measure θₜ via staircases",
        "θₜ increase ≥ 0.3 SD",
        "No effect or lowered threshold.",
        "5–10 years",
        "Pending",
    ),
    (
        "P12",
        "T1",
        "Single-ignition cost exceeds the\nLandauer minimum by κ (≈100 ATP/bit)",
        "Estimate energy cost per ignition (ATP\nequivalents); compare to Landauer limit (k_BT ln 2)",
        "κ ≥ 100 ATP/bit\n(~4×10⁻²⁰ J/bit)",
        "κ below the canonical floor or\nBOLD/direct diverge > 10×.",
        "5–10 years",
        "Pending",
    ),
]

COLS = [
    "Prediction",
    "Measurement protocol",
    "Target effect size",
    "Falsification criterion",
    "Feasibility\nhorizon",
    "Validation\nstatus",
]
COL_WIDTHS = [0.19, 0.23, 0.15, 0.19, 0.07, 0.11]  # fractions of table width


def plot(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(19, 12))
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
                fontsize=6.2,
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
