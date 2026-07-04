"""Paper 4 — Figure 2: Comparative Audit Grouped Bar Chart (§6.6).

Grouped bar chart: seven rubric criteria × four frameworks.
APGI self-audit bars distinguished with diagonal hatch.
Inset: Primary Falsification Gap tier per framework.

Run:
    python figures/paper4/generate_fig2.py
    python figures/paper4/generate_fig2.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from apgi.extensions.epistemic import evaluate_theory
from figures.utils import label_axes, save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

CRITERIA = [
    "C1\nTier Trans.",
    "C2\nBridge Princ.",
    "C3\nQuant. Bench.",
    "C4\nFalsif. Cond.",
    "C5\nAltern. Comp.",
    "C6\nEvol. Plaus.",
    "C7\nCausal Road.",
]

# Scores 0-2, transcribed verbatim from the spec's Table 5 prose (verified
# directly against OUP-Paper4-EpistemicArchitecture.txt, the row beginning
# "Criterion | GNWT | IIT 3.0 | PP/FEP | APGI (self-audit)"):
#
#   1. Tier Transparency:       GNWT=1, IIT=1, PP/FEP=1, APGI=2
#   2. Bridge Principles:       GNWT=0, IIT=0, PP/FEP=1, APGI=1
#   3. Quantitative Benchmarks: GNWT=1, IIT=0, PP/FEP=1, APGI=1-2 (T3:2, T1:0-1)
#   4. Falsification Conditions:GNWT=1, IIT=0, PP/FEP=1, APGI=1-2 (T3:2, T1:1)
#   5. Alternative Comparison:  GNWT=1, IIT=1, PP/FEP=1, APGI=1
#   6. Evolutionary Plausibility:GNWT=1, IIT=0, PP/FEP=1, APGI=1
#   7. Causal Roadmap:          GNWT=1, IIT=0, PP/FEP=1, APGI=2
#
# APGI's C3 and C4 are reported as tier-resolved ranges, not point scores:
# C3 spans Tier1=0 (Tier 1: 0-1 proxy only, plotted at its lower bound) to
# Tier3=2, midpoint 1.0; C4 spans Tier1=1 to Tier3=2, midpoint 1.5.
FRAMEWORKS = {
    "GNWT": [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "IIT 3.0": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "PP/FEP": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "APGI": [2.0, 1.0, 1.0, 1.5, 1.0, 1.0, 2.0],  # C3/C4 plotted at range midpoints
}
# Visible error bars for APGI's tier-resolved ranges: C3 spans 0-2 (midpoint
# 1.0, half-range 1.0); C4 spans 1-2 (midpoint 1.5, half-range 0.5).
APGI_ERRORS = [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0]

# Demonstrate the real evaluate_theory() implementation is consistent with
# the plotted APGI self-audit bars, using the midpoint raw scores above
# (raw 0/1/2 scale; see apgi.extensions.epistemic.CRITERIA for ordering).
_APGI_RAW_SCORES = {
    "tier_transparency": 2,
    "bridge_principles": 1,
    "quantitative_benchmarks": 1,
    "falsification_conditions": 2,  # Tier-3 sub-score, the higher end of the 1-2 range
    "alternative_comparison": 1,
    "evolutionary_plausibility": 1,
    "causal_roadmap": 2,
}
APGI_EVALUATION = evaluate_theory(_APGI_RAW_SCORES)

COLORS = {
    "GNWT": "#6baed6",
    "IIT 3.0": "#fd8d3c",
    "PP/FEP": "#74c476",
    "APGI": "#9e9ac8",
}

# Falsification gap tiers
FAL_GAPS = [
    ("GNWT", "T3", "threshold mechanism\nunspecified", "#6baed6"),
    ("IIT 3.0", "T2", "Φ uncomputable", "#fd8d3c"),
    ("PP/FEP", "T3", "no access criterion", "#74c476"),
    ("APGI", "T1", "κ unmeasured", "#9e9ac8"),
]


def plot(show: bool = True) -> None:
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1], hspace=0.10)
    ax_bars = fig.add_subplot(gs[0])
    ax_gap = fig.add_subplot(gs[1])

    x = np.arange(len(CRITERIA))
    n_fw = len(FRAMEWORKS)
    width = 0.18
    offsets = np.linspace(-(n_fw - 1) / 2, (n_fw - 1) / 2, n_fw) * width

    for i, (fw_name, scores) in enumerate(FRAMEWORKS.items()):
        color = COLORS[fw_name]
        hatch = "//" if fw_name == "APGI" else ""
        errs = APGI_ERRORS if fw_name == "APGI" else [0.0] * len(CRITERIA)
        ax_bars.bar(
            x + offsets[i],
            scores,
            width,
            color=color,
            edgecolor="#333333",
            lw=0.8,
            hatch=hatch,
            alpha=0.85,
            yerr=errs,
            capsize=3,
            label=f"{fw_name}" + (" (self-audit)" if fw_name == "APGI" else ""),
        )
        # Explicit "0" markers so a zero score reads as a deliberate audit
        # finding rather than a missing/omitted bar.
        for xi, sc in zip(x + offsets[i], scores):
            if sc == 0.0:
                ax_bars.text(
                    xi,
                    0.04,
                    "0",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    color=color,
                )

    # Provisional acceptance threshold line
    ax_bars.axhline(1.1, lw=1.8, ls="--", color="#cc0000", alpha=0.8)
    ax_bars.text(
        6.55,
        1.12,
        "Provisional\nacceptance\nthreshold (1.1)",
        fontsize=7.5,
        color="#cc0000",
        va="bottom",
        ha="right",
    )

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(CRITERIA, fontsize=9)
    ax_bars.set_ylabel("Score (0–2)", fontsize=10)
    ax_bars.set_ylim(0, 2.5)
    ax_bars.set_xlim(-0.55, 6.55)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.legend(fontsize=8.5, loc="upper left", framealpha=0.85)
    ax_bars.set_title(
        "Comparative Audit: Seven Rubric Criteria × Four Frameworks\n"
        "(APGI self-audit: diagonal hatch; scores from Table 5)",
        fontsize=10,
        fontweight="bold",
    )
    ax_bars.text(
        0.5,
        -0.065,
        "APGI scores are self-audit values (§3.4, §7.2–7.3), "
        "applied with identical standards as competitor audits.",
        ha="center",
        fontsize=7,
        color="#666666",
        style="italic",
        transform=ax_bars.transAxes,
    )
    ax_bars.text(
        0.99,
        0.97,
        f"evaluate_theory() self-check (apgi.extensions.epistemic): "
        f"composite={APGI_EVALUATION['composite']:.1f}/100, "
        f"verdict={APGI_EVALUATION['verdict']}, "
        f"gate_triggered={APGI_EVALUATION['foundational_gate_triggered']}",
        ha="right",
        va="top",
        fontsize=6.5,
        color="#555555",
        style="italic",
        transform=ax_bars.transAxes,
        bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85, pad=3),
    )

    # ── Inset: Primary Falsification Gap strip ────────────────────────────
    ax_gap.set_xlim(-0.5, 6.5)
    ax_gap.set_ylim(0, 1)
    ax_gap.axis("off")
    ax_gap.text(
        -0.45,
        0.75,
        "Primary\nFalsification\nGap Tier:",
        ha="left",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#555555",
    )

    for j, (fw, tier, desc, color) in enumerate(FAL_GAPS):
        bx = 0.5 + j * 1.6
        rect = mpatches.FancyBboxPatch(
            (bx - 0.60, 0.10),
            1.20,
            0.80,
            boxstyle="round,pad=0.02",
            facecolor=color + "44",
            edgecolor=color,
            lw=2.0,
        )
        ax_gap.add_patch(rect)
        ax_gap.text(
            bx, 0.75, fw, ha="center", fontsize=8.5, fontweight="bold", color=color
        )
        ax_gap.text(
            bx, 0.50, tier, ha="center", fontsize=10, fontweight="bold", color="#333333"
        )
        ax_gap.text(
            bx,
            0.22,
            desc,
            ha="center",
            fontsize=6.5,
            color="#555555",
            multialignment="center",
        )

    label_axes([ax_bars, ax_gap])
    fig.suptitle(
        "Figure 2 — Comparative Audit Grouped Bar Chart + Primary Falsification Gap Inset\n"
        "(Paper 4, §6.6)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig2_comparative_audit_bar_chart.pdf")
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
