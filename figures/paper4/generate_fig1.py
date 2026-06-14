"""Paper 4 — Figure 1: Three-Tier Epistemic Architecture with Bridge Principles (§3.2).

Three-tier vertical stack with valid downward bridges (solid) and
absent upward bridges (dashed red X). Four-step double-bridge calculation inset.
L4 (phenomenal) deliberately excluded.

Run:
    python figures/paper4/generate_fig1.py
    python figures/paper4/generate_fig1.py --no-show
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

TIERS = [
    {
        "name": "Tier 3 — Computational",
        "currency": "prediction errors ε, precision Π,\nvariational F, d′, AUC",
        "evidence": "single-trial EEG, model fit, BIC/AIC",
        "color": "#2166ac",
        "y": 0.72,
    },
    {
        "name": "Tier 2 — Information-Theoretic",
        "currency": "bits, nats, mutual information,\nchannel capacity, Φ",
        "evidence": "transfer entropy, bias-corrected MI, Φ",
        "color": "#4dac26",
        "y": 0.42,
    },
    {
        "name": "Tier 1 — Thermodynamic",
        "currency": "joules, ATP, watts, entropy",
        "evidence": "³¹P-MRS, two-photon imaging,\nBOLD-calibrated fMRI (proxy)",
        "color": "#d6604d",
        "y": 0.12,
    },
]

BRIDGES = [
    {
        "label": "T3→T2: κ bridge (ATP/bit)\napprox. inference → MI optimisation\nStatus: grounded, κ unmeasured",
        "y_from": 0.72,
        "y_to": 0.52,
        "valid": True,
        "color": "#4dac26",
    },
    {
        "label": "T2→T1: Landauer bridge\nE ≥ kT·ln2 per bit erased\nStatus: theoretically bounded, κ unmeasured",
        "y_from": 0.42,
        "y_to": 0.22,
        "valid": True,
        "color": "#d6604d",
    },
    {
        "label": "T3→T1: Double bridge (3→2→1)\nrequires both above\nStatus: composite, unvalidated",
        "y_from": 0.72,
        "y_to": 0.22,
        "valid": True,
        "color": "#888888",
        "offset_x": -0.25,
    },
]

ABSENT_BRIDGES = [
    {"y_from": 0.12, "y_to": 0.42, "label": "T1→T2\nNot yet specified"},
    {"y_from": 0.42, "y_to": 0.72, "label": "T2→T3\nNot yet specified"},
]

CALC_STEPS = [
    "(1) Bits processed per ignition",
    "(2) × Landauer min. (3×10⁻²¹ J at 310K)",
    "(3) × neural inefficiency (~10¹³–10¹⁴)",
    "(4) compare to PET/BOLD expenditure",
]


def plot(show: bool = True) -> None:
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1.5, 1.2], wspace=0.30)
    ax_tiers = fig.add_subplot(gs[0])
    ax_bridges = fig.add_subplot(gs[1])
    ax_calc = fig.add_subplot(gs[2])

    for ax in [ax_tiers, ax_bridges, ax_calc]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ── Tier boxes ────────────────────────────────────────────────────────
    BOX_H = 0.20
    for tier in TIERS:
        rect = mpatches.FancyBboxPatch(
            (0.05, tier["y"]),
            0.90,
            BOX_H,
            boxstyle="round,pad=0.01",
            facecolor=tier["color"] + "22",
            edgecolor=tier["color"],
            lw=2.5,
            zorder=3,
        )
        ax_tiers.add_patch(rect)
        ax_tiers.text(
            0.50,
            tier["y"] + BOX_H / 2 + 0.02,
            tier["name"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=tier["color"],
            zorder=4,
        )
        ax_tiers.text(
            0.50,
            tier["y"] + 0.04,
            f"Currency: {tier['currency']}\nEvidence: {tier['evidence']}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#444444",
            zorder=4,
            multialignment="center",
        )

    # L4 exclusion note
    ax_tiers.text(
        0.50,
        0.03,
        "L4 (phenomenal) deliberately excluded from this architecture (§8.4)",
        ha="center",
        fontsize=7,
        color="#888888",
        style="italic",
    )

    ax_tiers.set_title(
        "Three-Tier Epistemic Architecture", fontsize=9, fontweight="bold"
    )

    # ── Valid bridges ─────────────────────────────────────────────────────
    for br in BRIDGES:
        offset = br.get("offset_x", 0.0)
        x0 = 0.60 + offset
        ax_bridges.annotate(
            "",
            xy=(x0, br["y_to"] + 0.10),
            xytext=(x0, br["y_from"]),
            arrowprops=dict(
                arrowstyle="->",
                color=br["color"],
                lw=2.0,
                connectionstyle="arc3,rad=0.15" if offset != 0 else "arc3,rad=0.0",
            ),
            zorder=5,
        )
        mid_y = (br["y_from"] + br["y_to"] + 0.10) / 2
        ax_bridges.text(
            x0 + 0.08 + offset,
            mid_y,
            br["label"],
            ha="left",
            va="center",
            fontsize=6.5,
            color=br["color"],
            multialignment="left",
        )

    # Absent bridges (dashed + X)
    for ab in ABSENT_BRIDGES:
        ax_bridges.annotate(
            "",
            xy=(0.30, ab["y_to"] + 0.10),
            xytext=(0.30, ab["y_from"]),
            arrowprops=dict(
                arrowstyle="->",
                color="#cc0000",
                lw=1.5,
                linestyle="dashed",
            ),
            zorder=5,
        )
        mid_y = (ab["y_from"] + ab["y_to"] + 0.10) / 2
        ax_bridges.text(
            0.20, mid_y, "✗", ha="center", fontsize=14, color="#cc0000", zorder=6
        )
        ax_bridges.text(
            0.08,
            mid_y,
            ab["label"],
            ha="center",
            fontsize=6.5,
            color="#cc0000",
            multialignment="center",
            style="italic",
        )

    ax_bridges.text(
        0.50,
        0.96,
        "Valid bridges (solid)\nAbsent bridges (dashed ✗)",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
    )

    # ── Four-step double-bridge calculation ───────────────────────────────
    ax_calc.text(
        0.50,
        0.94,
        "Double-bridge\ncalculation",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#555555",
    )
    for i, step in enumerate(CALC_STEPS):
        y = 0.80 - i * 0.17
        circ = mpatches.Circle(
            (0.13, y), 0.05, facecolor="#333333", edgecolor="white", lw=1.0, zorder=3
        )
        ax_calc.add_patch(circ)
        ax_calc.text(
            0.13,
            y,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        ax_calc.text(
            0.22,
            y,
            step,
            ha="left",
            va="center",
            fontsize=7.5,
            color="#333333",
            multialignment="left",
        )
        if i < len(CALC_STEPS) - 1:
            ax_calc.annotate(
                "",
                xy=(0.13, y - 0.07),
                xytext=(0.13, y - 0.02),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
            )

    fig.suptitle(
        "Figure 1 — Three-Tier Epistemic Architecture with Canonical Bridge Principles\n"
        "(Paper 4, §3.2)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig1_three_tier_epistemic_architecture.pdf")
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
