"""Paper 2 — Figure 1: Four Biological Constraints × Four APGI Computational Requirements.

Four-quadrant schematic with one-to-one mapping, LSM substrate at centre,
and comparison panel showing alternative architectures as partial satisfiers.

Run:
    python figures/paper2/generate_fig1.py
    python figures/paper2/generate_fig1.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import label_axes, save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

PAIRS = [
    {
        "bio": "High-dimensional\ntransient dynamics",
        "comp": r"Prediction error separation"
        + "\n"
        + r"($\varepsilon^e$, $\varepsilon^i$)",
        "color": "#2166ac",
    },
    {
        "bio": "Fading memory\n(~200–500 ms)",
        "comp": r"Accumulation window" + "\n" + r"($\tau_S \approx 200{-}500$ ms)",
        "color": "#4dac26",
    },
    {
        "bio": "Nonlinear\nseparability",
        "comp": "Ignition probability\nsigmoid",
        "color": "#d6604d",
    },
    {
        "bio": "Sparse event-\ndriven activity",
        "comp": "Metabolic constraint /\nallostatic threshold",
        "color": "#7b3294",
    },
]

# Bottom-to-top order for barh (matplotlib plots index 0 at the bottom), chosen
# so the rendered top-to-bottom order matches the PDF spec exactly: LSM/LNN,
# Attractor network, Transformer, Feedforward.
ARCHITECTURES = ["Feedforward", "Transformer", "Attractor\nnetwork", "LSM/LNN\n(APGI)"]
# Number of constraints satisfied out of 4
ARCH_SCORES = [2, 2, 3, 4]
# Constraint number(s) each alternative architecture FAILS (1-indexed, matching
# PAIRS / the quadrant numbering). Supports the §1 mechanistic argument, not just
# the satisfied-count claim.
#   C1 high-dimensional transient dynamics, C2 fading memory,
#   C3 nonlinear separability, C4 sparse event-driven activity
ARCH_FAILS = [[1, 2], [2, 4], [2], []]


def plot(show: bool = True) -> None:
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
    ax_main = fig.add_subplot(gs[0])
    ax_comp = fig.add_subplot(gs[1])

    # ── Main four-quadrant panel ───────────────────────────────────────────
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis("off")

    # Column headers
    ax_main.text(
        0.20,
        0.95,
        "Biological Constraints",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#333333",
    )
    ax_main.text(
        0.75,
        0.95,
        "APGI Computational\nRequirements",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#333333",
    )

    # LSM centre label
    ax_main.text(
        0.49,
        0.50,
        "LSM /\nLNN",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
        zorder=10,
    )
    circ = mpatches.Circle(
        (0.49, 0.50), 0.065, facecolor="#333333", edgecolor="white", lw=2.0, zorder=9
    )
    ax_main.add_patch(circ)

    ROW_Y = [0.78, 0.58, 0.38, 0.18]
    BOX_W, BOX_H = 0.30, 0.14

    for i, (pair, y) in enumerate(zip(PAIRS, ROW_Y)):
        color = pair["color"]

        # Bio box (left)
        rect_l = mpatches.FancyBboxPatch(
            (0.04, y - BOX_H / 2),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0.01",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#f7f7f7",
            zorder=3,
        )
        ax_main.add_patch(rect_l)
        ax_main.text(
            0.04 + BOX_W / 2,
            y,
            pair["bio"],
            ha="center",
            va="center",
            fontsize=8,
            color="#222222",
            zorder=4,
            multialignment="center",
        )

        # Comp box (right)
        rect_r = mpatches.FancyBboxPatch(
            (0.64, y - BOX_H / 2),
            BOX_W,
            BOX_H,
            boxstyle="round,pad=0.01",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#f7f7f7",
            zorder=3,
        )
        ax_main.add_patch(rect_r)
        ax_main.text(
            0.64 + BOX_W / 2,
            y,
            pair["comp"],
            ha="center",
            va="center",
            fontsize=8,
            color="#222222",
            zorder=4,
            multialignment="center",
        )

        # Arrows bio → LSM → comp
        ax_main.annotate(
            "",
            xy=(0.425, y),
            xytext=(0.34, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            zorder=5,
        )
        ax_main.annotate(
            "",
            xy=(0.64, y),
            xytext=(0.555, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            zorder=5,
        )

        # Number label
        ax_main.text(
            0.01,
            y,
            f"({i+1})",
            ha="right",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax_main.set_title(
        "Biological Constraints ↔ APGI Computational Requirements\n"
        "(LSM/LNN as unique simultaneous satisfier)",
        fontsize=10,
        fontweight="bold",
    )

    # ── Comparison panel ──────────────────────────────────────────────────
    y_pos = np.arange(len(ARCHITECTURES))
    colors = ["#aaaaaa", "#aaaaaa", "#aaaaaa", "#2166ac"]
    bars = ax_comp.barh(
        y_pos, ARCH_SCORES, color=colors, edgecolor="#333333", lw=1.2, height=0.55
    )
    ax_comp.set_yticks(y_pos)
    ax_comp.set_yticklabels(ARCHITECTURES, fontsize=9)
    ax_comp.set_xlim(0, 4.6)
    ax_comp.set_xlabel("Constraints satisfied (out of 4)", fontsize=9)
    ax_comp.axvline(4, lw=1.5, ls="--", color="#333333")
    ax_comp.text(4.05, -0.4, "All 4", fontsize=8, color="#333333")
    constraint_colors = [p["color"] for p in PAIRS]
    for bar, v, fails in zip(bars, ARCH_SCORES, ARCH_FAILS):
        yc = bar.get_y() + bar.get_height() / 2
        ax_comp.text(
            v + 0.05,
            yc,
            str(v),
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#2166ac" if v == 4 else "#555555",
        )
        # Identify the specific failed constraint(s), coloured to match the
        # quadrant numbering so the failure pattern (not just the count) is read.
        if fails:
            ax_comp.text(
                v + 0.28,
                yc + 0.22,
                "fails",
                va="center",
                ha="left",
                fontsize=6.5,
                color="#888888",
                style="italic",
            )
            x = v + 0.28
            for k, c in enumerate(fails):
                lbl = f"C{c}" + ("," if k < len(fails) - 1 else "")
                ax_comp.text(
                    x,
                    yc - 0.05,
                    lbl,
                    va="center",
                    ha="left",
                    fontsize=8,
                    fontweight="bold",
                    color=constraint_colors[c - 1],
                )
                x += 0.42
        else:
            ax_comp.text(
                v + 0.28,
                yc,
                "(all 4)",
                va="center",
                ha="left",
                fontsize=7,
                color="#2166ac",
                fontweight="bold",
            )
    ax_comp.set_title("Architecture comparison", fontsize=9, fontweight="bold")
    ax_comp.spines["top"].set_visible(False)
    ax_comp.spines["right"].set_visible(False)

    # Disambiguation: constraint colours are not epistemic-tier colours.
    ax_comp.text(
        0.0,
        -0.16,
        "Constraint colours C1–C4 denote biological constraints, not epistemic tiers.",
        transform=ax_comp.transAxes,
        fontsize=6.3,
        color="#777777",
        style="italic",
    )

    label_axes([ax_main, ax_comp])

    # Abbreviation key (per prompt: "Include abbreviation key").
    abbrev = (
        "Abbreviation key —  LSM, liquid state machine;  LNN, liquid neural network;  "
        "APGI, Allostatic Precision-Gated Ignition;  "
        r"$\varepsilon^e/\varepsilon^i$, excitatory/inhibitory prediction errors;  "
        r"$\tau_S$, decay/accumulation time constant."
    )
    fig.text(0.5, -0.03, abbrev, ha="center", va="top", fontsize=8, color="#444444")

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig1_biological_constraints_lsm.pdf")
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
