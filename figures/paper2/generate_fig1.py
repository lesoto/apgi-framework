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
        "comp": r"Prediction error separation" + "\n" + r"($\varepsilon^e$, $\varepsilon^i$)",
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

ARCHITECTURES = ["Feedforward", "Attractor\nnetwork", "Transformer", "LSM/LNN\n(APGI)"]
# Number of constraints satisfied out of 4
ARCH_SCORES = [2, 3, 2, 4]


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
    ax_main.text(0.20, 0.95, "Biological Constraints", ha="center", fontsize=9,
                 fontweight="bold", color="#333333")
    ax_main.text(0.75, 0.95, "APGI Computational\nRequirements", ha="center",
                 fontsize=9, fontweight="bold", color="#333333")

    # LSM centre label
    ax_main.text(0.49, 0.50, "LSM /\nLNN", ha="center", va="center",
                 fontsize=10, fontweight="bold", color="white", zorder=10)
    circ = mpatches.Circle((0.49, 0.50), 0.065, facecolor="#333333",
                            edgecolor="white", lw=2.0, zorder=9)
    ax_main.add_patch(circ)

    ROW_Y = [0.78, 0.58, 0.38, 0.18]
    BOX_W, BOX_H = 0.30, 0.14

    for i, (pair, y) in enumerate(zip(PAIRS, ROW_Y)):
        color = pair["color"]

        # Bio box (left)
        rect_l = mpatches.FancyBboxPatch(
            (0.04, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.01", linewidth=1.5,
            edgecolor=color, facecolor="#f7f7f7", zorder=3,
        )
        ax_main.add_patch(rect_l)
        ax_main.text(0.04 + BOX_W / 2, y, pair["bio"], ha="center", va="center",
                     fontsize=8, color="#222222", zorder=4, multialignment="center")

        # Comp box (right)
        rect_r = mpatches.FancyBboxPatch(
            (0.64, y - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.01", linewidth=1.5,
            edgecolor=color, facecolor="#f7f7f7", zorder=3,
        )
        ax_main.add_patch(rect_r)
        ax_main.text(0.64 + BOX_W / 2, y, pair["comp"], ha="center", va="center",
                     fontsize=8, color="#222222", zorder=4, multialignment="center")

        # Arrows bio → LSM → comp
        ax_main.annotate("", xy=(0.425, y), xytext=(0.34, y),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=5)
        ax_main.annotate("", xy=(0.64, y), xytext=(0.555, y),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=5)

        # Number label
        ax_main.text(0.01, y, f"({i+1})", ha="right", va="center",
                     fontsize=9, color=color, fontweight="bold")

    ax_main.set_title(
        "Biological Constraints ↔ APGI Computational Requirements\n"
        "(LSM/LNN as unique simultaneous satisfier)",
        fontsize=10, fontweight="bold",
    )

    # ── Comparison panel ──────────────────────────────────────────────────
    y_pos = np.arange(len(ARCHITECTURES))
    colors = ["#aaaaaa", "#aaaaaa", "#aaaaaa", "#2166ac"]
    bars = ax_comp.barh(y_pos, ARCH_SCORES, color=colors, edgecolor="#333333",
                        lw=1.2, height=0.55)
    ax_comp.set_yticks(y_pos)
    ax_comp.set_yticklabels(ARCHITECTURES, fontsize=9)
    ax_comp.set_xlim(0, 4.6)
    ax_comp.set_xlabel("Constraints satisfied (out of 4)", fontsize=9)
    ax_comp.axvline(4, lw=1.5, ls="--", color="#333333")
    ax_comp.text(4.05, -0.4, "All 4", fontsize=8, color="#333333")
    for bar, v in zip(bars, ARCH_SCORES):
        ax_comp.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                     str(v), va="center", fontsize=10, fontweight="bold",
                     color="#2166ac" if v == 4 else "#555555")
    ax_comp.set_title("Architecture comparison", fontsize=9, fontweight="bold")
    ax_comp.spines["top"].set_visible(False)
    ax_comp.spines["right"].set_visible(False)

    label_axes([ax_main, ax_comp])
    fig.suptitle(
        "Figure 1 — Four Biological Constraints × Four APGI Computational Requirements: The LSM Substrate\n"
        "(Paper 2, §1)",
        fontsize=11, fontweight="bold", y=1.01,
    )
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
