"""Paper 3 — Figure S1: Protocol 6 Clinical Study Design (supplementary).

Panel A: 2D scatter schematic (PCI vs HEP) with predicted group clusters.
Panel B: Longitudinal assessment timeline for four groups.

Run:
    python figures/paper3/generate_figS1.py
    python figures/paper3/generate_figS1.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import label_axes, save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

RNG = np.random.default_rng(99)

GROUPS = [
    {"name": "VS/UWS",   "N": 30, "pci": 0.15, "hep": 0.20, "color": "#a50f15"},
    {"name": "MCS",      "N": 30, "pci": 0.40, "hep": 0.45, "color": "#fc8d59"},
    {"name": "EMCS",     "N": 20, "pci": 0.65, "hep": 0.68, "color": "#2166ac"},
    {"name": "Controls", "N": 30, "pci": 0.85, "hep": 0.88, "color": "#4dac26"},
]

TIMEPOINTS = [
    {"label": "Baseline\n(wk 0)",    "x": 0.15},
    {"label": "Follow-up 1\n(3 mo)", "x": 0.50},
    {"label": "Follow-up 2\n(6 mo)", "x": 0.85},
]
ASSESSMENTS = ["EEG/HEP", "TMS-EEG/PCI", "CRS-R"]


def draw_scatter(ax):
    for g in GROUPS:
        pts = RNG.multivariate_normal(
            [g["pci"], g["hep"]],
            [[0.015, 0.010], [0.010, 0.015]],
            size=g["N"],
        )
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=g["color"],
                   alpha=0.6, label=f"{g['name']} (N={g['N']})", zorder=4)
        # Group centroid
        ax.plot(g["pci"], g["hep"], "*", ms=12, color=g["color"],
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    # Decision boundary (linear diagonal)
    x_dec = np.linspace(0, 1, 200)
    ax.plot(x_dec, x_dec * 0.95 + 0.05, "k--", lw=1.5, alpha=0.6,
            label="Joint AUC ≥ 0.80 decision boundary")

    ax.set_xlabel("PCI (perturbational complexity index)", fontsize=10)
    ax.set_ylabel("HEP amplitude (norm.)", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("A — Predicted group clusters\n(PCI × HEP joint biomarker)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.50, -0.04,
            "Deeper substrates including thalamic and claustral contributions are not directly testable\n"
            "by transcranial stimulation; EP-2 constrains only the cortically accessible nodes of the gating network. (§5.3)",
            ha="center", fontsize=5.5, color="#888888", style="italic", transform=ax.transAxes)


def draw_timeline(ax):
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.1, 1.0)
    ax.axis("off")
    ax.set_title("B — Longitudinal assessment timeline", fontsize=9, fontweight="bold")

    # x positions for timepoints
    tp_x = [tp["x"] for tp in TIMEPOINTS]

    # Group rows
    ROW_Y = {g["name"]: 0.75 - i * 0.20 for i, g in enumerate(GROUPS)}

    # Header
    for tp in TIMEPOINTS:
        ax.text(tp["x"], 0.96, tp["label"], ha="center", va="top", fontsize=8,
                fontweight="bold", color="#333333")
        ax.axvline(tp["x"], ymin=0.02, ymax=0.94, lw=0.8, color="#dddddd", ls="--", zorder=1)

    for g in GROUPS:
        y = ROW_Y[g["name"]]
        ax.text(0.02, y, f"{g['name']}\n(N={g['N']})", ha="left", va="center",
                fontsize=8, color=g["color"], fontweight="bold")

        for x in tp_x:
            circ = mpatches.Circle((x, y), 0.025, facecolor=g["color"],
                                   edgecolor="white", lw=1.0, alpha=0.85, zorder=3)
            ax.add_patch(circ)
            # Assessments
            for j, assess in enumerate(ASSESSMENTS):
                ax.text(x, y - 0.045 - j * 0.028, f"• {assess}",
                        ha="center", fontsize=5.5, color="#555555")

        # Horizontal connector
        ax.plot(tp_x, [y] * len(tp_x), lw=1.2, color=g["color"], alpha=0.40, zorder=2)

    # Assessment legend
    ax.text(0.50, 0.05, "Assessments at each timepoint: " + " | ".join(ASSESSMENTS),
            ha="center", fontsize=7, color="#555555", style="italic")


def plot(show: bool = True) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_scatter(ax1)
    draw_timeline(ax2)
    label_axes([ax1, ax2])
    fig.suptitle(
        "Figure S1 — Protocol 6 Clinical Study Design: Patient Groups and Assessment Timeline\n"
        "(Paper 3, supplementary; cross-referenced from §5.2)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figS1_protocol6_clinical_design.pdf")
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
