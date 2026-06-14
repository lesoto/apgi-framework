"""Paper 1 — Figure 1: APGI Architecture and Five-Stage Processing Pipeline.

Five sequential processing stages with neural substrates, formal variables,
and sigmoid ignition inset panel. Arrow colour encodes epistemic tier.

Run:
    python figures/paper1/generate_fig1.py
    python figures/paper1/generate_fig1.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

TIER_COLORS = {1: "#2166ac", 2: "#4dac26", 3: "#d6604d"}
BOX_COLOR = "#f7f7f7"
BOX_EDGE = "#333333"

STAGES = [
    {
        "label": "Stage 1\nError\nGeneration",
        "variable": r"$\varepsilon^e / \varepsilon^i$",
        "substrate": "L2/3 mismatch",
        "tier": 1,
    },
    {
        "label": "Stage 2\nPrecision\nWeighting",
        "variable": r"$\Pi^e / \Pi^i_{\mathrm{eff}}$",
        "substrate": "PV+/SST+/VIP+ gating",
        "tier": 2,
    },
    {
        "label": "Stage 3\nSomatic-Marker\nAmplification",
        "variable": r"$\beta_{\mathrm{SM}}$ modulation",
        "substrate": "vmPFC–VIP pathway",
        "tier": 2,
    },
    {
        "label": r"Stage 4$\,S_t$" + "\nAccumulation",
        "variable": r"$S_t$",
        "substrate": "L5 pyramidal recruitment",
        "tier": 3,
    },
    {
        "label": "Stage 5\nIgnition Decision\n/ Global Broadcast",
        "variable": r"$P(B_t{=}1)$",
        "substrate": "frontoparietal–\nthalamic workspace",
        "tier": 3,
    },
]


def draw_pipeline(ax: plt.Axes) -> None:
    n = len(STAGES)
    box_w, box_h = 0.13, 0.30
    gap = (1.0 - n * box_w) / (n + 1)

    for i, stage in enumerate(STAGES):
        x = gap + i * (box_w + gap)
        y = 0.35
        color = TIER_COLORS[stage["tier"]]

        rect = mpatches.FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.01",
            linewidth=1.5,
            edgecolor=color,
            facecolor=BOX_COLOR,
            transform=ax.transAxes,
            zorder=3,
        )
        ax.add_patch(rect)

        ax.text(
            x + box_w / 2,
            y + box_h / 2,
            stage["label"],
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=4,
        )

        # Variable annotation below
        ax.text(
            x + box_w / 2,
            y - 0.07,
            stage["variable"],
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            transform=ax.transAxes,
            zorder=4,
        )

        # Neural substrate above
        ax.text(
            x + box_w / 2,
            y + box_h + 0.04,
            stage["substrate"],
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#555555",
            transform=ax.transAxes,
            zorder=4,
            style="italic",
        )

        # Arrow to next stage
        if i < n - 1:
            next_x = gap + (i + 1) * (box_w + gap)
            arrow_color = TIER_COLORS[STAGES[i + 1]["tier"]]
            # Stage 4→5 transition: thicker arrow labelled with the sigmoid function
            is_ignition_transition = i == 3
            lw = 3.0 if is_ignition_transition else 2.0
            ax.annotate(
                "",
                xy=(next_x, y + box_h / 2),
                xytext=(x + box_w, y + box_h / 2),
                xycoords="axes fraction",
                textcoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    lw=lw,
                ),
                zorder=5,
            )
            if is_ignition_transition:
                # Label the sigmoid transition on the arrow
                mid_x = (x + box_w + next_x) / 2
                ax.text(
                    mid_x,
                    y + box_h / 2 + 0.05,
                    r"$\sigma(S_t, \theta_t)$",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=arrow_color,
                    fontweight="bold",
                    transform=ax.transAxes,
                    zorder=6,
                )
                ax.text(
                    mid_x,
                    y + box_h / 2 - 0.05,
                    "see inset →",
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    color="#777777",
                    style="italic",
                    transform=ax.transAxes,
                    zorder=6,
                )

    # Stage labels: (a)(b)(c)
    ax.text(
        0.5,
        0.03,
        "(a) Neural substrate  |  (b) APGI variable  |  (c) Sigmoid inset → right panel",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#777777",
        transform=ax.transAxes,
    )


def draw_sigmoid_inset(ax: plt.Axes) -> None:
    S = np.linspace(-1, 3, 300)
    theta = 1.2
    gamma = 5.0
    P = 1.0 / (1.0 + np.exp(-gamma * (S - theta)))

    ax.plot(S, P, lw=2, color=TIER_COLORS[3])
    ax.axvline(theta, lw=1.2, linestyle="--", color="#555555")
    ax.fill_betweenx([0, 1], theta, 3, alpha=0.12, color=TIER_COLORS[3])
    ax.annotate(
        r"$\theta_t$",
        xy=(theta, 0.5),
        xytext=(theta + 0.3, 0.45),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    ax.set_xlabel(r"$S_t$", fontsize=10)
    ax.set_ylabel(r"$P(\mathrm{ignition}\,|\,S_t, \theta_t)$", fontsize=9)
    ax.set_title("Ignition sigmoid", fontsize=9, style="italic")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, 2.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(show: bool = True) -> None:
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 5, width_ratios=[3, 0.05, 0.05, 0.05, 1.4])
    ax_main = fig.add_subplot(gs[0])
    ax_sig = fig.add_subplot(gs[4])

    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis("off")

    draw_pipeline(ax_main)

    # Tier legend
    for tier, color in TIER_COLORS.items():
        ax_main.plot([], [], color=color, lw=3, label=f"Tier {tier}")
    ax_main.legend(
        loc="lower right",
        fontsize=8,
        title="Epistemic Tier",
        title_fontsize=8,
        framealpha=0.8,
    )

    ax_main.set_title(
        "Figure 1 — APGI Architecture: Five-Stage Processing Pipeline",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )

    draw_sigmoid_inset(ax_sig)

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig1_architecture_pipeline.pdf")
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
