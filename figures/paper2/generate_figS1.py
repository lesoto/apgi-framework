"""Paper 2 — Figure S1: EP-2 and EP-6 Protocol Schematics.

Panel A: Brain-surface rendering with TMS targets (EP-2).
Panel B: iEEG trial timeline for ρ_crit estimation (EP-6).

Run:
    python figures/paper2/generate_figS1.py
    python figures/paper2/generate_figS1.py --no-show
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


def draw_brain_panel(ax):
    """Schematic lateral brain with TMS target markers."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "A — EP-2: TMS Targets\n(lateral brain surface)", fontsize=9, fontweight="bold"
    )

    # Brain outline (simplified ellipse)
    brain = mpatches.Ellipse(
        (0.50, 0.55),
        0.75,
        0.65,
        facecolor="#f5f5dc",
        edgecolor="#888888",
        lw=2.0,
        zorder=2,
    )
    ax.add_patch(brain)

    # Frontal lobe region
    frontal = mpatches.FancyBboxPatch(
        (0.20, 0.58),
        0.22,
        0.22,
        boxstyle="round,pad=0.02",
        facecolor="#ffffcc",
        edgecolor="#888888",
        lw=0.8,
        alpha=0.5,
    )
    ax.add_patch(frontal)
    ax.text(0.31, 0.69, "PFC", ha="center", fontsize=7.5, color="#555555")

    # Target markers
    targets = [
        (0.48, 0.38, "#d6604d", "pIC\n[40,−6,−4]", "right posterior insula"),
        (0.28, 0.72, "#2166ac", "dlPFC\n[−44,36,20]", "left dlPFC"),
        (0.68, 0.68, "#4dac26", "PPC (bilateral)\n[±28,−60,46]", "bilateral PPC"),
        (0.50, 0.88, "#888888", "Vertex\nsham", "vertex sham"),
    ]
    for x, y, color, short, long_name in targets:
        # Bilateral PPC: show a faint contralateral (out-of-plane) homolog marker
        if short.startswith("PPC"):
            contra = mpatches.Circle(
                (x - 0.055, y + 0.03),
                0.032,
                facecolor="none",
                edgecolor=color,
                lw=1.2,
                ls="--",
                alpha=0.7,
                zorder=4,
            )
            ax.add_patch(contra)
            ax.text(
                x - 0.055,
                y + 0.085,
                "contra.",
                ha="center",
                va="bottom",
                fontsize=5.0,
                color=color,
                style="italic",
                zorder=6,
            )
        marker = mpatches.Circle(
            (x, y), 0.04, facecolor=color, edgecolor="white", lw=1.5, zorder=5
        )
        ax.add_patch(marker)
        ax.text(
            x + 0.06,
            y,
            short,
            ha="left",
            va="center",
            fontsize=6.5,
            color=color,
            fontweight="bold",
            zorder=6,
        )

    # Predicted dissociation
    ax.text(
        0.50,
        0.05,
        "pIC → HEP/P3b dissociation\ndlPFC/PPC → global ignition reduction",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#333333",
        style="italic",
    )
    ax.text(
        0.50,
        -0.02,
        "Deeper substrates including thalamic and claustral contributions are not directly testable\n"
        "by transcranial stimulation; EP-2 constrains only the cortically accessible nodes of the gating network. (§5.3)",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#888888",
        style="italic",
    )


def draw_ieeg_panel(ax):
    """EP-6: iEEG trial timeline."""
    ax.set_xlim(-60, 880)
    ax.set_ylim(-0.5, 3.8)
    ax.axis("off")
    ax.set_title(
        "B — EP-6: iEEG Trial Timeline\n(ρ_crit estimation)",
        fontsize=9,
        fontweight="bold",
    )

    BAR_H = 0.55
    # Pre-stimulus AC1 window drawn at its true 500 ms extent (~65% of the
    # pre-stimulus-to-response timeline), as stated in the manuscript text.
    phases = [
        {
            "label": "Pre-stimulus\nAC1 window\n(500 ms)",
            "t": 0,
            "dur": 500,
            "color": "#cce5ff",
            "y": 2.8,
        },
        {
            "label": "Near-threshold\nstimulus",
            "t": 500,
            "dur": 60,
            "color": "#fc9272",
            "y": 2.8,
        },
        {
            "label": "Response\nwindow",
            "t": 560,
            "dur": 200,
            "color": "#fee0d2",
            "y": 2.8,
        },
    ]
    branches = [
        {
            "label": "Supra-threshold:\npropagation to PFC\nhigh-γ 70–150 Hz",
            "t": 560,
            "dur": 250,
            "color": "#2166ac",
            "y": 1.7,
        },
        {
            "label": "Sub-threshold:\nlocal activation only",
            "t": 560,
            "dur": 250,
            "color": "#d6604d",
            "y": 0.5,
        },
    ]
    rho_win = {
        "label": r"$\rho_{\mathrm{crit}}$ estimation" + "\n(pre-stim period)",
        "t": 0,
        "dur": 500,
        "color": "#f2f0fb",
        "y": 0.5,
    }

    for ph in phases + branches:
        rect = mpatches.FancyBboxPatch(
            (ph["t"], ph["y"]),
            ph["dur"],
            BAR_H,
            boxstyle="round,pad=0.04",
            facecolor=ph["color"],
            edgecolor="#555555",
            lw=1.0,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            ph["t"] + ph["dur"] / 2,
            ph["y"] + BAR_H / 2,
            ph["label"],
            ha="center",
            va="center",
            fontsize=6.5,
            zorder=4,
            multialignment="center",
        )

    # ρ_crit window
    rect_rho = mpatches.FancyBboxPatch(
        (rho_win["t"], rho_win["y"]),
        rho_win["dur"],
        BAR_H,
        boxstyle="round,pad=0.04",
        facecolor=rho_win["color"],
        edgecolor="#7b3294",
        lw=1.5,
        linestyle="--",
        zorder=3,
    )
    ax.add_patch(rect_rho)
    ax.text(
        rho_win["t"] + rho_win["dur"] / 2,
        rho_win["y"] + BAR_H / 2,
        rho_win["label"],
        ha="center",
        va="center",
        fontsize=6.5,
        zorder=4,
        color="#7b3294",
        multialignment="center",
    )

    # Branching arrow (at stimulus onset, t = 500 ms)
    ax.annotate(
        "",
        xy=(560, 2.0),
        xytext=(560, 2.5),
        arrowprops=dict(arrowstyle="->", color="#2166ac", lw=1.3),
    )
    ax.annotate(
        "",
        xy=(560, 0.8),
        xytext=(560, 2.5),
        arrowprops=dict(arrowstyle="->", color="#d6604d", lw=1.3),
    )

    # Time axis
    ax.annotate(
        "",
        xy=(840, -0.3),
        xytext=(-10, -0.3),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
    )
    for t, lbl in {
        0: "0",
        100: "100",
        200: "200",
        300: "300",
        400: "400",
        500: "500",
        560: "560",
        760: "760",
        810: "810 ms",
    }.items():
        ax.text(t, -0.40, lbl, ha="center", fontsize=7.0)
    ax.text(420, -0.48, "Time (ms)", ha="center", fontsize=9)


def plot(show: bool = True) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_brain_panel(ax1)
    draw_ieeg_panel(ax2)
    label_axes([ax1, ax2])
    fig.suptitle(
        "Figure S1 — EP-2 (TMS Protocol) and EP-6 (iEEG Protocol) Schematics\n"
        "(Paper 2, supplementary; cross-referenced from §5.3 and §5.4)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figS1_ep2_ep6_protocol_schematics.pdf")
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
