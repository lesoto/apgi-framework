"""Paper 2 — Figure 2: Cortical Microcircuit — Precision Implementation & Somatic-Marker Pathway.

Three panels:
  A — Laminar column with receptor assignments
  B — vmPFC→insula→VIP+→SST+→L2/3 pyramid (baseline vs. active states)
  C — L5 thick-tufted pyramidal apical/basal mismatch

Run:
    python figures/paper2/generate_fig2.py
    python figures/paper2/generate_fig2.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import label_axes, save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

LAYER_COLORS = {
    "L1": "#e5f5e0",
    "L2/3": "#c7e9c0",
    "L4": "#a1d99b",
    "L5": "#74c476",
    "L6": "#41ab5d",
}
LAYER_H = 0.13
LAYER_Y = {"L1": 0.83, "L2/3": 0.68, "L4": 0.53, "L5": 0.38, "L6": 0.23}


def draw_laminar_column(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)
    ax.axis("off")
    ax.set_title(
        "A — Laminar column\n(receptor assignments)", fontsize=9, fontweight="bold"
    )

    for name, y in LAYER_Y.items():
        rect = mpatches.FancyBboxPatch(
            (0.15, y),
            0.5,
            LAYER_H,
            boxstyle="square,pad=0.01",
            facecolor=LAYER_COLORS[name],
            edgecolor="#555555",
            lw=1.0,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(0.11, y + LAYER_H / 2, name, ha="right", va="center", fontsize=8)

    # Receptor annotations
    annotations = [
        ("L1", "Interoceptive input\n(apical dendrites)", 0.80, "#7b3294"),
        ("L2/3", "ACh/M1 → PV+\nVIP+ ← vmPFC", 0.80, "#2166ac"),
        ("L4", "Exteroceptive input\n(thalamus → L4)", 0.80, "#4dac26"),
        ("L5", "NE/α2 → SST+\nL5 thick-tufted", 0.80, "#d6604d"),
        ("L6", "Thalamic feedback", 0.80, "#888888"),
    ]
    for layer, text, x, color in annotations:
        y = LAYER_Y[layer]
        ax.text(
            x,
            y + LAYER_H / 2,
            text,
            ha="left",
            va="center",
            fontsize=6.5,
            color=color,
            style="italic",
        )
        ax.plot(
            [0.65, x - 0.02],
            [y + LAYER_H / 2, y + LAYER_H / 2],
            lw=0.8,
            color=color,
            alpha=0.5,
        )

    ax.text(
        0.40,
        0.16,
        "Proposed circuit — causal dissociation requires\n"
        "optogenetic/chemogenetic validation (§6.2, Gap 1)",
        ha="center",
        va="top",
        fontsize=6,
        color="#888888",
        style="italic",
    )


def draw_pathway_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "B — vmPFC→VIP+→SST+ disinhibition\n(baseline vs. active)",
        fontsize=9,
        fontweight="bold",
    )

    nodes_base = [
        (0.20, 0.82, "vmPFC", "#cce5ff", 0.3),
        (0.20, 0.60, "VIP+", "#fff3cd", 0.2),
        (0.20, 0.38, "SST+", "#f8d7da", 0.8),
        (0.70, 0.50, "L2/3\npyramid", "#aaaaaa", 0.5),
    ]
    nodes_act = [
        (0.60, 0.82, "vmPFC", "#cce5ff", 0.95),
        (0.60, 0.60, "VIP+", "#fff3cd", 0.90),
        (0.60, 0.38, "SST+", "#f8d7da", 0.15),
        (0.90, 0.50, "L2/3\npyramid", "#4dac26", 0.90),
    ]

    for group, label, offset_x in [
        (nodes_base, "Baseline\n(β_SM low)", 0.05),
        (nodes_act, "Active\n(β_SM high)", 0.55),
    ]:
        ax.text(
            offset_x + 0.15,
            0.96,
            label,
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#333333",
        )
        for x, y, name, color, alpha in group:
            circ = mpatches.Circle(
                (x, y),
                0.07,
                facecolor=color,
                edgecolor="#333333",
                lw=1.2,
                alpha=alpha,
                zorder=3,
            )
            ax.add_patch(circ)
            ax.text(
                x,
                y,
                name,
                ha="center",
                va="center",
                fontsize=6.5,
                zorder=4,
                multialignment="center",
                alpha=min(alpha + 0.3, 1.0),
            )

    # Arrows for active state
    _EX, _IN, _DIS = "#2166ac", "#d6604d", "#4dac26"
    ax.annotate(
        "",
        xy=(0.60, 0.75),
        xytext=(0.60, 0.67),
        arrowprops=dict(arrowstyle="->", color=_EX, lw=1.5),
    )
    ax.annotate(
        "",
        xy=(0.60, 0.45),
        xytext=(0.60, 0.53),
        arrowprops=dict(arrowstyle="->", color=_IN, lw=1.5, linestyle="dashed"),
    )
    ax.annotate(
        "",
        xy=(0.83, 0.50),
        xytext=(0.67, 0.50),
        arrowprops=dict(arrowstyle="->", color=_DIS, lw=2.0),
    )

    # Canonical somatic-marker precision modulation (src/apgi/core.py §4.2):
    # Πⁱ_eff = Πⁱ_baseline · exp(β_SM · M̂(c,a))  — note the marker *estimate* hat.
    pi_eq = (
        r"$\Pi^i_{\mathrm{eff}} = \Pi^i_{\mathrm{baseline}} \cdot "
        r"\exp(\beta_{\mathrm{SM}} \cdot \hat{M}(c,a))$"
    )
    ax.text(0.50, 0.08, pi_eq, ha="center", fontsize=7.5, color="#333333")


def draw_pyramidal_panel(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "C — L5 pyramidal: apical/basal\nmismatch → burst signal",
        fontsize=9,
        fontweight="bold",
    )

    # Cell body
    soma = mpatches.Circle(
        (0.50, 0.35), 0.07, facecolor="#74c476", edgecolor="#333333", lw=1.5, zorder=3
    )
    ax.add_patch(soma)
    ax.text(0.50, 0.35, "L5\nsoma", ha="center", va="center", fontsize=7, zorder=4)

    # Apical dendrite
    ax.annotate(
        "",
        xy=(0.50, 0.82),
        xytext=(0.50, 0.42),
        arrowprops=dict(arrowstyle="-", color="#555555", lw=2.5),
    )
    rect_apic = mpatches.FancyBboxPatch(
        (0.35, 0.82),
        0.30,
        0.10,
        boxstyle="round,pad=0.01",
        facecolor="#cce5ff",
        edgecolor="#2166ac",
        lw=1.5,
    )
    ax.add_patch(rect_apic)
    ax.text(
        0.50,
        0.87,
        "Apical dendrite\n(top-down prediction)",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#2166ac",
    )

    # Basal dendrite
    ax.annotate(
        "",
        xy=(0.50, 0.12),
        xytext=(0.50, 0.28),
        arrowprops=dict(arrowstyle="-", color="#555555", lw=2.5),
    )
    rect_bas = mpatches.FancyBboxPatch(
        (0.35, 0.05),
        0.30,
        0.08,
        boxstyle="round,pad=0.01",
        facecolor="#fdd49e",
        edgecolor="#d6604d",
        lw=1.5,
    )
    ax.add_patch(rect_bas)
    ax.text(
        0.50,
        0.09,
        "Basal (thalamic\nbottom-up)",
        ha="center",
        va="center",
        fontsize=6.5,
        color="#d6604d",
    )

    # NMDA plateau annotation
    ax.annotate(
        "NMDA plateau →\nburst (εᵉ/εⁱ)",
        xy=(0.57, 0.60),
        xytext=(0.78, 0.65),
        fontsize=7,
        color="#7b3294",
        arrowprops=dict(arrowstyle="->", color="#7b3294", lw=1.0),
    )

    ax.text(
        0.50,
        0.00,
        "Thalamic/claustral substrates not\ntranscranially accessible (§5.3)",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color="#888888",
        style="italic",
    )


def plot(show: bool = True) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5))
    draw_laminar_column(axes[0])
    draw_pathway_panel(axes[1])
    draw_pyramidal_panel(axes[2])
    label_axes(list(axes))
    fig.suptitle(
        "Figure 2 — Cortical Microcircuit: Precision Implementation and Somatic-Marker Pathway\n"
        "(Paper 2, §3.1 and §3.3)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig2_cortical_microcircuit.pdf")
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
