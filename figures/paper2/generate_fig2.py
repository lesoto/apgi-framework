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

# Arrowhead key (used throughout Panel A): three connection types per the
# Fig. 2 prompt ("Three arrowhead types for excitatory/inhibitory/modulatory").
_EXC, _INH, _MOD = "#2166ac", "#d6604d", "#8c8c00"


def _arrow(ax, xy, xytext, kind, color=None, lw=1.4):
    """Draw one of the three canonical arrow types (excitatory/inhibitory/modulatory)."""
    style = {
        "exc": dict(arrowstyle="-|>", color=color or _EXC, lw=lw, linestyle="solid"),
        "inh": dict(arrowstyle="-|>", color=color or _INH, lw=lw, linestyle=(0, (3, 2))),
        "mod": dict(arrowstyle="-|>", color=color or _MOD, lw=lw, linestyle=(0, (1, 1.5))),
    }[kind]
    ax.annotate("", xy=xy, xytext=xytext, arrowprops=style, zorder=6)


def draw_laminar_column(ax):
    ax.set_xlim(0, 1.42)
    ax.set_ylim(0.02, 1.14)
    ax.axis("off")
    ax.set_title(
        "A — Laminar column\n(receptor/afferent assignments)",
        fontsize=9,
        fontweight="bold",
    )

    for name, y in LAYER_Y.items():
        rect = mpatches.FancyBboxPatch(
            (0.15, y),
            0.42,
            LAYER_H,
            boxstyle="square,pad=0.01",
            facecolor=LAYER_COLORS[name],
            edgecolor="#555555",
            lw=1.0,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(0.11, y + LAYER_H / 2, name, ha="right", va="center", fontsize=8)

    def cell(x, y, label, facecolor, r=0.032):
        circ = mpatches.Circle(
            (x, y), r, facecolor=facecolor, edgecolor="#333333", lw=1.1, zorder=5
        )
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", fontsize=5.3, zorder=6)

    def receptor_text(x, y, text, color):
        ax.text(x, y, text, ha="left", va="center", fontsize=6.2, color=color, style="italic")

    # ── L1: apical dendrites, interoceptive (visceral) afferents ───────────
    y1 = LAYER_Y["L1"] + LAYER_H / 2
    cell(0.57, y1, "Apical\ndendrites", "#f4cccc", r=0.045)
    _arrow(ax, (0.545, y1 + 0.035), (0.42, y1 + 0.075), "mod", color="#c0392b")
    ax.text(0.40, y1 + 0.09, "Interoceptive\n(visceral afferents)", ha="center",
            va="bottom", fontsize=6.0, color="#c0392b", style="italic")
    receptor_text(0.63, y1, "5-HT$_{2A}$, HCN,\nmGluR$_5$", "#c0392b")

    # ── L2/3: PV+ and VIP+ interneurons ─────────────────────────────────────
    # CRITICAL: the ONLY inputs to the VIP+ node below are (1) the
    # vmPFC -> VIP+ projection and (2) ACh/M1 modulation -- no other
    # node/arrow is connected to it.
    y23 = LAYER_Y["L2/3"] + LAYER_H / 2
    cell(0.50, y23 + 0.035, "PV+", "#a6cee3")
    cell(0.50, y23 - 0.035, "VIP+", "#fdd835")

    # vmPFC source node (top-down, distal) -> VIP+ only.
    vmpfc_xy = (0.30, y23 + 0.16)
    ax.add_patch(mpatches.FancyBboxPatch(
        (vmpfc_xy[0] - 0.075, vmpfc_xy[1] - 0.028), 0.15, 0.056,
        boxstyle="round,pad=0.01", facecolor="#cce5ff", edgecolor=_EXC, lw=1.2, zorder=5,
    ))
    ax.text(vmpfc_xy[0], vmpfc_xy[1], "vmPFC", ha="center", va="center", fontsize=6.5,
            fontweight="bold", color="#0b3d78", zorder=6)
    _arrow(ax, (0.485, y23 - 0.01), (vmpfc_xy[0] + 0.02, vmpfc_xy[1] - 0.03), "exc")

    # ACh/M1 modulatory source -> VIP+ only (second and last input to VIP+).
    ach_xy = (0.66, y23 - 0.06)
    ax.text(ach_xy[0], ach_xy[1], "ACh / M1", ha="left", va="center", fontsize=6.3,
            color=_MOD, fontweight="bold", zorder=6)
    _arrow(ax, (0.525, y23 - 0.035), (ach_xy[0] - 0.01, ach_xy[1]), "mod")

    # ACh/M1 also modulates PV+ (a separate arrow -- does not touch VIP+).
    _arrow(ax, (0.525, y23 + 0.035), (ach_xy[0] - 0.01, ach_xy[1] + 0.09), "mod")
    ax.text(ach_xy[0], ach_xy[1] + 0.09, "ACh / M1", ha="left", va="center", fontsize=6.3,
            color=_MOD, fontweight="bold", zorder=6)

    receptor_text(0.63, y23 + 0.035 + 0.045, "nAChR ($\\alpha7,\\beta2$)", "#2166ac")
    receptor_text(0.94, y23 - 0.035, "mGluR$_{2/3}$, D$_1$\n(top-down gating)", "#8c8c00")

    # ── L4: spiny stellate, exteroceptive thalamic drive ────────────────────
    y4 = LAYER_Y["L4"] + LAYER_H / 2
    cell(0.50, y4, "L4 spiny\nstellate", "#9ecae1", r=0.042)
    _arrow(ax, (0.465, y4), (0.34, y4), "exc")
    ax.text(0.30, y4, "Exteroceptive\n(thalamus)", ha="right", va="center",
            fontsize=6.2, color=_EXC, style="italic")
    receptor_text(0.63, y4, "AMPA, NMDA,\nmGluR$_5$", "#4dac26")

    # ── L5: SST+ interneuron and thick-tufted pyramidal ─────────────────────
    y5 = LAYER_Y["L5"] + LAYER_H / 2
    cell(0.50, y5 + 0.035, "SST+", "#fdb863")
    cell(0.50, y5 - 0.035, "L5 thick-\ntufted", "#8073ac", r=0.038)
    ne_xy = (0.30, y5 + 0.045)
    ax.text(ne_xy[0], ne_xy[1], "NE / LC", ha="center", va="center", fontsize=6.3,
            color="#a6611a", fontweight="bold", zorder=6)
    _arrow(ax, (0.475, y5 + 0.035), (ne_xy[0] + 0.03, ne_xy[1] + 0.005), "mod", color="#a6611a")
    receptor_text(0.63, y5 + 0.035, r"$\alpha_{2A}$-AR, GABA$_A$", "#d6604d")
    receptor_text(0.63, y5 - 0.035, r"$\alpha_2$-AR, D$_1$" + "\n(NE-gated output)", "#7b3294")

    # ── L6: pyramidal cells, thalamic feedback ──────────────────────────────
    y6 = LAYER_Y["L6"] + LAYER_H / 2
    cell(0.50, y6, "L6\npyramidal", "#9ecae1", r=0.038)
    _arrow(ax, (0.465, y6), (0.34, y6), "exc", color="#41ab5d")
    ax.text(0.30, y6, "Thalamic\nfeedback", ha="right", va="center",
            fontsize=6.2, color="#41ab5d", style="italic")
    receptor_text(0.63, y6, "AMPA, NMDA, mGluR$_5$\n(thalamic feedback/coord.)", "#41ab5d")

    # ── Arrowhead key (excitatory / inhibitory / modulatory) ────────────────
    kx0, ky0, dky = 1.00, 1.10, 0.055
    for i, (kind, text, color) in enumerate([
        ("exc", "Excitatory (glutamatergic)", _EXC),
        ("inh", "Inhibitory (GABAergic)", _INH),
        ("mod", "Modulatory (neuromodulator/receptor)", _MOD),
    ]):
        yk = ky0 - i * dky
        _arrow(ax, (kx0 + 0.09, yk), (kx0, yk), kind, color=color, lw=1.3)
        ax.text(kx0 + 0.12, yk, text, ha="left", va="center", fontsize=5.6, color=color)

    ax.text(
        0.40,
        0.055,
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
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.8))
    draw_laminar_column(axes[0])
    draw_pathway_panel(axes[1])
    draw_pyramidal_panel(axes[2])
    label_axes(list(axes))

    # Abbreviation key (figure-level, per prompt: "Include an abbreviation key").
    abbrev = (
        "Abbreviation key —  vmPFC, ventromedial prefrontal cortex;  PV$^+$/SST$^+$/VIP$^+$, "
        "parvalbumin/somatostatin/vasoactive intestinal peptide-positive interneurons;  "
        "ACh, acetylcholine;  NE, norepinephrine;  $\\hat{M}(c,a)$, somatic-marker estimate;  "
        r"$\varepsilon^e/\varepsilon^i$, exteroceptive/interoceptive prediction error."
    )
    fig.text(0.5, -0.045, abbrev, ha="center", va="top", fontsize=7.2, color="#444444")

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
