"""Paper 1 — Figure 2: Somatic-Marker Disinhibition Circuit (§4.2).

Schematic microcircuit diagram: vmPFC → insula → VIP+ → SST+ → L2/3 pyramid.
Two inset panels (resting vs. somatic-marker retrieval state).

Run:
    python figures/paper1/generate_fig2.py
    python figures/paper1/generate_fig2.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# Connection-type palette chosen to NOT collide with the series-wide
# Tier 1/2/3 colour convention (blue/red/green). These encode connection
# type, not epistemic tier (see disambiguation note on the main panel).
_EX = "#000000"  # excitatory  — black
_IN = "#888888"  # inhibitory  — grey
_DIS = "#daa520"  # disinhibition — gold


def _draw_node(ax, xy, label, color="#f7f7f7", edgecolor="#333333", r=0.06, fontsize=8):
    circ = mpatches.Circle(
        xy, r, facecolor=color, edgecolor=edgecolor, lw=1.5, zorder=3
    )
    ax.add_patch(circ)
    ax.text(
        xy[0],
        xy[1],
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        zorder=4,
        wrap=True,
        multialignment="center",
    )


def _arrow(ax, src, dst, color, style="->", lw=1.5, label=None, label_offset=(0, 0.04)):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        zorder=5,
    )
    if label:
        mid = (
            (src[0] + dst[0]) / 2 + label_offset[0],
            (src[1] + dst[1]) / 2 + label_offset[1],
        )
        ax.text(mid[0], mid[1], label, fontsize=7, color=color, ha="center")


def draw_main_circuit(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = {
        "vmPFC": (0.12, 0.75),
        "aInsula": (0.35, 0.75),
        "VIP": (0.58, 0.75),
        "SST": (0.58, 0.45),
        "L23pyr": (0.82, 0.45),
    }

    colors = {
        "vmPFC": "#cce5ff",
        "aInsula": "#d4edda",
        "VIP": "#fff3cd",
        "SST": "#f8d7da",
        "L23pyr": "#e2d9f3",
    }
    labels = {
        "vmPFC": "vmPFC",
        "aInsula": "ant.\ninsula",
        "VIP": "VIP+\ninterneuron",
        "SST": "SST+\nMartinotti",
        "L23pyr": "L2/3\npyramidal\n(dendritic)",
    }

    for key, xy in nodes.items():
        _draw_node(ax, xy, labels[key], color=colors[key], r=0.075)

    # vmPFC → aInsula (excitatory)
    _arrow(
        ax,
        (nodes["vmPFC"][0] + 0.075, nodes["vmPFC"][1]),
        (nodes["aInsula"][0] - 0.075, nodes["aInsula"][1]),
        _EX,
        label="direct\nprojection",
        label_offset=(0, 0.06),
    )

    # aInsula → VIP (excitatory)
    _arrow(
        ax,
        (nodes["aInsula"][0] + 0.075, nodes["aInsula"][1]),
        (nodes["VIP"][0] - 0.075, nodes["VIP"][1]),
        _EX,
    )

    # VIP → SST (inhibitory)
    _arrow(
        ax,
        (nodes["VIP"][0], nodes["VIP"][1] - 0.075),
        (nodes["SST"][0], nodes["SST"][1] + 0.075),
        _IN,
        label="inhibition",
        label_offset=(0.06, 0),
    )

    # SST → L23pyr (releases inhibition → disinhibition arrow)
    _arrow(
        ax,
        (nodes["SST"][0] + 0.075, nodes["SST"][1]),
        (nodes["L23pyr"][0] - 0.075, nodes["L23pyr"][1]),
        _DIS,
        style="-|>",
        label="disinhibition\n→ high $\\Pi^i_{\\mathrm{eff}}$",
        label_offset=(0, -0.08),
    )

    # Result annotation
    ax.text(
        nodes["L23pyr"][0],
        nodes["L23pyr"][1] - 0.14,
        r"$\Pi^i_{\mathrm{eff}}$ elevation",
        ha="center",
        va="top",
        fontsize=8,
        color=_DIS,
        fontweight="bold",
    )

    # Legend
    for color, label in [
        (_EX, "Excitatory"),
        (_IN, "Inhibitory"),
        (_DIS, "Disinhibition"),
    ]:
        ax.plot([], [], color=color, lw=2, label=label)
    leg = ax.legend(
        loc="lower left",
        fontsize=7,
        framealpha=0.8,
        title="Connection type (not epistemic tier)",
        title_fontsize=6.5,
    )
    leg.get_title().set_color("#555555")

    ax.set_title(
        "Proposed circuit — vmPFC→insula→VIP+→SST+→L2/3 pyramid\n(convergent anatomical evidence; causal validation pending)",
        fontsize=8.5,
    )

    # Source data & statistics box formula: Pi^i_eff = Pi^i_baseline . exp(beta_SM . M(c,a))
    ax.text(
        0.5,
        0.93,
        r"$\Pi^i_{\mathrm{eff}} = \Pi^i_{\mathrm{baseline}} \cdot \exp(\beta_{\mathrm{SM}} \cdot M(c,a))$",
        ha="center",
        va="top",
        fontsize=8,
        color="#555555",
        style="italic",
    )

    # Caveat
    ax.text(
        0.5,
        0.01,
        "Proposed pathway — causal evidence pending chemogenetic/optogenetic validation (§6.6)",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#888888",
        style="italic",
        transform=ax.transAxes,
    )


def draw_state_inset(ax, title, beta_sm_state):
    """Draw resting (low) or active (high) β_SM state."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    active = beta_sm_state == "high"
    vip_alpha = 0.9 if active else 0.2
    sst_alpha = 0.15 if active else 0.8
    dend_color = _DIS if active else "#aaaaaa"

    items = [
        (0.25, 0.78, "vmPFC", "#cce5ff", 0.9 if active else 0.3),
        (0.25, 0.55, "VIP+", "#fff3cd", vip_alpha),
        (0.25, 0.32, "SST+", "#f8d7da", sst_alpha),
        (0.75, 0.50, "L2/3\npyr.\ndendrite", dend_color, 0.85),
    ]
    for x, y, lbl, col, alpha in items:
        circ = mpatches.Circle(
            (x, y),
            0.10,
            facecolor=col,
            edgecolor="#333333",
            lw=1.2,
            alpha=alpha,
            zorder=3,
        )
        ax.add_patch(circ)
        ax.text(
            x,
            y,
            lbl,
            ha="center",
            va="center",
            fontsize=6.5,
            zorder=4,
            alpha=min(alpha + 0.2, 1.0),
            fontweight="bold",
            multialignment="center",
        )

    # Arrows
    _arrow(ax, (0.25, 0.68), (0.25, 0.65), _EX if active else "#aaaaaa", lw=1.2)
    _arrow(ax, (0.25, 0.45), (0.25, 0.42), _IN if not active else "#aaaaaa", lw=1.2)
    _arrow(ax, (0.35, 0.50), (0.65, 0.50), dend_color, lw=1.5)

    pi_label = (
        r"high $\Pi^i_{\mathrm{eff}}$" if active else r"low $\Pi^i_{\mathrm{eff}}$"
    )
    ax.text(
        0.75,
        0.35,
        pi_label,
        ha="center",
        fontsize=7,
        color=dend_color,
        fontweight="bold",
    )

    bsm_label = (
        r"$\beta_{\mathrm{SM}}$ high" if active else r"$\beta_{\mathrm{SM}}$ low"
    )
    ax.text(0.5, 0.07, bsm_label, ha="center", fontsize=7.5, color="#7b3294")
    ax.set_title(title, fontsize=8, fontweight="bold")


def draw_l5_pyramid(ax):
    """Panel C — L5 thick-tufted pyramidal cell: apical vs. basal dendritic
    mismatch computation with NMDA Ca2+ plateau supporting eps^e / eps^i."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Cortical layer reference bands
    for y0, y1, lbl, col in [
        (0.84, 1.00, "L1", "#f2f2f2"),
        (0.42, 0.84, "L2/3", "#fafafa"),
        (0.08, 0.42, "L5", "#f2f2f2"),
    ]:
        ax.add_patch(
            mpatches.Rectangle(
                (0.0, y0), 1.0, y1 - y0, facecolor=col, edgecolor="none", zorder=0
            )
        )
        ax.text(
            0.03,
            (y0 + y1) / 2,
            lbl,
            fontsize=7,
            color="#999999",
            va="center",
            ha="left",
            style="italic",
            zorder=1,
        )

    trunk_x = 0.52
    soma_y = 0.30

    # Apical trunk (soma -> L1 tuft)
    ax.plot(
        [trunk_x, trunk_x], [soma_y + 0.06, 0.88], color="#333333", lw=2.2, zorder=2
    )
    # Apical tuft branches in L1
    for dx in (-0.14, -0.07, 0.0, 0.07, 0.14):
        ax.plot(
            [trunk_x, trunk_x + dx], [0.88, 0.97], color="#333333", lw=1.2, zorder=2
        )
    # Basal dendrites
    for dx in (-0.16, -0.08, 0.08, 0.16):
        ax.plot(
            [trunk_x, trunk_x + dx],
            [soma_y - 0.05, soma_y - 0.16],
            color="#333333",
            lw=1.2,
            zorder=2,
        )
    # Axon
    ax.annotate(
        "",
        xy=(trunk_x, 0.10),
        xytext=(trunk_x, soma_y - 0.06),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5),
        zorder=2,
    )

    # Soma
    ax.add_patch(
        mpatches.Ellipse(
            (trunk_x, soma_y),
            0.13,
            0.15,
            facecolor="#e2d9f3",
            edgecolor="#333333",
            lw=1.5,
            zorder=3,
        )
    )
    ax.text(
        trunk_x,
        soma_y,
        "L5\nsoma",
        ha="center",
        va="center",
        fontsize=6.5,
        fontweight="bold",
        zorder=4,
    )

    # NMDA Ca2+ plateau / coincidence-detection zone on apical trunk
    ax.add_patch(
        mpatches.Ellipse(
            (trunk_x, 0.62),
            0.07,
            0.10,
            facecolor=_DIS,
            edgecolor="#7a5c00",
            lw=1.0,
            alpha=0.85,
            zorder=3,
        )
    )

    # Apical input (top-down prediction) — excitatory
    ax.annotate(
        "",
        xy=(trunk_x - 0.10, 0.92),
        xytext=(0.20, 0.92),
        arrowprops=dict(arrowstyle="-|>", color=_EX, lw=1.4),
        zorder=4,
    )
    ax.text(
        0.19,
        0.92,
        "top-down\nprediction",
        ha="right",
        va="center",
        fontsize=6.5,
        color=_EX,
    )

    # Basal input (bottom-up evidence) — excitatory
    ax.annotate(
        "",
        xy=(trunk_x - 0.10, soma_y - 0.13),
        xytext=(0.20, soma_y - 0.13),
        arrowprops=dict(arrowstyle="-|>", color=_EX, lw=1.4),
        zorder=4,
    )
    ax.text(
        0.19,
        soma_y - 0.13,
        "bottom-up\nevidence",
        ha="right",
        va="center",
        fontsize=6.5,
        color=_EX,
    )

    # Plateau annotation
    ax.annotate(
        "NMDA Ca²⁺ plateau\n(apical–basal\ncoincidence)",
        xy=(trunk_x + 0.035, 0.62),
        xytext=(0.72, 0.66),
        fontsize=6.5,
        color="#7a5c00",
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="->", color="#7a5c00", lw=0.9),
    )

    # Output mismatch annotation
    ax.text(
        trunk_x,
        0.045,
        r"mismatch $\to\ \varepsilon^e / \varepsilon^i$",
        ha="center",
        va="top",
        fontsize=8,
        color="#333333",
        fontweight="bold",
    )

    ax.set_title(
        "L5 thick-tufted pyramidal cell\napical–basal mismatch computation",
        fontsize=8,
        fontweight="bold",
    )


def plot(show: bool = True) -> None:
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.4, wspace=0.35)

    ax_main = fig.add_subplot(gs[0, :])
    ax_rest = fig.add_subplot(gs[1, 0])
    ax_active = fig.add_subplot(gs[1, 1])
    ax_l5 = fig.add_subplot(gs[1, 2])

    draw_main_circuit(ax_main)
    draw_state_inset(ax_rest, "Resting state\n(β_SM low)", "low")
    draw_state_inset(ax_active, "Somatic-marker\nretrieval (β_SM high)", "high")
    draw_l5_pyramid(ax_l5)

    # Panel letters
    for ax, lbl in [(ax_main, "A"), (ax_rest, "B"), (ax_l5, "C")]:
        ax.text(
            -0.02,
            1.04,
            lbl,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="right",
        )

    # No baked figure title/number per the shared rendering specification;
    # the caption is supplied separately in the manuscript.
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig2_somatic_marker_circuit.pdf")
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
