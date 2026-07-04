"""Paper 1 — Figure 3: The Thermodynamic Bridge and the κ Parameter (§4.6).

Schematic bridge diagram: Landauer limit (J/bit, kT·ln2 @ 310K) -> ATP/spike
biological overhead -> the APGI bridge parameter κ (≈100 ATP/bit, conditional
on the network-state bit count). Explicitly distinguishes the per-bit unit
conversion cost (κ) from the whole-event thermodynamic inefficiency factor
(~1,700x the Landauer minimum for κ ≈ 100 ATP/bit), preventing cross-tier
conflation (§4.6 caption).

The Landauer minimum energy is computed via
``apgi.extensions.epistemic.landauer_minimum_energy`` rather than hardcoded,
so this figure stays numerically consistent with the rest of the codebase.

Run:
    python figures/paper1/generate_fig3.py
    python figures/paper1/generate_fig3.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import save_figure
from apgi.extensions.epistemic import (
    BODY_TEMPERATURE_K,
    KAPPA_ATP_PER_BIT_DEFAULT,
    landauer_minimum_energy,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# §4.6 derivation constants (order-of-magnitude bridge, not fitted constants)
# ---------------------------------------------------------------------------
ATP_HYDROLYSIS_ENERGY_J = 5.07e-20  # J released per ATP molecule hydrolysed
ATP_PER_SPIKE = 2.4e9  # ATP molecules per cortical action potential (Lennie 2003)
SPIKES_PER_IGNITION = 1e9  # ~10^8 active neurons x ~10 spikes over ~500 ms
TOTAL_ATP_PER_IGNITION = ATP_PER_SPIKE * SPIKES_PER_IGNITION  # ~2.4e18 ATP

# Landauer minimum per bit, computed from the real function (not hardcoded).
LANDAUER_J_PER_BIT = landauer_minimum_energy(1.0, BODY_TEMPERATURE_K)
LANDAUER_ATP_PER_BIT = LANDAUER_J_PER_BIT / ATP_HYDROLYSIS_ENERGY_J

# Network-state bit-count convention (§4.6): ~2e16 bits -> kappa ~ 1.2e2 ATP/bit
NETWORK_STATE_BITS = TOTAL_ATP_PER_IGNITION / KAPPA_ATP_PER_BIT_DEFAULT

# Whole-event thermodynamic inefficiency factor: kappa / Landauer(ATP/bit)
INEFFICIENCY_FACTOR = KAPPA_ATP_PER_BIT_DEFAULT / LANDAUER_ATP_PER_BIT


def _box(ax, xy, w, h, text, color, fontsize=8, edgecolor="#333333"):
    rect = mpatches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        zorder=4,
        multialignment="center",
    )


def _arrow(ax, src, dst, color="#333333", lw=2.0, label=None, label_dy=0.05):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
        zorder=5,
    )
    if label:
        mid = ((src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2 + label_dy)
        ax.text(mid[0], mid[1], label, ha="center", fontsize=7, color=color)


def draw_bridge(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stage_y = 0.38
    box_h = 0.22
    box_w = 0.24
    gap = (1.0 - 3 * box_w) / 4

    x1 = gap
    x2 = gap + (box_w + gap)
    x3 = gap + 2 * (box_w + gap)

    _box(
        ax,
        (x1, stage_y),
        box_w,
        box_h,
        f"Landauer limit\n$kT\\ln2 \\approx$ {LANDAUER_J_PER_BIT:.2e} J/bit\n(310 K)",
        "#cce5ff",
        fontsize=7.5,
    )
    _box(
        ax,
        (x2, stage_y),
        box_w,
        box_h,
        "ATP hydrolysis\n"
        f"{ATP_PER_SPIKE:.1e} ATP/spike\n"
        f"$\\approx {LANDAUER_ATP_PER_BIT:.2f}$ ATP/bit\n(thermodynamic floor)",
        "#d4edda",
        fontsize=7,
    )
    _box(
        ax,
        (x3, stage_y),
        box_w,
        box_h,
        r"APGI bridge $\kappa$"
        f"\n$\\approx {KAPPA_ATP_PER_BIT_DEFAULT:.0f}$ ATP/bit\n"
        "(network-state\nbit-count convention)",
        "#fff3cd",
        fontsize=7.5,
    )

    _arrow(
        ax,
        (x1 + box_w, stage_y + box_h / 2),
        (x2, stage_y + box_h / 2),
        color="#2166ac",
        label="unit\nconversion",
    )
    _arrow(
        ax,
        (x2 + box_w, stage_y + box_h / 2),
        (x3, stage_y + box_h / 2),
        color="#4dac26",
        label="biological\noverhead",
    )

    # Whole-event inefficiency annotation spanning stage 1 -> stage 3.
    ax.annotate(
        "",
        xy=(x3 + box_w / 2, stage_y - 0.10),
        xytext=(x1 + box_w / 2, stage_y - 0.10),
        arrowprops=dict(
            arrowstyle="-",
            color="#d6604d",
            lw=1.3,
            linestyle="--",
            connectionstyle="bar,fraction=0.15",
        ),
        zorder=2,
    )
    ax.text(
        (x1 + x3 + box_w) / 2,
        stage_y - 0.24,
        f"whole-event thermodynamic inefficiency $\\approx {INEFFICIENCY_FACTOR:,.0f}\\times$\n"
        "the Landauer minimum (distinct from $\\kappa$, the per-bit unit cost)",
        ha="center",
        va="top",
        fontsize=6.8,
        color="#d6604d",
        style="italic",
    )

    ax.text(
        0.5,
        0.98,
        "Figure 3 — The Thermodynamic Bridge and the $\\kappa$ Parameter (§4.6)",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.90,
        r"$\kappa \approx 100$ ATP/bit is an order-of-magnitude calibration target, not a fitted constant"
        "\n(conditional on the network-state bit-count convention; falsification window $\\kappa \\in [10, 1{,}000]$ ATP/bit)",
        ha="center",
        va="top",
        fontsize=7.2,
        color="#555555",
        style="italic",
        transform=ax.transAxes,
    )


def draw_ignition_energy_panel(ax) -> None:
    """Panel B — the ~2x10^18 ATP-per-ignition-event derivation (§4.6 steps 1-3)."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    steps = [
        (
            "1. ATP per spike",
            f"{ATP_PER_SPIKE:.1e} ATP\n(Attwell & Laughlin 2001;\nscaled per Lennie 2003)",
        ),
        (
            "2. Spikes per ignition",
            f"{SPIKES_PER_IGNITION:.0e} spikes\n(~$10^8$ active neurons\n"
            "x ~10 spikes / ~500 ms)",
        ),
        (
            "3. Total ATP per ignition",
            f"{TOTAL_ATP_PER_IGNITION:.1e} ATP\n(order-of-magnitude)",
        ),
    ]
    y = 0.78
    dy = 0.30
    for title, body in steps:
        ax.text(0.02, y, title, fontsize=8, fontweight="bold", va="top")
        ax.text(0.06, y - 0.09, body, fontsize=7.5, va="top", color="#333333")
        y -= dy

    ax.text(
        0.5,
        1.0,
        "Per-ignition energy budget",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
    )


def draw_bit_count_panel(ax) -> None:
    """Panel C — network-state vs. percept-level bit-count conventions."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        1.0,
        "Bit-count convention determines $\\kappa$",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
    )

    rows = [
        ("Percept-level coding\n(~1-2 bits/spike)", "~$10^9$-$10^{10}$ bits", r"$\kappa \approx 10^8$-$10^9$ ATP/bit"),
        ("Network-state activation\npatterns (~$10^{12}$ synapses)", "~$2\\times10^{16}$ bits", r"$\kappa \approx 10^2$ ATP/bit"),
    ]
    y = 0.78
    dy = 0.36
    for label, bits, kappa in rows:
        ax.text(0.02, y, label, fontsize=7.2, va="top", fontweight="bold", color="#333333")
        ax.text(0.5, y, bits, fontsize=7.5, va="top", ha="center", color="#2166ac")
        ax.text(0.98, y, kappa, fontsize=7.5, va="top", ha="right", color="#4dac26")
        y -= dy

    ax.text(
        0.5,
        0.05,
        "Six orders of magnitude apart — the [10, 1,000] ATP/bit\n"
        "falsification window is conditional on the network-state convention.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#888888",
        style="italic",
    )


def plot(show: bool = True) -> None:
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.3)

    ax_main = fig.add_subplot(gs[0, :])
    ax_energy = fig.add_subplot(gs[1, 0])
    ax_bits = fig.add_subplot(gs[1, 1])

    draw_bridge(ax_main)
    draw_ignition_energy_panel(ax_energy)
    draw_bit_count_panel(ax_bits)

    for ax, lbl in [(ax_main, "A"), (ax_energy, "B"), (ax_bits, "C")]:
        ax.text(
            -0.02,
            1.05,
            lbl,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="right",
        )

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig3_thermodynamic_bridge.pdf")
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
