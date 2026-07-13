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
from apgi.extensions.epistemic import (
    BODY_TEMPERATURE_K,
    KAPPA_ATP_PER_BIT_DEFAULT,
    inefficiency_ratio,
    landauer_minimum_energy,
)
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# §4.6 derivation constants (order-of-magnitude bridge, not fitted constants)
# ---------------------------------------------------------------------------
ATP_HYDROLYSIS_ENERGY_J = 5.07e-20  # J released per ATP molecule hydrolysed
ATP_PER_SPIKE = 2.4e9  # ATP molecules per cortical action potential (Lennie 2003)
SPIKES_PER_IGNITION = 1e9  # ~10^8 active neurons x ~10 spikes over ~500 ms
TOTAL_ATP_PER_IGNITION = ATP_PER_SPIKE * SPIKES_PER_IGNITION  # ~2.4e18 ATP

# Landauer minimum per bit, computed from the real function (not hardcoded).
# At 310 K this evaluates to ~2.97e-21 J/bit (NOT 2.82e-21 — a prior review
# defect; kept numerically live here so it can never silently drift back).
LANDAUER_J_PER_BIT = landauer_minimum_energy(1.0, BODY_TEMPERATURE_K)
LANDAUER_ATP_PER_BIT = LANDAUER_J_PER_BIT / ATP_HYDROLYSIS_ENERGY_J

# Network-state bit-count convention (§4.6): ~2e16 bits -> kappa ~ 1.2e2 ATP/bit
NETWORK_STATE_BITS = TOTAL_ATP_PER_IGNITION / KAPPA_ATP_PER_BIT_DEFAULT

# kappa expressed in energy units (per-bit unit-conversion cost, microscopic
# bridge): ~100 ATP/bit * 5.07e-20 J/ATP =~ 5.07e-18 J/bit.
KAPPA_J_PER_BIT = KAPPA_ATP_PER_BIT_DEFAULT * ATP_HYDROLYSIS_ENERGY_J

# (i) Per-bit unit-conversion cost vs the Landauer floor: kappa (in ATP/bit)
# relative to the Landauer minimum expressed in ATP/bit -> ~1.7e3x. This is
# the "biological overhead" of the bridge parameter itself, NOT the
# whole-event inefficiency below -- keeping the two numerically distinct is
# the whole point of this figure (§4.6 caption; CRITICAL correction).
KAPPA_VS_LANDAUER_RATIO = KAPPA_ATP_PER_BIT_DEFAULT / LANDAUER_ATP_PER_BIT

# (ii) Whole-event thermodynamic inefficiency (macroscopic): the ratio of the
# TOTAL biological energy actually spent on one ignition event to the
# Landauer minimum for the amount of information a conscious percept
# plausibly carries (~20-25 bits/event, not the ~2e16-bit network-state
# convention used for kappa above -- using the wrong bit count here is
# exactly the cross-tier conflation this figure must prevent). This uses the
# same inefficiency_ratio() helper as apgi.extensions.epistemic so the value
# stays consistent with the rest of the codebase (~1.7e18x per that
# module's own docstring).
TOTAL_ENERGY_PER_IGNITION_J = TOTAL_ATP_PER_IGNITION * ATP_HYDROLYSIS_ENERGY_J
CONSCIOUS_BITS_PER_IGNITION = 24.0  # ~20-25 bits/event, illustrative order-of-magnitude
WHOLE_EVENT_INEFFICIENCY = inefficiency_ratio(
    TOTAL_ENERGY_PER_IGNITION_J, CONSCIOUS_BITS_PER_IGNITION, BODY_TEMPERATURE_K
)


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


def _arrow(ax, src, dst, color="#333333", lw=2.0, label=None, label_dy=0.05, fontsize=7):
    ax.annotate(
        "",
        xy=dst,
        xytext=src,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
        zorder=5,
    )
    if label:
        mid = ((src[0] + dst[0]) / 2, (src[1] + dst[1]) / 2 + label_dy)
        ax.text(
            mid[0],
            mid[1],
            label,
            ha="center",
            fontsize=fontsize,
            color=color,
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.85),
        )


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
        f"\n$\\approx {KAPPA_ATP_PER_BIT_DEFAULT:.0f}$ ATP/bit $\\approx {KAPPA_J_PER_BIT:.1e}$ J/bit\n"
        "(network-state\nbit-count convention)",
        "#fff3cd",
        fontsize=7,
    )

    _arrow(
        ax,
        (x1 + box_w, stage_y + box_h / 2),
        (x2, stage_y + box_h / 2),
        color="#2166ac",
        label="unit\nconversion",
        fontsize=6,
    )
    _arrow(
        ax,
        (x2 + box_w, stage_y + box_h / 2),
        (x3, stage_y + box_h / 2),
        color="#4dac26",
        label="biological\noverhead",
        fontsize=6,
    )

    # Panel D-style callout: explicitly separate the per-bit unit-conversion
    # cost (kappa, microscopic bridge) from the whole-event thermodynamic
    # inefficiency (macroscopic), joined by a "not-equal" glyph so the two
    # very different orders of magnitude can never be conflated (CRITICAL
    # correction: keep kappa and eta_thermo visually and numerically distinct).
    callout_y = stage_y - 0.12
    callout_h = 0.20
    left_w = 0.40
    right_w = 0.40
    mid_gap = 0.06
    left_x = (1.0 - (left_w + mid_gap + right_w)) / 2
    right_x = left_x + left_w + mid_gap

    _box(
        ax,
        (left_x, callout_y - callout_h),
        left_w,
        callout_h,
        r"Per-bit unit-conversion cost $\kappa$ (microscopic)"
        f"\n$\\kappa \\approx {KAPPA_ATP_PER_BIT_DEFAULT:.0f}$ ATP/bit $\\approx {KAPPA_J_PER_BIT:.1e}$ J/bit"
        f"\n$\\approx {KAPPA_VS_LANDAUER_RATIO:,.0f}\\times$ the per-bit Landauer floor"
        "\n(converts bits erased -> ATP; NOT the whole-event inefficiency)",
        "#ffffff",
        fontsize=6.3,
        edgecolor="#d6604d",
    )
    _box(
        ax,
        (right_x, callout_y - callout_h),
        right_w,
        callout_h,
        r"Whole-event thermodynamic inefficiency $\eta_{\mathrm{thermo}}$ (macroscopic)"
        f"\n$\\eta_{{\\mathrm{{thermo}}}} \\approx {WHOLE_EVENT_INEFFICIENCY:.1e}\\times$ the Landauer minimum"
        "\n(total ATP used per ignition / Landauer limit per bit)"
        "\n(NOT the per-bit conversion cost)",
        "#ffffff",
        fontsize=6.3,
        edgecolor="#d6604d",
    )
    ax.text(
        left_x + left_w + mid_gap / 2,
        callout_y - callout_h / 2,
        r"$\neq$",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#d6604d",
        zorder=5,
    )

    ax.text(
        0.5,
        0.98,
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
