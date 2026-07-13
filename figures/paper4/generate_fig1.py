"""Paper 4 — Figure 1: Three-Tier Epistemic Architecture with Bridge Principles (§3.2).

Three-tier vertical stack with valid downward bridges (solid) and
absent upward bridges (dashed red X). Four-step double-bridge calculation inset.
L4 (phenomenal) deliberately excluded.

Run:
    python figures/paper4/generate_fig1.py
    python figures/paper4/generate_fig1.py --no-show
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from apgi.extensions.epistemic import (
    KAPPA_ATP_PER_BIT_DEFAULT,
    inefficiency_ratio,
    landauer_minimum_energy,
)
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# Landauer minimum, computed from the real implementation rather than
# hardcoded, so the figure and apgi.extensions.epistemic cannot drift apart.
_LANDAUER_J_PER_BIT = landauer_minimum_energy(n_bits=1, temperature_k=310.0)

# CRITICAL (spec Fig.1 prompt): the double-bridge chain's step-3 neural-
# inefficiency factor is the bandwidth-derived whole-event ratio (~20
# transmitted bits/event vs ~1e-1 J actual energy), which evaluates to
# ≈1.7e18 — the framework's primary Tier-1 falsification target. This is
# numerically and conceptually DISTINCT from the ~1e13-1e14 general
# synaptic-energy overhead factor (Attwell & Laughlin 2001) and must never
# be substituted for it in this figure.
_N_BITS_PER_IGNITION = 20.0
_ACTUAL_ENERGY_J_PER_IGNITION = 1e-1
_INEFFICIENCY_FACTOR = inefficiency_ratio(
    actual_energy_j=_ACTUAL_ENERGY_J_PER_IGNITION, n_bits=_N_BITS_PER_IGNITION
)

# kappa: the per-bit ATP metabolic-conversion cost (Tier2->Tier1 Landauer
# bridge quantity), converted to joules/bit for display. ~5e-18 J/bit at
# ~50 zJ/ATP hydrolysis energy. Distinct from the whole-event inefficiency
# factor above (different quantity: per-bit unit cost vs whole-event ratio).
_JOULES_PER_ATP = 5e-20
_KAPPA_J_PER_BIT = KAPPA_ATP_PER_BIT_DEFAULT * _JOULES_PER_ATP


def _format_landauer(value: float) -> str:
    """Format a small energy value as "a×10^b J" to match the figure's display precision."""
    exponent = math.floor(math.log10(value))
    mantissa = value / (10 ** exponent)
    return f"≈{mantissa:.0f}×10{_superscript(exponent)} J"


def _format_sci(value: float, mantissa_fmt: str = "{:.0f}") -> str:
    """Format a value as "a×10^b" (no unit) using the same superscript style."""
    exponent = math.floor(math.log10(value))
    mantissa = value / (10 ** exponent)
    return f"≈{mantissa_fmt.format(mantissa)}×10{_superscript(exponent)}"


_SUPERSCRIPT_MAP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _superscript(n: int) -> str:
    return str(n).translate(_SUPERSCRIPT_MAP)

TIERS = [
    {
        "name": "Tier 3 — Computational",
        "currency": "prediction errors ε, precision Π,\nvariational F, d′, AUC",
        "evidence": "single-trial EEG, model fit, BIC/AIC",
        "color": "#2166ac",
        "y": 0.72,
    },
    {
        "name": "Tier 2 — Information-Theoretic",
        "currency": "bits, nats, mutual information,\nchannel capacity, Φ",
        "evidence": "transfer entropy, bias-corrected MI, Φ",
        "color": "#4dac26",
        "y": 0.42,
    },
    {
        "name": "Tier 1 — Thermodynamic",
        "currency": "joules, ATP, watts, entropy",
        "evidence": "³¹P-MRS, two-photon imaging,\nBOLD-calibrated fMRI (proxy)",
        "color": "#d6604d",
        "y": 0.12,
    },
]

BRIDGES = [
    {
        # Spec §3.2.2 / Table 1: the T3->T2 bridge is the "Approximate-Inference
        # Bridge" (variational F minimisation -> MI maximisation). Kappa (the
        # ATP/bit metabolic-cost parameter) is EXCLUSIVELY a Tier2->Tier1
        # quantity (Glossary: "kappa is a Tier 2-> Tier 1 quantity") and must
        # not appear on this bridge's label.
        "label": "T3→T2: Approximate-Inference Bridge\nvariational F minimisation → MI maximisation\nStatus: derived in ML, unconverged in vivo",
        "y_from": 0.72,
        "y_to": 0.52,
        "valid": True,
        "color": "#4dac26",
    },
    {
        "label": (
            "T2→T1: Landauer bridge\nE ≥ kT·ln2 per bit erased\n"
            f"κ ≈ {KAPPA_ATP_PER_BIT_DEFAULT:.0f} ATP/bit {_format_sci(_KAPPA_J_PER_BIT)} J/bit\n"
            "Status: theoretically bounded, κ unmeasured"
        ),
        "y_from": 0.42,
        "y_to": 0.22,
        "valid": True,
        "color": "#d6604d",
    },
    {
        "label": "T3→T1: Double bridge (3→2→1)\nrequires both above\nStatus: composite, unvalidated",
        "y_from": 0.72,
        "y_to": 0.22,
        "valid": True,
        "color": "#888888",
        # Arc this curve outward to the RIGHT of the solid-bridge column
        # (x0=0.60) rather than left, where it previously crossed through the
        # dashed upward-bridge arrows plotted at x=0.30. The label is placed
        # below the other bridge labels (fixed y) so it never overlaps the
        # arc itself or the other two bridge labels.
        "curve_rad": -0.55,
        "label_pos": (0.60, 0.05),
    },
]

ABSENT_BRIDGES = [
    {"y_from": 0.12, "y_to": 0.42, "label": "T1→T2\nNot yet specified"},
    {"y_from": 0.42, "y_to": 0.72, "label": "T2→T3\nNot yet specified"},
]

# CRITICAL (spec Fig.1 prompt): step (3) MUST render the inefficiency factor
# EXACTLY as "≈1.7×10¹⁸×" — never 10¹³, 10¹⁶, nor the separate ~10¹²–10¹⁴
# synaptic-overhead bound (kept out of this list; see module docstring above
# double_bridge_energy_estimate for that distinct secondary quantity).
CALC_STEPS = [
    "(1) Bits processed per ignition\n(~20 transmitted bits/event, bandwidth-derived)",
    f"(2) × Landauer minimum\n({_format_landauer(_LANDAUER_J_PER_BIT)}/bit at 310K; precisely {_format_sci(_LANDAUER_J_PER_BIT, '{:.2f}')} J/bit)",
    f"(3) × neural-inefficiency factor {_format_sci(_INEFFICIENCY_FACTOR, '{:.1f}')}×\n"
    "(the framework's primary Tier-1 falsification target)",
    "(4) compare with PET/BOLD expenditure\n(currently unvalidated)",
]
CALC_STEP_COLORS = ["#2166ac", "#4dac26", "#d6604d", "#7b3fe4"]


def plot(show: bool = True) -> None:
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1.5, 1.2], wspace=0.30)
    ax_tiers = fig.add_subplot(gs[0])
    ax_bridges = fig.add_subplot(gs[1])
    ax_calc = fig.add_subplot(gs[2])

    for ax in [ax_tiers, ax_bridges, ax_calc]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ── Tier boxes ────────────────────────────────────────────────────────
    BOX_H = 0.20
    for tier in TIERS:
        rect = mpatches.FancyBboxPatch(
            (0.05, tier["y"]),
            0.90,
            BOX_H,
            boxstyle="round,pad=0.01",
            facecolor=tier["color"] + "22",
            edgecolor=tier["color"],
            lw=2.5,
            zorder=3,
        )
        ax_tiers.add_patch(rect)
        ax_tiers.text(
            0.50,
            tier["y"] + BOX_H / 2 + 0.02,
            tier["name"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=tier["color"],
            zorder=4,
        )
        ax_tiers.text(
            0.50,
            tier["y"] + 0.04,
            f"Currency: {tier['currency']}\nEvidence: {tier['evidence']}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#444444",
            zorder=4,
            multialignment="center",
        )

    # L4 exclusion note
    ax_tiers.text(
        0.50,
        0.03,
        "Footnote: phenomenal level L4 (Φ-boundary) is excluded by design (§8.4)",
        ha="center",
        fontsize=7,
        color="#888888",
        style="italic",
    )

    ax_tiers.set_title(
        "Three-Tier Epistemic Architecture", fontsize=9, fontweight="bold"
    )

    # ── Valid bridges ─────────────────────────────────────────────────────
    x0 = 0.60
    for br in BRIDGES:
        rad = br.get("curve_rad", 0.0)
        ax_bridges.annotate(
            "",
            xy=(x0, br["y_to"] + 0.10),
            xytext=(x0, br["y_from"]),
            arrowprops=dict(
                arrowstyle="->",
                color=br["color"],
                lw=2.0,
                connectionstyle=f"arc3,rad={rad}",
            ),
            zorder=5,
        )
        if "label_pos" in br:
            label_x, label_y = br["label_pos"]
        else:
            mid_y = (br["y_from"] + br["y_to"] + 0.10) / 2
            label_x, label_y = x0 + 0.08, mid_y
        ax_bridges.text(
            label_x,
            label_y,
            br["label"],
            ha="left",
            va="center",
            fontsize=6.5,
            color=br["color"],
            multialignment="left",
        )

    # Absent bridges (dashed + X)
    for ab in ABSENT_BRIDGES:
        ax_bridges.annotate(
            "",
            xy=(0.30, ab["y_to"] + 0.10),
            xytext=(0.30, ab["y_from"]),
            arrowprops=dict(
                arrowstyle="->",
                color="#cc0000",
                lw=1.5,
                linestyle="dashed",
            ),
            zorder=5,
        )
        mid_y = (ab["y_from"] + ab["y_to"] + 0.10) / 2
        ax_bridges.text(
            0.20, mid_y, "✗", ha="center", fontsize=14, color="#cc0000", zorder=6
        )
        ax_bridges.text(
            0.08,
            mid_y,
            ab["label"],
            ha="center",
            fontsize=6.5,
            color="#cc0000",
            multialignment="center",
            style="italic",
        )

    ax_bridges.text(
        0.50,
        0.96,
        "Valid bridges (solid)\nAbsent bridges (dashed ✗)",
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
    )

    # ── Four-step double-bridge calculation ───────────────────────────────
    ax_calc.text(
        0.50,
        0.94,
        "Double-bridge\ncalculation",
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="#555555",
    )
    for i, step in enumerate(CALC_STEPS):
        y = 0.82 - i * 0.20
        step_color = CALC_STEP_COLORS[i]
        circ = mpatches.Circle(
            (0.13, y), 0.055, facecolor=step_color, edgecolor="white", lw=1.0, zorder=3
        )
        ax_calc.add_patch(circ)
        ax_calc.text(
            0.13,
            y,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            zorder=4,
        )
        ax_calc.text(
            0.22,
            y,
            step,
            ha="left",
            va="center",
            fontsize=7.2,
            color="#333333",
            multialignment="left",
        )
        if i < len(CALC_STEPS) - 1:
            ax_calc.annotate(
                "",
                xy=(0.13, y - 0.10),
                xytext=(0.13, y - 0.025),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
            )

    # ── Compact shared legend (arrow meaning + tier/comparison colour key) ──
    legend_handles = [
        mpatches.FancyArrow(
            0, 0, 0, -1, width=0.001, color="#333333", label="Downward bridge (specified)"
        ),
        mpatches.FancyArrow(
            0, 0, 0, 1, width=0.001, color="#cc0000", label="Upward bridge (deliberately unspecified)"
        ),
        mpatches.Patch(facecolor="#2166ac", label="Computational (Tier 3)"),
        mpatches.Patch(facecolor="#4dac26", label="Information-theoretic (Tier 2)"),
        mpatches.Patch(facecolor="#d6604d", label="Thermodynamic (Tier 1)"),
        mpatches.Patch(facecolor="#7b3fe4", label="Empirical comparison"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig1_three_tier_epistemic_architecture.pdf")
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
