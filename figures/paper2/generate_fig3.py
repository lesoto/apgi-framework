"""Paper 2 — Figure 3: APGI-LNN Bifurcation Analysis (§4.5).

Two panels (corrected from critique):
  A — Bifurcation diagram: ρ_crit vs. |y(t)|/θₜ, three zones
  B — Ignition-probability sigmoid family as a function of Π(t)

Caption notes: heterogeneous-τ form used; proof-of-concept scale only.

Run:
    python figures/paper2/generate_fig3.py
    python figures/paper2/generate_fig3.py --no-show
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


def plot(show: bool = True) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel A: Bifurcation diagram ──────────────────────────────────────
    rho = np.linspace(0.0, 1.5, 600)
    # Below critical manifold: sub-threshold
    rho_crit = 0.95
    # |y(t)|/θₜ: sub-threshold basin grows nonlinearly near ρ_crit
    signal_low = 0.3 * (rho / rho_crit) ** 2
    # Above ρ_crit: ignition zone
    signal_high = np.where(rho >= rho_crit, 0.8 + 0.5 * (rho - rho_crit), np.nan)

    ax1.fill_between(
        rho, 0, signal_low, alpha=0.25, color="#6baed6", label="Sub-threshold basin"
    )
    ax1.fill_between(
        rho,
        np.where(rho >= rho_crit, signal_low, np.nan),
        np.where(rho >= rho_crit, signal_high, np.nan),
        alpha=0.25,
        color="#d6604d",
        label="Ignition zone",
    )
    ax1.plot(rho, signal_low, lw=1.8, color="#2166ac")
    ax1.plot(
        rho[rho >= rho_crit], signal_high[rho >= rho_crit], lw=1.8, color="#d6604d"
    )

    # Critical manifold boundary
    ax1.axvline(rho_crit, lw=1.8, ls="--", color="#333333")
    ax1.annotate(
        r"Saddle-node: $\rho_{\mathrm{crit}} \approx 0.95$",
        xy=(rho_crit, 0.5),
        xytext=(rho_crit + 0.08, 0.55),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=1.0),
    )

    ax1.set_xlabel(r"Spectral radius $\rho_{\mathrm{crit}}$", fontsize=10)
    ax1.set_ylabel(r"$|y(t)| / \theta_t$ (normalised signal)", fontsize=10)
    ax1.set_title("Bifurcation diagram\n(three zones)", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(0.0, 1.45)
    ax1.set_ylim(0, 1.5)

    # Critical manifold label
    ax1.text(
        rho_crit / 2,
        0.02,
        "Sub-threshold\nbasin",
        ha="center",
        fontsize=8,
        color="#2166ac",
    )
    ax1.text(1.2, 1.1, "Ignition\nzone", ha="center", fontsize=8, color="#d6604d")
    ax1.text(
        rho_crit + 0.01,
        0.02,
        "Critical\nmanifold",
        ha="left",
        fontsize=7.5,
        color="#555555",
        style="italic",
    )

    # ── Panel B: Sigmoid family ────────────────────────────────────────────
    x = np.linspace(-1, 3, 400)
    theta_val = 1.0
    precision_levels = [
        (0.5, "#d0d0d0", "Low Π (shallow, graded)"),
        (2.0, "#9ecae1", "Mid Π"),
        (5.0, "#4292c6", "High Π"),
        (12.0, "#08519c", r"High Π (α_psy ≥ 10, near-discrete)"),
    ]
    for pi, color, label in precision_levels:
        gamma = pi * 5.0
        P = 1 / (1 + np.exp(-gamma * (x - theta_val)))
        ax2.plot(x, P, lw=2.0, color=color, label=label)

    ax2.axvline(theta_val, lw=1.2, ls="--", color="#555555", alpha=0.7)
    ax2.axhspan(0.5, 1.0, xmin=0.55, alpha=0.06, color="#d6604d")
    ax2.annotate(
        r"$\alpha_{psy} \geq 10$" + "\n(predicted range)",
        xy=(1.6, 0.75),
        fontsize=8,
        color="#d6604d",
        ha="center",
    )
    ax2.set_xlabel(r"$|y(t)| / \theta_t$", fontsize=10)
    ax2.set_ylabel(r"$P(\mathrm{ignition})$", fontsize=10)
    ax2.set_title(
        "Ignition-probability sigmoid family\n(steepness ∝ Π(t))",
        fontsize=10,
        fontweight="bold",
    )
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_xlim(-0.5, 2.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    label_axes([ax1, ax2])

    caption_note = (
        "Heterogeneous-τ form used (§4.1). "
        "Proof-of-concept scale only; biologically realistic validation (N ≥ 1,000 units) pending."
    )
    fig.text(
        0.5,
        -0.03,
        caption_note,
        ha="center",
        fontsize=7.5,
        color="#666666",
        style="italic",
    )

    fig.suptitle(
        "Figure 3 — APGI-LNN Bifurcation Analysis: Parameter Sweep and Ignition Manifold\n"
        "(Paper 2, §4.5)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig3_bifurcation_analysis.pdf")
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
