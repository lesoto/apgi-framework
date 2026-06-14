"""Paper 1 — Figure S2: Parameter Sensitivity Tornado Plot (Appendix A.5).

Horizontal tornado plot: parameters ranked by ±% change in P(ignition),
plus robustness percentages bar chart.

Run:
    python figures/paper1/generate_figS2.py
    python figures/paper1/generate_figS2.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import label_axes, save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# Ranked by influence magnitude (±% change in P(ignition), ±50% parameter sweep)
PARAMS = [
    (r"$\theta_0$", 34),
    (r"$\Pi^i_{\mathrm{baseline}}$", 22),
    (r"$\beta_{\mathrm{SM}}$", 18),
    (r"$\tau_S$", 12),
    (r"$\gamma_{\mathrm{sig}}$", 8),
    (r"$\kappa_{\mathrm{meta}}$", 5),
    (r"$\lambda_\theta$", 3),
    (r"$\lambda_\Pi$", 2),
]

ROBUSTNESS = [
    ("Bistability\npreserved", 94.2),
    ("Cardiac-phase\neffect", 87.6),
    ("HEP–P3b\ncorrelation", 91.8),
    ("Post-block\nθ elevation", 96.4),
]


def plot(show: bool = True) -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.8, 1]}
    )

    # ── Tornado plot ───────────────────────────────────────────────────────
    labels = [p[0] for p in PARAMS]
    vals = [p[1] for p in PARAMS]
    y = np.arange(len(labels))
    colors_pos = ["#2166ac"] * len(labels)
    colors_neg = ["#d6604d"] * len(labels)

    ax1.barh(
        y,
        vals,
        left=0,
        color=colors_pos,
        edgecolor="#333333",
        lw=0.8,
        label="+50% parameter",
        height=0.55,
    )
    ax1.barh(
        y,
        [-v for v in vals],
        left=0,
        color=colors_neg,
        edgecolor="#333333",
        lw=0.8,
        label="−50% parameter",
        height=0.55,
        alpha=0.75,
    )
    ax1.axvline(0, color="#333333", lw=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("% change in P(ignition)", fontsize=10)
    ax1.set_title(
        "Sensitivity Tornado\n(one-at-a-time ±50% sweep)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=8, loc="lower right")

    # annotate values
    for i, v in enumerate(vals):
        ax1.text(v + 0.5, i, f"+{v}%", va="center", fontsize=7.5, color="#2166ac")
        ax1.text(
            -v - 0.5,
            i,
            f"−{v}%",
            va="center",
            ha="right",
            fontsize=7.5,
            color="#d6604d",
        )

    # ── Robustness bar chart ───────────────────────────────────────────────
    rob_labels = [r[0] for r in ROBUSTNESS]
    rob_vals = [r[1] for r in ROBUSTNESS]
    x2 = np.arange(len(rob_labels))
    bars = ax2.bar(
        x2,
        rob_vals,
        color=["#4dac26", "#4dac26", "#2166ac", "#2166ac"],
        edgecolor="#333333",
        lw=1.0,
        width=0.55,
    )
    ax2.axhline(80, lw=1.2, ls="--", color="#888888")
    ax2.text(3.3, 80.5, "80%\nthreshold", fontsize=7.5, color="#888888")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(rob_labels, fontsize=7.5)
    ax2.set_ylabel("% simulations passing", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.set_title("Robustness\n(Monte Carlo)", fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, v in zip(bars, rob_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.5,
            f"{v}%",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )

    label_axes([ax1, ax2])
    fig.suptitle(
        "Figure S2 — Parameter Sensitivity Tornado Plot (Appendix A.5)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figS2_sensitivity_tornado.pdf")
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
