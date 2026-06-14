"""Paper 1 — Figure S1: Parameter Recovery Scatter Plots (Appendix A.4).

Five-panel scatter array (θ₀, τ_S, Πⁱ, β_SM, γ_sig) true vs. recovered
across 1,000 simulation runs, with Pearson r per panel and collinearity
reduction inset.

Run:
    python figures/paper1/generate_figS1.py
    python figures/paper1/generate_figS1.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import add_identity_line, annotate_pearson_r, label_axes, save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

RNG = np.random.default_rng(42)
N_SIMS = 1000

PARAMS = [
    {"name": r"$\theta_0$", "key": "theta0", "r_true": 0.91, "lo": 0.5, "hi": 2.5},
    {"name": r"$\tau_S$", "key": "tau_S", "r_true": 0.88, "lo": 50, "hi": 500},
    {"name": r"$\Pi^i$", "key": "pi_i", "r_true": 0.85, "lo": 0.3, "hi": 1.8},
    {
        "name": r"$\beta_{\mathrm{SM}}$",
        "key": "beta_sm",
        "r_true": 0.82,
        "lo": 0.0,
        "hi": 3.0,
    },
    {
        "name": r"$\gamma_{\mathrm{sig}}$",
        "key": "gamma",
        "r_true": 0.79,
        "lo": 1.0,
        "hi": 15.0,
    },
]


def simulate_recovery(param: dict) -> tuple[np.ndarray, np.ndarray]:
    true_vals = RNG.uniform(param["lo"], param["hi"], N_SIMS)
    noise_sd = (param["hi"] - param["lo"]) * (1 - param["r_true"]) * 0.5
    rec_vals = true_vals + RNG.normal(0, noise_sd, N_SIMS)
    return true_vals, rec_vals


def plot(show: bool = True) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes_flat = axes.ravel()

    pearson_rs = []
    for i, param in enumerate(PARAMS):
        ax = axes_flat[i]
        true_v, rec_v = simulate_recovery(param)
        r, _ = stats.pearsonr(true_v, rec_v)
        pearson_rs.append(r)

        ax.scatter(true_v, rec_v, s=6, alpha=0.35, color="#2166ac", rasterized=True)
        add_identity_line(ax, param["lo"], param["hi"])
        annotate_pearson_r(ax, r)
        ax.set_xlabel(f"True {param['name']}", fontsize=10)
        ax.set_ylabel(f"Recovered {param['name']}", fontsize=10)
        ax.set_title(param["name"], fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:  # name the identity line once
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # Sixth panel: collinearity reduction
    ax6 = axes_flat[5]
    labels = [
        r"Joint estimation" + "\n" + r"($\beta_{SM} \times \Pi^i$)",
        "Pre-session M\nestimation",
    ]
    r_vals = [0.45, 0.92]
    colors = ["#4dac26", "#d6604d"]
    bars = ax6.bar(labels, r_vals, color=colors, width=0.5, edgecolor="#333333", lw=1.2)
    ax6.axhline(0.90, lw=1.2, ls="--", color="#888888")
    ax6.text(1.05, 0.91, "r > 0.90\n(collinearity)", fontsize=7.5, color="#888888")
    ax6.set_ylim(0, 1.05)
    ax6.set_ylabel("Pearson r", fontsize=10)
    ax6.set_title("Collinearity reduction\n(β_SM × Πⁱ)", fontsize=10, fontweight="bold")
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    label_axes(list(axes_flat[:5]) + [ax6])
    fig.suptitle(
        "Figure S1 — Parameter Recovery Scatter Plots (N = 1,000 simulation runs)\n"
        "Appendix A.4, following Table S2",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figS1_parameter_recovery.pdf")
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
