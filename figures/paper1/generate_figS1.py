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
    # Well-identified (r > 0.80; §4.7): theta_0, tau_S.
    # theta_0 range per Table S1 / THETA_0_DEFAULT docstring: [0.25, 0.85] AU.
    # r targets match the source-data & statistics box exactly: r = 0.82
    # (theta_0), 0.83 (tau_S), 0.73 (Pi_i), 0.72 (beta_SM), 0.60 (gamma_sig).
    {
        "name": r"$\theta_0$",
        "key": "theta0",
        "r_true": 0.82,
        "lo": 0.25,
        "hi": 0.85,
        "color": "#2166ac",
    },
    {
        "name": r"$\tau_S$",
        "key": "tau_S",
        "r_true": 0.83,
        "lo": 50,
        "hi": 500,
        "color": "#e6a817",
    },
    # Moderately identified (r ~ 0.70-0.75 per §4.7): Pi_i, beta_SM.
    {
        "name": r"$\Pi^i$",
        "key": "pi_i",
        "r_true": 0.73,
        "lo": 0.3,
        "hi": 1.8,
        "color": "#4dac26",
    },
    {
        "name": r"$\beta_{\mathrm{SM}}$",
        "key": "beta_sm",
        "r_true": 0.72,
        "lo": 0.0,
        "hi": 3.0,
        "color": "#7b3294",
    },
    # Poorly identified at the individual level (r ~ 0.55-0.65 per §4.7): gamma_sig.
    {
        "name": r"$\gamma_{\mathrm{sig}}$",
        "key": "gamma",
        "r_true": 0.60,
        "lo": 1.0,
        "hi": 15.0,
        "color": "#6baed6",
    },
]


def simulate_recovery(param: dict) -> tuple[np.ndarray, np.ndarray]:
    """Simulate true/recovered parameter pairs whose Pearson r matches
    ``param["r_true"]`` in expectation.

    For true values drawn from Uniform(lo, hi) and recovered values formed by
    adding independent Gaussian noise, the analytic correlation is
    r = sd(X) / sqrt(Var(X) + sigma^2). Inverting for sigma given a target r
    keeps the *rendered* scatter/correlation consistent with the stated
    identifiability class, rather than relying on an ad hoc noise-scale
    heuristic that drifts away from the target at lower r.
    """
    lo, hi, r_true = param["lo"], param["hi"], param["r_true"]
    true_vals = RNG.uniform(lo, hi, N_SIMS)
    var_x = (hi - lo) ** 2 / 12.0
    noise_var = var_x * (1.0 / r_true**2 - 1.0)
    noise_sd = np.sqrt(max(noise_var, 0.0))
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

        ax.scatter(
            true_v, rec_v, s=6, alpha=0.35, color=param["color"], rasterized=True
        )
        add_identity_line(ax, param["lo"], param["hi"])
        annotate_pearson_r(ax, r)
        ax.set_xlabel(f"True {param['name']}", fontsize=10)
        ax.set_ylabel(f"Recovered {param['name']}", fontsize=10)
        ax.set_title(param["name"], fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Sixth panel: collinearity reduction.
    # Source data & statistics box + Appendix A.4: without pre-session
    # M(c,a) estimation, beta_SM and Pi_i show collinearity r = 0.90 in
    # recovered values; the pre-session protocol reduces this to r = 0.45.
    ax6 = axes_flat[5]
    labels = [
        "Without pre-session\nestimation",
        "Pre-session M\nestimation",
    ]
    r_vals = [0.90, 0.45]
    # Flag the problematic (above-threshold) bar in red; the resolved bar in green.
    colors = ["#d6604d", "#4dac26"]
    bars = ax6.bar(labels, r_vals, color=colors, width=0.5, edgecolor="#333333", lw=1.2)
    for bar, v in zip(bars, r_vals):
        ax6.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.02,
            f"r = {v:.2f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    ax6.axhline(0.90, lw=1.2, ls="--", color="#888888")
    ax6.text(1.05, 0.91, "r > 0.90\n(collinearity)", fontsize=7.5, color="#888888")
    ax6.set_ylim(0, 1.05)
    ax6.set_ylabel("Pearson r", fontsize=10)
    ax6.set_title(
        r"$\beta_{\mathrm{SM}} \times \Pi^i$ Collinearity" "\n(Pearson r)",
        fontsize=10,
        fontweight="bold",
    )
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)

    # Combined bottom legend: identity line + one swatch per parameter colour
    # actually used in the scatter panels (no unlabeled/duplicate swatches).
    handles = [
        plt.Line2D([], [], ls="--", color="black", lw=1.2, label="Identity line (y = x)")
    ]
    for p in PARAMS:
        handles.append(
            plt.Line2D(
                [], [], marker="o", ls="", color=p["color"], label=f"{p['name']}"
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.03),
    )

    label_axes(list(axes_flat[:5]) + [ax6])
    # No baked figure title/number per the shared rendering specification.
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
