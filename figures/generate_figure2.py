"""Figure 2 — Parameter recovery scatter plot (Appendix A.4).

Plots true vs recovered β and Πⁱ values with Pearson r annotations.

Run:
    python figures/generate_figure2.py
    python figures/generate_figure2.py --no-show   # CI mode
"""

import argparse
import pathlib
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from apgi.parameter_recovery import run_recovery_simulation  # noqa: E402
from figures.utils import (  # noqa: E402
    HALF_WIDTH,
    PALETTE,
    PANEL_HEIGHT,
    add_identity_line,
    annotate_pearson_r,
    label_axes,
    make_figure,
    save_figure,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"


def plot(results: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=2, width=HALF_WIDTH * 2, height=PANEL_HEIGHT)

    pairs = [
        (
            results["beta_sm_true"],
            results["beta_sm_hat"],
            r"$\beta_{SM}$",
            PALETTE["beta"],
        ),
        (results["pi_i_true"], results["pi_i_hat"], r"$\Pi^i$", PALETTE["pi_i"]),
    ]

    for ax, (true_vals, hat_vals, label, color) in zip(axes, pairs):
        true_arr = np.asarray(true_vals)
        hat_arr = np.asarray(hat_vals)
        r, _ = pearsonr(true_arr, hat_arr)

        ax.scatter(
            true_arr,
            hat_arr,
            s=30,
            alpha=0.7,
            color=color,
            edgecolors="white",
            linewidths=0.4,
        )

        lo = min(true_arr.min(), hat_arr.min()) * 0.95
        hi = max(true_arr.max(), hat_arr.max()) * 1.05
        add_identity_line(ax, lo, hi)

        ax.set_xlabel(f"True {label}", fontsize=11)
        ax.set_ylabel(f"Recovered {label}", fontsize=11)
        ax.set_title(f"Recovery of {label}  (r = {r:.3f})", fontsize=11)
        annotate_pearson_r(ax, r)

    label_axes(axes)
    fig.suptitle("Parameter Recovery — Figure 2", fontsize=12, y=1.01)
    fig.tight_layout()

    save_figure(fig, OUTPUT_DIR / "figure2.pdf")

    if show:
        import matplotlib.pyplot as plt

        plt.show()
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 2")
    parser.add_argument(
        "--no-show", action="store_true", help="Skip plt.show() (CI mode)"
    )
    parser.add_argument("--n-sim", type=int, default=40, help="Number of simulations")
    args = parser.parse_args()

    print(f"Running recovery simulation ({args.n_sim} simulations)…")
    results = run_recovery_simulation(
        n_simulations=args.n_sim, n_trials_per_sim=150, seed=2024
    )
    print(
        f"  r_beta_sm = {results['r_beta_sm']:.3f}  |  r_pi_i = {results['r_pi_i']:.3f}"
    )
    plot(results, show=not args.no_show)


if __name__ == "__main__":
    main()
