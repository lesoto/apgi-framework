"""Figure 2 — Parameter recovery scatter plot (Appendix A.4).

Plots true vs recovered β and Πⁱ values with Pearson r annotations.

Run:
    python figures/generate_figure2.py
    python figures/generate_figure2.py --no-show   # CI mode
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

from apgi.parameter_recovery import run_recovery_simulation

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"


def plot(results: dict, show: bool = True) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

    pairs = [
        (results["beta_true"], results["beta_hat"], r"$\beta$", "#2166ac"),
        (results["pi_i_true"], results["pi_i_hat"], r"$\Pi^i$", "#d6604d"),
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
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)

        ax.set_xlabel(f"True {label}", fontsize=11)
        ax.set_ylabel(f"Recovered {label}", fontsize=11)
        ax.set_title(f"Recovery of {label}  (r = {r:.3f})", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.annotate(
            f"r = {r:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10
        )

    fig.suptitle("Parameter Recovery — Figure 2", fontsize=12, y=1.01)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "figure2.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show:
        plt.show()
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
    print(f"  r_beta = {results['r_beta']:.3f}  |  r_pi_i = {results['r_pi_i']:.3f}")
    plot(results, show=not args.no_show)


if __name__ == "__main__":
    main()
