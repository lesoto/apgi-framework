"""Figure 1 — APGI ignition dynamics.

Plots Sₜ and θₜ over a simulated trial sequence, highlighting ignition events.

Run:
    python figures/generate_figure1.py
    python figures/generate_figure1.py --no-show   # CI mode
"""

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from apgi.core import compute_pi_i_eff  # noqa: E402
from apgi.core import (
    BETA_SM_DEFAULT,
    compute_S_t,
    ignition_criterion,
    step_theta,
    theta_equilibrium,
)
from figures.utils import save_figure  # noqa: E402
from figures.utils import PALETTE, make_figure, vlines_ignition

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"


def simulate(
    n_steps: int = 200,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)

    pi_e = rng.uniform(0.8, 1.5, n_steps)
    z_e = rng.uniform(0.2, 1.0, n_steps)
    pi_i = rng.uniform(0.5, 1.5, n_steps)
    z_i = rng.uniform(0.1, 0.8, n_steps)
    C_metabolic = rng.uniform(0.5, 2.0, n_steps)
    V_information = rng.uniform(0.1, 1.0, n_steps)

    alpha, beta = 0.3, 0.7
    theta_t = theta_equilibrium(
        C_metabolic[0], V_information[0], lambda_theta=alpha, kappa_meta=beta
    )

    S_t_series = np.empty(n_steps)
    theta_series = np.empty(n_steps)
    ignition_series = np.zeros(n_steps, dtype=bool)

    for t in range(n_steps):
        pi_i_eff = compute_pi_i_eff(float(pi_i[t]), BETA_SM_DEFAULT, 0.0)
        S_t = compute_S_t(float(pi_e[t]), float(z_e[t]), pi_i_eff, float(z_i[t]))
        S_t_series[t] = S_t
        theta_series[t] = theta_t
        ignition_series[t] = ignition_criterion(S_t, theta_t)
        theta_t = step_theta(
            theta_t,
            float(C_metabolic[t]),
            float(V_information[t]),
            lambda_theta=alpha,
            kappa_meta=beta,
        )

    return {
        "S_t": S_t_series,
        "theta": theta_series,
        "ignition": ignition_series,
    }


def plot(data: dict, show: bool = True) -> None:
    fig, (ax,) = make_figure(ncols=1, width=10, height=4)
    t = np.arange(len(data["S_t"]))

    ax.plot(
        t,
        data["S_t"],
        lw=1.5,
        color=PALETTE["S_t"],
        label=r"$S_t$ (global integration)",
    )
    ax.plot(
        t,
        data["theta"],
        lw=1.5,
        color=PALETTE["theta"],
        linestyle="--",
        label=r"$\theta_t$ (threshold)",
    )

    vlines_ignition(ax, t, data["ignition"])

    patch = mpatches.Patch(
        color=PALETTE["ignition"], alpha=0.4, label="Ignition events"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [patch], fontsize=9, loc="upper right")

    ax.set_xlabel("Trial", fontsize=11)
    ax.set_ylabel("Signal", fontsize=11)
    ax.set_title("APGI Ignition Dynamics — Figure 1", fontsize=12)
    fig.tight_layout()

    save_figure(fig, OUTPUT_DIR / "figure1.pdf")

    if show:
        import matplotlib.pyplot as plt

        plt.show()
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 1")
    parser.add_argument(
        "--no-show", action="store_true", help="Skip plt.show() (CI mode)"
    )
    args = parser.parse_args()

    data = simulate()
    plot(data, show=not args.no_show)


if __name__ == "__main__":
    main()
