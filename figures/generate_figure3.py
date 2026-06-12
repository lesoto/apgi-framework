"""Figure 3 — Protocol 1: HEP-modulated detection by cardiac phase (Pred 1.a–Pred 1.c).

Simulates the key prediction of protocol_1_cardiac_eeg.json:
HEP amplitude (Πⁱ proxy) predicts P3b amplitude trial-by-trial, and
detection rate is higher at diastole than systole.

Run:
    python figures/generate_figure3.py
    python figures/generate_figure3.py --no-show   # CI mode
"""

import argparse
import pathlib
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from apgi.core import (
    BETA_SM_DEFAULT,
    compute_pi_i_eff,  # noqa: E402
    compute_S_t,
    ignition_criterion,
    theta_equilibrium,
)
from figures.utils import (
    HALF_WIDTH,
    PALETTE,
    PANEL_HEIGHT,  # noqa: E402
    label_axes,
    make_figure,
    save_figure,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# Protocol 1 APGI parameters (from protocol_1_cardiac_eeg.json)
ALPHA = 0.3
BETA = 0.7
PI_I_SYSTOLE = 0.6
PI_I_DIASTOLE = 1.2
N_TRIALS = 480


def simulate(seed: int = 2025) -> dict:
    rng = np.random.default_rng(seed)

    cardiac_phase = np.array(["systole", "diastole"] * (N_TRIALS // 2))
    rng.shuffle(cardiac_phase)

    pi_i = np.where(cardiac_phase == "systole", PI_I_SYSTOLE, PI_I_DIASTOLE)
    C_metabolic = rng.uniform(0.5, 2.0, N_TRIALS)

    # HEP amplitude is a noisy proxy for Πⁱ_eff (Pred 1.a); compute per-trial (pi_i varies by phase)
    pi_i_eff = np.array(
        [
            compute_pi_i_eff(float(pi_i[i]), BETA_SM_DEFAULT, 0.0)
            for i in range(N_TRIALS)
        ]
    )
    hep_amplitude = pi_i_eff + rng.normal(0, 0.08, N_TRIALS)

    pi_e = rng.uniform(0.8, 1.5, N_TRIALS)
    z_e = rng.uniform(0.2, 1.0, N_TRIALS)
    z_i = rng.uniform(0.1, 0.8, N_TRIALS)
    V_info = rng.uniform(0.1, 1.0, N_TRIALS)

    S_t = np.array(
        [
            compute_S_t(
                float(pi_e[i]), float(z_e[i]), float(pi_i_eff[i]), float(z_i[i])
            )
            for i in range(N_TRIALS)
        ]
    )
    theta_t = np.array(
        [
            theta_equilibrium(
                float(C_metabolic[i]),
                float(V_info[i]),
                lambda_theta=ALPHA,
                kappa_meta=BETA,
            )
            for i in range(N_TRIALS)
        ]
    )
    detected = np.array(
        [ignition_criterion(S_t[i], theta_t[i]) for i in range(N_TRIALS)]
    )

    # P3b amplitude scales with S_t when detected (ignition proxy)
    p3b_amplitude = S_t * detected + rng.normal(0, 0.05, N_TRIALS)

    return {
        "cardiac_phase": cardiac_phase,
        "hep_amplitude": hep_amplitude,
        "p3b_amplitude": p3b_amplitude,
        "S_t": S_t,
        "detected": detected,
    }


def plot(data: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=3, width=HALF_WIDTH * 3, height=PANEL_HEIGHT)

    # Panel A: Detection rate by cardiac phase
    ax = axes[0]
    mask_sys = data["cardiac_phase"] == "systole"
    mask_dia = data["cardiac_phase"] == "diastole"
    rates = [data["detected"][mask_sys].mean(), data["detected"][mask_dia].mean()]
    colors = [PALETTE["theta"], PALETTE["S_t"]]
    bars = ax.bar(
        ["Systole", "Diastole"],
        rates,
        color=colors,
        alpha=0.85,
        edgecolor="white",
        width=0.5,
    )
    ax.set_ylabel("Detection rate", fontsize=10)
    ax.set_title("Pred 1.b — Cardiac phase\n× detection rate", fontsize=10)
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.02,
            f"{rate:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel B: HEP–P3b coupling (Pred 1.a)
    ax = axes[1]
    r, _ = pearsonr(data["hep_amplitude"], data["p3b_amplitude"])
    ax.scatter(
        data["hep_amplitude"],
        data["p3b_amplitude"],
        s=8,
        alpha=0.4,
        color=PALETTE["S_t"],
        edgecolors="none",
    )
    m, b = np.polyfit(data["hep_amplitude"], data["p3b_amplitude"], 1)
    x_line = np.linspace(data["hep_amplitude"].min(), data["hep_amplitude"].max(), 100)
    ax.plot(x_line, m * x_line + b, color=PALETTE["theta"], lw=1.5)
    ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9)
    ax.set_xlabel(r"HEP amplitude ($\Pi^i$ proxy)", fontsize=9)
    ax.set_ylabel(r"P3b amplitude (ignition proxy)", fontsize=9)
    ax.set_title("Pred 1.a — HEP → P3b\ncoupling", fontsize=10)

    # Panel C: S_t distributions by phase
    ax = axes[2]
    ax.hist(
        data["S_t"][mask_sys],
        bins=25,
        color=PALETTE["theta"],
        alpha=0.7,
        edgecolor="white",
        label="Systole",
        density=True,
    )
    ax.hist(
        data["S_t"][mask_dia],
        bins=25,
        color=PALETTE["S_t"],
        alpha=0.7,
        edgecolor="white",
        label="Diastole",
        density=True,
    )
    ax.set_xlabel(r"$S_t$", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(r"$S_t$ distribution" + "\nby cardiac phase", fontsize=10)
    ax.legend(fontsize=8)

    label_axes(axes)
    fig.suptitle("Figure 3 — Protocol 1 — Cardiac-EEG", fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure3.pdf")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 3")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    plot(simulate(), show=not args.no_show)


if __name__ == "__main__":
    main()
