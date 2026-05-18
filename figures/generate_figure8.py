"""Figure 8 — Protocol 6: iEEG all-or-none ignition dynamics (P6a–P6c).

Simulates intracranial EEG predictions from protocol_6_icEEG_ignition_dynamics.json:
  A — Bimodal high-gamma distribution in frontoparietal electrodes at threshold (P6a)
  B — Regional specificity: frontoparietal bimodal vs. occipital graded (P6b)
  C — AC1 pre-ignition slowing: detected vs. non-detected trials (P6c — bifurcation criterion)

P6c is the critical prediction distinguishing APGI from standard GWT.

Run:
    python figures/generate_figure8.py
    python figures/generate_figure8.py --no-show   # CI mode
"""

import sys as _sys
import pathlib as _pathlib

_sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))

import argparse
import pathlib

import numpy as np

from apgi.core import compute_pi_i_eff, compute_S_t, compute_theta_t, ignition_criterion
from figures.utils import (
    PALETTE,
    HALF_WIDTH,
    PANEL_HEIGHT,
    label_axes,
    make_figure,
    save_figure,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

N_TRIALS = 300
N_NEAR_THRESHOLD = 150  # 50% near-threshold per protocol
AC1_WINDOW_BINS = 10  # 10 × 50 ms windows in 500 ms pre-ignition epoch


def simulate_ieeg(seed: int = 9) -> dict:
    rng = np.random.default_rng(seed)

    # --- Frontoparietal: bimodal at near-threshold contrast (P6a)
    # Detected trials cluster at high gamma, non-detected at low gamma
    n_detected = N_NEAR_THRESHOLD // 2
    n_missed = N_NEAR_THRESHOLD - n_detected
    hg_detected = rng.normal(0.72, 0.08, n_detected)  # high state
    hg_missed = rng.normal(0.21, 0.07, n_missed)  # low state
    hg_fp = np.concatenate([hg_detected, hg_missed])  # bimodal

    # --- Occipital: graded (unimodal) — P6b
    hg_occ = rng.normal(0.45, 0.18, N_NEAR_THRESHOLD)  # unimodal

    # --- AC1 pre-ignition slowing (P6c)
    # Detected trials: AC1 increases in 500 ms window before ignition
    # Non-detected: AC1 flat
    time_bins = np.linspace(-500, 0, AC1_WINDOW_BINS)
    ac1_detected = (
        0.20
        + 0.035 * np.arange(AC1_WINDOW_BINS)
        + rng.normal(0, 0.025, AC1_WINDOW_BINS)
    )
    ac1_missed = 0.22 + rng.normal(0, 0.025, AC1_WINDOW_BINS)

    return {
        "hg_fp": hg_fp,
        "hg_occ": hg_occ,
        "hg_detected": hg_detected,
        "hg_missed": hg_missed,
        "time_bins": time_bins,
        "ac1_detected": np.clip(ac1_detected, 0, 1),
        "ac1_missed": np.clip(ac1_missed, 0, 1),
    }


def plot(data: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=3, width=HALF_WIDTH * 3, height=PANEL_HEIGHT)

    # Panel A: Bimodal frontoparietal high-gamma (P6a)
    ax = axes[0]
    ax.hist(
        data["hg_detected"],
        bins=20,
        color=PALETTE["S_t"],
        alpha=0.75,
        edgecolor="white",
        label="Detected (high state)",
        density=True,
    )
    ax.hist(
        data["hg_missed"],
        bins=20,
        color=PALETTE["theta"],
        alpha=0.75,
        edgecolor="white",
        label="Non-detected (low state)",
        density=True,
    )
    ax.set_xlabel("High-gamma amplitude (a.u.)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("P6a — Frontoparietal\nbimodal distribution", fontsize=10)
    ax.legend(fontsize=7)

    # Panel B: Regional specificity — fp bimodal vs occipital graded (P6b)
    ax = axes[1]
    ax.hist(
        data["hg_fp"],
        bins=22,
        color=PALETTE["S_t"],
        alpha=0.75,
        edgecolor="white",
        label="Frontoparietal (bimodal)",
        density=True,
    )
    ax.hist(
        data["hg_occ"],
        bins=22,
        color="#AAAAAA",
        alpha=0.65,
        edgecolor="white",
        label="Occipital (graded)",
        density=True,
    )
    ax.set_xlabel("High-gamma amplitude (a.u.)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        "P6b — Regional specificity:\nFP bimodal, occipital graded", fontsize=10
    )
    ax.legend(fontsize=7)

    # Panel C: AC1 pre-ignition slowing (P6c — bifurcation criterion)
    ax = axes[2]
    ax.plot(
        data["time_bins"],
        data["ac1_detected"],
        "o-",
        color=PALETTE["S_t"],
        lw=1.8,
        ms=5,
        label="Detected (ignition)",
    )
    ax.plot(
        data["time_bins"],
        data["ac1_missed"],
        "s--",
        color=PALETTE["theta"],
        lw=1.5,
        ms=4,
        alpha=0.8,
        label="Non-detected",
    )
    ax.set_xlabel("Time before ignition (ms)", fontsize=10)
    ax.set_ylabel("AC1 (lag-1 autocorrelation)", fontsize=10)
    ax.set_title(
        "P6c — Critical slowing:\nAC1 ↑ before ignition\n(APGI vs. GWT criterion)",
        fontsize=10,
    )
    ax.legend(fontsize=7)

    label_axes(axes)
    fig.suptitle(
        "Figure 8 — Protocol 6: iEEG All-or-None Ignition Dynamics", fontsize=11, y=1.02
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure8.pdf")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 8")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    plot(simulate_ieeg(), show=not args.no_show)


if __name__ == "__main__":
    main()
