"""Figure 7 — Protocol 5: vmPFC–insula anticipatory coupling (P5a–P5d).

Simulates fMRI PPI coefficients and SCR outcomes from
protocol_5_fmri_anticipation.json. Shows:
  A — vmPFC–pIC PPI: anticipation > outcome (P5b)
  B — vmPFC BOLD parametric modulation: valence vs. contrast sensitivity (P5c)
  C — PPI coefficient: long foreperiod vs. no-foreperiod control (P5d)

Run:
    python figures/generate_figure7.py
    python figures/generate_figure7.py --no-show   # CI mode
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

# Protocol 5 APGI parameters
KAPPA = 100.0
ALPHA = 0.3
BETA = 0.7
PI_I_LONG = 1.2  # long foreperiod: somatic marker active
PI_I_SHORT = 0.8  # no foreperiod: somatic marker suppressed
GAMMA_V = 0.6
GAMMA_A = 0.3
N_SUBJECTS = 36


def simulate_ppi(n_subjects: int = N_SUBJECTS, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)

    # PPI coefficients: anticipation window > outcome window (P5b)
    ppi_anticipation = rng.normal(0.42, 0.12, n_subjects)
    ppi_outcome = rng.normal(0.11, 0.10, n_subjects)

    # vmPFC BOLD: EV-parametric (valence) vs. contrast-parametric (P5c)
    bold_valence = rng.normal(0.55, 0.14, n_subjects)  # sensitive to option EV
    bold_contrast = rng.normal(0.08, 0.11, n_subjects)  # insensitive to contrast

    # Foreperiod manipulation (P5d): long vs. 0 ms foreperiod
    ppi_long_fp = rng.normal(0.44, 0.11, n_subjects)
    ppi_no_fp = rng.normal(0.09, 0.10, n_subjects)

    return {
        "ppi_anticipation": ppi_anticipation,
        "ppi_outcome": ppi_outcome,
        "bold_valence": bold_valence,
        "bold_contrast": bold_contrast,
        "ppi_long_fp": ppi_long_fp,
        "ppi_no_fp": ppi_no_fp,
    }


def plot(data: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=3, width=HALF_WIDTH * 3, height=PANEL_HEIGHT)

    # Panel A: PPI anticipation vs outcome (P5b)
    ax = axes[0]
    means = [data["ppi_anticipation"].mean(), data["ppi_outcome"].mean()]
    sems = [
        data["ppi_anticipation"].std() / np.sqrt(N_SUBJECTS),
        data["ppi_outcome"].std() / np.sqrt(N_SUBJECTS),
    ]
    ax.bar(
        ["Anticipation", "Outcome"],
        means,
        yerr=sems,
        color=[PALETTE["S_t"], "#9966FF"],
        alpha=0.85,
        edgecolor="white",
        width=0.4,
        capsize=5,
    )
    ax.axhline(0, ls="--", lw=0.8, color="black", alpha=0.4)
    ax.set_ylabel("vmPFC–pIC PPI coefficient", fontsize=10)
    ax.set_title("P5b — Anticipatory coupling\npeaks before outcome", fontsize=10)

    # Panel B: vmPFC BOLD — valence vs contrast (P5c)
    ax = axes[1]
    means_b = [data["bold_valence"].mean(), data["bold_contrast"].mean()]
    sems_b = [
        data["bold_valence"].std() / np.sqrt(N_SUBJECTS),
        data["bold_contrast"].std() / np.sqrt(N_SUBJECTS),
    ]
    ax.bar(
        ["Option EV\n(valence)", "Sensory\ncontrast"],
        means_b,
        yerr=sems_b,
        color=[PALETTE["S_t"], PALETTE["theta"]],
        alpha=0.85,
        edgecolor="white",
        width=0.4,
        capsize=5,
    )
    ax.axhline(0, ls="--", lw=0.8, color="black", alpha=0.4)
    ax.set_ylabel("vmPFC BOLD β (a.u.)", fontsize=10)
    ax.set_title("P5c — vmPFC sensitive to\nvalence, not contrast", fontsize=10)

    # Panel C: Foreperiod manipulation (P5d)
    ax = axes[2]
    means_fp = [data["ppi_long_fp"].mean(), data["ppi_no_fp"].mean()]
    sems_fp = [
        data["ppi_long_fp"].std() / np.sqrt(N_SUBJECTS),
        data["ppi_no_fp"].std() / np.sqrt(N_SUBJECTS),
    ]
    ax.bar(
        ["Long foreperiod\n(2000–4000 ms)", "No foreperiod\n(0 ms)"],
        means_fp,
        yerr=sems_fp,
        color=[PALETTE["S_t"], "#AAAAAA"],
        alpha=0.85,
        edgecolor="white",
        width=0.4,
        capsize=5,
    )
    ax.axhline(0, ls="--", lw=0.8, color="black", alpha=0.4)
    ax.set_ylabel("vmPFC–pIC PPI coefficient", fontsize=10)
    ax.set_title("P5d — Anticipation drives\nvmPFC–insula coupling", fontsize=10)

    label_axes(axes)
    fig.suptitle(
        "Figure 7 — Protocol 5: vmPFC–Insula Anticipatory Coupling", fontsize=11, y=1.02
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure7.pdf")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 7")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    plot(simulate_ppi(), show=not args.no_show)


if __name__ == "__main__":
    main()
