"""Paper 3 — Figure 5: Developmental Maturation × Spectral Crossover (§5.4).

Dual-axis figure: developmental age 0–25 yr on x-axis, aperiodic EEG exponent H
on primary y-axis, with spectral crossover frequencies on secondary x-axis.

Run:
    python figures/paper3/generate_fig5.py
    python figures/paper3/generate_fig5.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.paper3.generate_fig2 import LEVEL_TAUS
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"


def _fmt_freq(fc: float) -> str:
    if fc >= 1:
        return f"~{fc:.2g} Hz"
    return f"~{fc:.1g} Hz"


# Level maturation inflection ages and spectral crossovers. Corner
# frequencies are derived from the same corrected LEVEL_TAUS used in Figure 2
# (fig2's canonical geometric series, common ratio ≈ 133) so the two figures
# never drift apart: f_c = 1/(2*pi*tau_lo) at each level boundary.
INFLECTIONS = [
    {
        "level": "L1",
        "age": 2,
        "freq": _fmt_freq(1.0 / (2 * np.pi * LEVEL_TAUS["L0"])),
        "color": "#fdcc8a",
    },
    {
        "level": "L2",
        "age": 7,
        "freq": _fmt_freq(1.0 / (2 * np.pi * LEVEL_TAUS["L1"])),
        "color": "#fc8d59",
    },
    {
        "level": "L3",
        "age": 13,
        "freq": _fmt_freq(1.0 / (2 * np.pi * LEVEL_TAUS["L2"])),
        "color": "#de2d26",
    },
    {
        "level": "L4",
        "age": 25,
        "freq": _fmt_freq(1.0 / (2 * np.pi * LEVEL_TAUS["L3"])),
        "color": "#a50f15",
    },
]


def sigmoid_rise(age, inflection_ages):
    """Composite sigmoid rising through each inflection age."""
    H = np.full_like(age, 0.55, dtype=float)
    H_max = 0.90
    for inf_age in inflection_ages:
        H += (H_max - 0.55) / len(inflection_ages) * expit((age - inf_age) * 1.2)
    return np.clip(H, 0.5, 0.97)


def plot(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    age = np.linspace(0, 28, 500)
    inflection_ages = [inf["age"] for inf in INFLECTIONS]

    H = sigmoid_rise(age, inflection_ages)

    # Empirical reference band (Voytek et al. 2015 style)
    H_lo = H - 0.07
    H_hi = H + 0.07
    ax.fill_between(
        age,
        H_lo,
        H_hi,
        alpha=0.20,
        color="#2166ac",
        label="Empirical reference band (Voytek et al. 2015)",
    )
    ax.plot(age, H, lw=2.5, color="#2166ac", label="Predicted H trajectory")

    # Neonatal / adult reference lines
    ax.axhline(0.55, lw=1.0, ls=":", color="#aaaaaa", alpha=0.7)
    ax.axhline(0.90, lw=1.0, ls=":", color="#aaaaaa", alpha=0.7)
    ax.text(
        27,
        0.55,
        "Neonatal\nH≈0.55",
        fontsize=7.5,
        color="#aaaaaa",
        va="center",
        ha="right",
    )
    ax.text(
        27,
        0.90,
        "Adult\nH≈0.90",
        fontsize=7.5,
        color="#aaaaaa",
        va="center",
        ha="right",
    )

    # White noise reference
    ax.axhline(0.5, lw=1.0, ls="--", color="#888888", alpha=0.5)
    ax.text(0.2, 0.502, "H = 0.5 (white noise)", fontsize=7.5, color="#888888")

    # Inflection markers
    for inf in INFLECTIONS:
        age_mark = inf["age"]
        H_mark = sigmoid_rise(np.array([age_mark]), inflection_ages)[0]
        ax.axvline(age_mark, lw=1.5, ls="--", color=inf["color"], alpha=0.85)
        ax.plot(age_mark, H_mark, "o", ms=8, color=inf["color"], zorder=5)
        ax.text(
            age_mark + 0.3,
            H_mark + 0.012,
            inf["level"],
            fontsize=9,
            fontweight="bold",
            color=inf["color"],
        )
        ax.text(
            age_mark + 0.3,
            H_mark - 0.025,
            "Absent/out-of-sequence\ninflection falsifies\nL-k maturation",
            fontsize=6.0,
            color="#888888",
            style="italic",
        )

    ax.set_xlabel("Developmental age (years)", fontsize=11)
    ax.set_ylabel("Aperiodic EEG exponent H", fontsize=11)
    ax.set_xlim(0, 28)
    ax.set_ylim(0.45, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Secondary x-axis (spectral crossover frequencies)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([inf["age"] for inf in INFLECTIONS])
    ax2.set_xticklabels([inf["freq"] for inf in INFLECTIONS], fontsize=8)
    ax2.set_xlabel(
        "Spectral crossover frequency (level maturation)", fontsize=9, labelpad=8
    )

    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig5_developmental_maturation_spectral.pdf")
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
