"""Paper 3 — Figure 2: 1/f Power Spectrum from OU Superposition (§2.4).

Log-log PSD plot with five Lorentzian curves (one per level),
composite S_total(f), partial-hierarchy overlay (L0–L2), and
annotated crossover frequencies and Hurst exponent ranges.

Run:
    python figures/paper3/generate_fig2.py
    python figures/paper3/generate_fig2.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# Level timescales (seconds) → corner frequencies f_c = 1/(2π τ)
LEVEL_TAUS = {
    "L0": 0.050,
    "L1": 0.500,
    "L2": 15.0,
    "L3": 12 * 3600,
    "L4": 6 * 30 * 24 * 3600,
}
LEVEL_COLORS = {
    "L0": "#333333",
    "L1": "#fc8d59",
    "L2": "#d6604d",
    "L3": "#2166ac",
    "L4": "#08519c",
}


def lorentzian(f, tau, sigma2=1.0):
    fc = 1.0 / (2 * np.pi * tau)
    return sigma2 / (1 + (f / fc) ** 2)


def plot(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    f = np.logspace(-6, 2, 2000)
    S_total = np.zeros_like(f)
    S_partial = np.zeros_like(f)

    for name, tau in LEVEL_TAUS.items():
        S_lev = lorentzian(f, tau)
        S_total += S_lev
        if name in ("L0", "L1", "L2"):
            S_partial += S_lev
        ax.loglog(
            f,
            S_lev,
            lw=1.2,
            color=LEVEL_COLORS[name],
            linestyle="--",
            alpha=0.7,
            label=f"{name} (τ≈{tau:.3g} s)",
        )
        # Corner freq label
        fc = 1 / (2 * np.pi * tau)
        ax.axvline(fc, lw=0.7, color=LEVEL_COLORS[name], alpha=0.35, ls=":")

    # Composite (full hierarchy)
    ax.loglog(
        f,
        S_total,
        lw=2.5,
        color="#333333",
        label=r"$S_{\mathrm{total}}(f)$ — full (H ≈ 0.85–0.95)",
    )

    # Partial hierarchy (L0–L2 only)
    ax.loglog(
        f,
        S_partial,
        lw=2.0,
        color="#999999",
        linestyle="-.",
        label="L0–L2 only — EEG amplitude-envelope (H ≈ 0.65–0.75)",
    )

    # White noise baseline
    ax.axhline(1e-3, lw=1.0, ls=":", color="#aaaaaa", alpha=0.7)
    ax.text(
        1e-6, 1.2e-3, "White noise baseline (H = 0.5)", fontsize=7.5, color="#aaaaaa"
    )

    # Crossover annotations
    crossovers = [
        (10, "~10 Hz\nL0/L1"),
        (0.6, "~0.6 Hz\nL1/L2"),
        (0.01, "~0.01 Hz\nL2/L3"),
        (1e-4, "~10⁻⁴ Hz\nL3/L4"),
    ]
    for fc, label in crossovers:
        ax.axvline(fc, lw=1.2, color="#555555", linestyle="--", alpha=0.6)
        ax.text(
            fc * 1.1,
            S_total[np.argmin(np.abs(f - fc))] * 1.5,
            label,
            fontsize=7,
            color="#555555",
            va="bottom",
        )

    # Mid-band slope annotation
    ax.text(
        1e-3,
        50,
        r"$H \approx 0.85{-}0.95$" + "\n(full hierarchy)",
        fontsize=9,
        color="#333333",
        fontweight="bold",
    )
    ax.text(
        3e-3, 3, r"$H \approx 0.65{-}0.75$" + "\n(L0–L2)", fontsize=8.5, color="#666666"
    )

    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.set_ylabel("Power Spectral Density (a.u.)", fontsize=11)
    ax.set_title(
        "Figure 2 — 1/f Power Spectrum from OU Superposition\n"
        "Five Lorentzians + Composite + Partial-Hierarchy Overlay (§2.4, Paper 3)",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower left", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    caption = (
        "Simulation parameters: τ ≈ 50 ms, 500 ms, 15 min, 12 h, 6 months; "
        "σ_ℓ equal; w_ℓ equal weights. Code available in project repository (Data Availability)."
    )
    fig.text(
        0.5, -0.03, caption, ha="center", fontsize=7.5, color="#666666", style="italic"
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig2_1f_power_spectrum.pdf")
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
