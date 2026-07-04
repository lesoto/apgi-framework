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
# Derived from the spec's canonical geometric series (§2.4): tau_0 = 50 ms,
# common ratio ≈ 133 per step (τ ≈ 50 ms, 6.6 s, 15 min, 1.4 d, 6 mo).
# Values below are computed from the ratio for internal self-consistency;
# they land close to the spec's stated approximations.
_COMMON_RATIO = 133.0
_TAU_L0 = 0.050
LEVEL_TAUS = {
    "L0": _TAU_L0,
    "L1": _TAU_L0 * _COMMON_RATIO,
    "L2": _TAU_L0 * _COMMON_RATIO**2,
    "L3": _TAU_L0 * _COMMON_RATIO**3,
    "L4": _TAU_L0 * _COMMON_RATIO**4,
}
LEVEL_COLORS = {
    "L0": "#333333",
    "L1": "#fc8d59",
    "L2": "#d6604d",
    "L3": "#2166ac",
    "L4": "#08519c",
}


def lorentzian(f, tau, sigma2=1.0):
    """Single-level OU spectrum per spec §2.4: S_l(f) = sigma^2 * tau / (1 + (2*pi*f*tau)^2).

    Amplitude scales with tau (not normalized to a unit peak) so that, with
    roughly equal sigma_l across levels, the superposition of geometrically
    spaced poles approximates a 1/f spectrum (log-uniform pole density).
    """
    return sigma2 * tau / (1 + (2 * np.pi * f * tau) ** 2)


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

    # Crossover annotations — corner frequencies computed programmatically
    # from LEVEL_TAUS (f_c = 1/(2*pi*tau)) so labels always match the curves.
    def _fmt_freq(fc: float) -> str:
        if fc >= 1:
            return f"~{fc:.3g} Hz"
        return f"~{fc:.2g} Hz"

    # Per spec (§2.4 "Crossover frequencies"): f_c,ℓ = 1/(2*pi*tau_ℓ), i.e. each
    # level-boundary crossover is evaluated at the faster (shorter-tau) of the
    # two adjacent levels' own time constants — giving ~3.2 Hz at L0/L1, etc.
    level_order = ["L0", "L1", "L2", "L3", "L4"]
    crossovers = []
    for lo, hi in zip(level_order[:-1], level_order[1:]):
        fc = 1.0 / (2 * np.pi * LEVEL_TAUS[lo])
        crossovers.append((fc, f"{_fmt_freq(fc)}\n{lo}/{hi}"))
    y_top = S_total.max() * 3.0
    for fc, label in crossovers:
        ax.axvline(fc, lw=1.2, color="#555555", linestyle="--", alpha=0.6)
        ax.text(
            fc * 1.1,
            y_top,
            label,
            fontsize=7,
            color="#555555",
            va="top",
        )

    # Mid-band slope annotation — actual least-squares fit of log(S_total)
    # vs log(f), not a hardcoded guess. Fit window spans the mid-band between
    # the L0/L1 and L3/L4 corner frequencies (~7 decades, per the spec's
    # "least-squares slope ≈ -1.0 over ~7 decades" claim).
    fc_lo = crossovers[0][0]  # L0/L1 corner (highest frequency)
    fc_hi = crossovers[-1][0]  # L3/L4 corner (lowest frequency)
    fit_mask = (f <= fc_lo) & (f >= fc_hi)
    log_f_fit = np.log10(f[fit_mask])
    log_S_fit = np.log10(S_total[fit_mask])
    slope_total, intercept_total = np.polyfit(log_f_fit, log_S_fit, 1)
    # fGn convention: S(f) ~ 1/f^beta, beta = 1 - slope (slope is negative);
    # H = (beta + 1) / 2 for fGn (per spec §2.4 "Conventions").
    beta_total = -slope_total
    H_total = (beta_total + 1) / 2

    log_S_partial_fit = np.log10(S_partial[fit_mask])
    slope_partial, _ = np.polyfit(log_f_fit, log_S_partial_fit, 1)
    beta_partial = -slope_partial
    H_partial = (beta_partial + 1) / 2

    # Place annotations near the fitted line itself: pick a mid-band anchor
    # frequency and evaluate the actual curve value there.
    f_anchor = np.sqrt(fc_lo * fc_hi)  # geometric-mean mid-band frequency
    S_anchor_total = S_total[np.argmin(np.abs(f - f_anchor))]
    S_anchor_partial = S_partial[np.argmin(np.abs(f - f_anchor))]
    ax.text(
        f_anchor,
        S_anchor_total * 10,
        f"fitted slope = {slope_total:.2f}"
        + "\n"
        + rf"$H \approx {H_total:.2f}$ (full hierarchy)",
        fontsize=9,
        color="#333333",
        fontweight="bold",
        ha="center",
    )
    ax.text(
        f_anchor,
        S_anchor_partial * 0.05,
        f"fitted slope = {slope_partial:.2f}"
        + "\n"
        + rf"$H \approx {H_partial:.2f}$ (L0–L2)",
        fontsize=8.5,
        color="#666666",
        ha="center",
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
        "Simulation parameters: τ ≈ 50 ms, 6.6 s, 15 min, 1.4 days, 6 months "
        "(common ratio ≈ 133); σ_ℓ equal; w_ℓ equal weights. "
        "Code available in project repository (Data Availability)."
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
