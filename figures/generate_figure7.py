"""Figure 7 — Protocol 5 — Ignition-iEEG: iEEG all-or-none ignition dynamics (Pred 5.a–Pred 5.d).

Simulates intracranial EEG predictions from protocol_5_ignition_ieeg.json:
  A — Bimodal high-gamma distribution in frontoparietal electrodes at threshold (Pred 5.a)
  B — Regional specificity: frontoparietal bimodal vs. occipital graded (Pred 5.b)
  C — AC1 pre-ignition slowing: detected vs. non-detected trials (Pred 5.c — bifurcation criterion)
  D — Long-range gamma coherence (15–80 Hz) predicts detection (Pred 5.d — three-tier confirmation):
      Criteria (1+2) = GNW-consistent; criteria (1+2+3) with HEP–coherence r > 0.25 = APGI-specific.

Pred 5.c is the bifurcation falsification criterion distinguishing APGI from standard GWT.
Pred 5.d is the APGI-specific extension via HEP–coherence coupling (criterion 3).

Run:
    python figures/generate_figure7.py
    python figures/generate_figure7.py --no-show   # CI mode
"""

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))

import argparse
import pathlib

import numpy as np

from figures.utils import (
    HALF_WIDTH,
    PALETTE,
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

    # --- Frontoparietal: bimodal at near-threshold contrast (Pred 5.a)
    # Detected trials cluster at high gamma, non-detected at low gamma
    n_detected = N_NEAR_THRESHOLD // 2
    n_missed = N_NEAR_THRESHOLD - n_detected
    hg_detected = rng.normal(0.72, 0.08, n_detected)  # high state
    hg_missed = rng.normal(0.21, 0.07, n_missed)  # low state
    hg_fp = np.concatenate([hg_detected, hg_missed])  # bimodal

    # --- Occipital: graded (unimodal) — Pred 5.b
    hg_occ = rng.normal(0.45, 0.18, N_NEAR_THRESHOLD)  # unimodal

    # --- AC1 pre-ignition slowing (Pred 5.c)
    # Detected trials: AC1 increases in 500 ms window before ignition
    # Non-detected: AC1 flat
    time_bins = np.linspace(-500, 0, AC1_WINDOW_BINS)
    ac1_detected = (
        0.20
        + 0.035 * np.arange(AC1_WINDOW_BINS)
        + rng.normal(0, 0.025, AC1_WINDOW_BINS)
    )
    ac1_missed = 0.22 + rng.normal(0, 0.025, AC1_WINDOW_BINS)

    # --- Pred 5.d: Long-range frontoparietal gamma coherence (15–80 Hz) predicts detection
    # Criterion 1: frontoparietal coherence r > 0.4, peaking 200–400 ms
    # Criterion 2: occipital coherence r < 0.20 (frontoparietal specificity)
    # Criterion 3 (APGI-specific): HEP–coherence r > 0.25 in seen trials
    time_post = np.linspace(-100, 800, 90)  # ms post-stimulus
    # Frontoparietal coherence: rises ~200 ms, peaks 200–400 ms in detected trials
    coh_fp_seen = np.where(
        (time_post >= 200) & (time_post <= 400),
        0.55 + rng.normal(0, 0.04, 90),
        0.25 + rng.normal(0, 0.04, 90),
    )
    coh_fp_unseen = 0.24 + rng.normal(0, 0.04, 90)
    # Occipital coherence: flat (< 0.20) — criterion 2 specificity
    coh_occ_seen = 0.14 + rng.normal(0, 0.03, 90)

    # HEP–coherence correlation in seen trials (criterion 3, APGI-specific)
    n_seen = n_detected
    hep_seen = rng.normal(0.7, 0.15, n_seen)
    coherence_peak_seen = 0.28 * hep_seen + rng.normal(0, 0.06, n_seen)

    return {
        "hg_fp": hg_fp,
        "hg_occ": hg_occ,
        "hg_detected": hg_detected,
        "hg_missed": hg_missed,
        "time_bins": time_bins,
        "ac1_detected": np.clip(ac1_detected, 0, 1),
        "ac1_missed": np.clip(ac1_missed, 0, 1),
        "time_post": time_post,
        "coh_fp_seen": coh_fp_seen,
        "coh_fp_unseen": coh_fp_unseen,
        "coh_occ_seen": coh_occ_seen,
        "hep_seen": hep_seen,
        "coherence_peak_seen": coherence_peak_seen,
    }


def plot(data: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=4, width=HALF_WIDTH * 4, height=PANEL_HEIGHT)

    # Panel A: Bimodal frontoparietal high-gamma (Pred 5.a)
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
    ax.set_title("Pred 5.a — Frontoparietal\nbimodal distribution", fontsize=10)
    ax.legend(fontsize=7)

    # Panel B: Regional specificity — fp bimodal vs occipital graded (Pred 5.b)
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
        "Pred 5.b — Regional specificity:\nFP bimodal, occipital graded", fontsize=10
    )
    ax.legend(fontsize=7)

    # Panel C: AC1 pre-ignition slowing (Pred 5.c — bifurcation criterion)
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
        "Pred 5.c — Critical slowing:\nAC1 ↑ before ignition\n(APGI vs. GWT criterion)",
        fontsize=10,
    )
    ax.legend(fontsize=7)

    # Panel D: Long-range gamma coherence predicts detection (Pred 5.d — three-tier)
    ax = axes[3]
    ax.plot(
        data["time_post"],
        data["coh_fp_seen"],
        "-",
        color=PALETTE["S_t"],
        lw=1.8,
        label="FP coherence — seen (crit 1)",
    )
    ax.plot(
        data["time_post"],
        data["coh_fp_unseen"],
        "--",
        color=PALETTE["theta"],
        lw=1.5,
        alpha=0.8,
        label="FP coherence — unseen",
    )
    ax.plot(
        data["time_post"],
        data["coh_occ_seen"],
        ":",
        color="#AAAAAA",
        lw=1.5,
        label="Occipital — seen (crit 2 ↓)",
    )
    ax.axvspan(
        200, 400, alpha=0.10, color=PALETTE["S_t"], label="Peak window 200–400 ms"
    )
    ax.axhline(
        0.4, ls="--", lw=0.9, color=PALETTE["S_t"], alpha=0.5, label="r > 0.4 threshold"
    )
    ax.axhline(
        0.20, ls="--", lw=0.9, color="#AAAAAA", alpha=0.5, label="r < 0.20 (occipital)"
    )
    ax.set_xlabel("Time post-stimulus (ms)", fontsize=10)
    ax.set_ylabel("Coherence (15–80 Hz)", fontsize=10)
    ax.set_title(
        "Pred 5.d — Gamma coherence predicts detection\n(1+2=GNW; +HEP–coh r>0.25=APGI)",
        fontsize=9,
    )
    ax.legend(fontsize=6)

    label_axes(axes)
    fig.suptitle(
        "Figure 7 — Protocol 5 — Ignition-iEEG: iEEG All-or-None Ignition Dynamics (Pred 5.a–Pred 5.d)",
        fontsize=11,
        y=1.02,
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
    plot(simulate_ieeg(), show=not args.no_show)


if __name__ == "__main__":
    main()
