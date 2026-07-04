"""Figure 5 — Protocol 4 (Metabolic-State Crossover): predicted allostatic
elevation of the ignition threshold under metabolic depletion (Pred 4.A-Pred 4.C).

Per OUP-Protocols.txt Figure 5 caption:
  (A) Perceptual sensitivity d' plotted as a Metabolic State (rested/fed vs
      depleted) x Interoceptive Load (neutral/low vs high) interaction:
      depletion selectively reduces d' for high-interoceptive-load stimuli
      while sparing neutral exteroceptive stimuli (Pred 4.A).
  (B) The same interaction on P3b amplitude, the neural ignition proxy --
      disproportionate suppression of interoceptive targets (Pred 4.B).
  (C) The interaction survives covariation for trial-level pupil diameter
      and RMSSD, dissociating metabolic allostatic triage (elevated
      theta_t) from generalised LC-NE arousal/fatigue (Pred 4.C).

No prior generate_figureN.py script implemented Protocol 4 -- this is a new
figure written to close that gap, using the archived seed dataset
data/seeds/sim7_metabolic_crossover.npz (2x2 within-subject design,
N=60 subjects across 2 sites) rather than inventing data inline.

Run:
    python figures/generate_figure5.py
    python figures/generate_figure5.py --no-show   # CI mode
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from figures.utils import (  # noqa: E402
    HALF_WIDTH,
    PALETTE,
    PANEL_HEIGHT,
    label_axes,
    make_figure,
    save_figure,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
DATA_DIR = pathlib.Path(
    os.environ.get("APGI_DATA_DIR", pathlib.Path(__file__).resolve().parent.parent / "data" / "seeds")
)

STATE_ORDER = ["fed", "depleted"]
LOAD_ORDER = ["low", "high"]
STATE_LABELS = {"fed": "Fed / rested", "depleted": "Depleted / fatigued"}
LOAD_LABELS = {"low": "Low\n(exteroceptive)", "high": "High\n(interoceptive)"}


def load_data(path: pathlib.Path | None = None) -> dict:
    npz_path = path or (DATA_DIR / "sim7_metabolic_crossover.npz")
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def cell_stats(data: dict, var: str) -> dict:
    ms, il = data["metabolic_state"], data["interoceptive_load"]
    values = data[var]
    stats = {}
    for state in STATE_ORDER:
        for load in LOAD_ORDER:
            mask = (ms == state) & (il == load)
            v = values[mask]
            stats[(state, load)] = (float(v.mean()), float(v.std() / np.sqrt(max(mask.sum(), 1))))
    return stats


def partial_covariate_residual(data: dict, var: str) -> dict:
    """Residualize `var` on pupil diameter and RMSSD (arousal covariates)
    via simple linear regression, then recompute the Metabolic State x
    Interoceptive Load cell means on the residuals. If the interaction
    survives after removing arousal-covariate variance, this supports
    allostatic triage (elevated theta_t) rather than generic LC-NE
    arousal/fatigue (Pred 4.C)."""
    y = data[var]
    X = np.column_stack([np.ones(len(y)), data["pupil_diameter"], data["rmssd"]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta + y.mean()  # re-center residual around original mean
    ms, il = data["metabolic_state"], data["interoceptive_load"]
    stats = {}
    for state in STATE_ORDER:
        for load in LOAD_ORDER:
            mask = (ms == state) & (il == load)
            v = residual[mask]
            stats[(state, load)] = (float(v.mean()), float(v.std() / np.sqrt(max(mask.sum(), 1))))
    return stats


def _plot_interaction(ax, stats: dict, ylabel: str, title: str) -> None:
    x = np.arange(len(LOAD_ORDER))
    width = 0.35
    for i, state in enumerate(STATE_ORDER):
        means = [stats[(state, load)][0] for load in LOAD_ORDER]
        sems = [stats[(state, load)][1] for load in LOAD_ORDER]
        color = PALETTE["S_t"] if state == "fed" else PALETTE["theta"]
        ax.bar(
            x + (i - 0.5) * width, means, width, yerr=sems,
            color=color, alpha=0.85, edgecolor="white", capsize=4,
            label=STATE_LABELS[state],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([LOAD_LABELS[l] for l in LOAD_ORDER], fontsize=9)
    ax.set_xlabel("Interoceptive load", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)


def plot(data: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=3, width=HALF_WIDTH * 3, height=PANEL_HEIGHT)

    # Panel A: d' interaction (Pred 4.A)
    dprime_stats = cell_stats(data, "d_prime")
    _plot_interaction(
        axes[0], dprime_stats, "Perceptual sensitivity d′",
        "Pred 4.A — Depletion selectively\nreduces d′ for high load",
    )

    # Panel B: P3b amplitude interaction (Pred 4.B)
    p3b_stats = cell_stats(data, "p3b_amplitude")
    _plot_interaction(
        axes[1], p3b_stats, "P3b amplitude (μV)",
        "Pred 4.B — Disproportionate P3b\nsuppression, high load",
    )

    # Panel C: interaction after covarying pupil diameter + RMSSD (Pred 4.C)
    p3b_resid_stats = partial_covariate_residual(data, "p3b_amplitude")
    _plot_interaction(
        axes[2], p3b_resid_stats, "P3b amplitude, arousal-residualized (μV)",
        "Pred 4.C — Interaction survives\npupil/RMSSD covariation",
    )

    label_axes(axes)
    fig.suptitle(
        "Figure 5 — Protocol 4 — Metabolic-State Crossover: Allostatic Threshold Modulation (Pred 4.A–Pred 4.C)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure5.pdf")

    if show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 5")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (CI mode)")
    args = parser.parse_args()

    data = load_data()
    dprime_stats = cell_stats(data, "d_prime")
    depleted_high = dprime_stats[("depleted", "high")][0]
    depleted_low = dprime_stats[("depleted", "low")][0]
    fed_high = dprime_stats[("fed", "high")][0]
    fed_low = dprime_stats[("fed", "low")][0]
    interaction = (fed_high - depleted_high) - (fed_low - depleted_low)
    print(
        f"  d' cells: fed/low={fed_low:.3f} fed/high={fed_high:.3f} "
        f"depleted/low={depleted_low:.3f} depleted/high={depleted_high:.3f}  "
        f"interaction contrast={interaction:.3f}"
    )
    plot(data, show=not args.no_show)


if __name__ == "__main__":
    main()
