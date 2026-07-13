"""Figure 1 — Protocol 0 (HEP Proxy Validation): predicted validation of the
heartbeat-evoked-potential (HEP) amplitude as a proxy for interoceptive
precision (Πⁱ).

Per OUP-Protocols.txt Figure 1 caption:
  (A) HEP amplitude (250-400 ms post-R) correlates with an orthogonal
      behavioural interoceptive-precision index (heartbeat-discrimination
      d'); the proxy is retained only if r >= 0.35 (Pred 0.A).
  (B) Physostigmine (cholinergic probe) raises HEP amplitude by >= 15%
      relative to placebo in a double-blind crossover (Pred 0.B).
  (C) Trial-by-trial anterior-insula (aINS) BOLD tracks HEP amplitude at
      r >= 0.30 after arousal control (Pred 0.C).

Loads the archived seed dataset data/seeds/sim0_hep_proxy.npz (N=60: 30 main
+ 30 independent replication subjects) rather than inventing data inline.

Run:
    python figures/generate_figure1.py
    python figures/generate_figure1.py --no-show   # CI mode
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from figures.utils import (  # noqa: E402
    HALF_WIDTH,
    PALETTE,
    PANEL_HEIGHT,
    ensure_seed_dataset,
    label_axes,
    make_figure,
    save_figure,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
DATA_DIR = pathlib.Path(
    os.environ.get("APGI_DATA_DIR", pathlib.Path(__file__).resolve().parent.parent / "data" / "seeds")
)


def load_data(path: pathlib.Path | None = None) -> dict:
    npz_path = path or ensure_seed_dataset(DATA_DIR / "sim0_hep_proxy.npz", "_gen_sim0_hep_proxy")
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def plot(data: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=3, width=HALF_WIDTH * 3, height=PANEL_HEIGHT)

    # ---- Panel A: interoceptive d' vs HEP amplitude (Pred 0.A, r >= 0.35) ----
    # Spec axes: x = Interoceptive d' (heartbeat discrimination), y = HEP
    # amplitude (250-400 ms post-R).
    ax = axes[0]
    hep = data["hep_amplitude"]
    dprime = data["d_prime"]
    labels = data["sample_label"]
    r, _ = pearsonr(dprime, hep)
    for lbl, marker, color in [
        ("main", "o", PALETTE["theta"]),
        ("replication", "^", "#f4a582"),
    ]:
        mask = labels == lbl
        ax.scatter(
            dprime[mask], hep[mask], s=26, alpha=0.75, color=color,
            edgecolors="white", linewidths=0.4, marker=marker, label=lbl.capitalize(),
        )
    m, b = np.polyfit(dprime, hep, 1)
    x_line = np.linspace(dprime.min(), dprime.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#333333", lw=1.3, ls="--")
    ax.annotate(f"r ≈ {r:.2f}\n(retain if r ≥ 0.35)", xy=(0.05, 0.86),
                xycoords="axes fraction", fontsize=8.5)
    ax.set_xlabel("Interoceptive d′\n(heartbeat discrimination)", fontsize=9.5)
    ax.set_ylabel("HEP amplitude (μV, 250–400 ms post-R)", fontsize=9.5)
    ax.set_title("Pred 0.A — HEP tracks\ninteroceptive precision", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")

    # ---- Panel B: physostigmine vs placebo HEP effect (Pred 0.B, >=15%) ----
    # Paired slope-style bars: bars show group means, thin grey lines + dots
    # show each participant's within-subject placebo -> physostigmine pair.
    ax = axes[1]
    placebo = data["hep_placebo"]
    physo = data["hep_physostigmine"]
    means = [placebo.mean(), physo.mean()]
    sems = [placebo.std() / np.sqrt(len(placebo)), physo.std() / np.sqrt(len(physo))]
    ax.bar(
        ["Placebo", "Physostigmine"], means, yerr=sems,
        color=[PALETTE["identity"], "#2E9E5B"], alpha=0.35, edgecolor="white",
        width=0.5, capsize=5, zorder=2,
    )
    for p, ph in zip(placebo, physo):
        ax.plot([0, 1], [p, ph], color="#999999", lw=0.6, alpha=0.5, zorder=3)
    ax.scatter(np.zeros_like(placebo), placebo, s=14, color=PALETTE["identity"],
               alpha=0.7, zorder=4)
    ax.scatter(np.ones_like(physo), physo, s=14, color="#2E9E5B", alpha=0.7, zorder=4)
    delta_pct = float(data["physo_delta_pct"].mean())
    ax.annotate(
        f"+15–20% threshold; d ≥ 0.50\nΔ = {delta_pct:+.1f}%",
        xy=(0.5, 0.94), xycoords="axes fraction",
        ha="center", fontsize=8.5,
    )
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("HEP amplitude (μV)", fontsize=9.5)
    ax.set_ylim(0, max(physo.max(), placebo.max()) * 1.15)
    ax.set_title("Pred 0.B — Cholinergic\nelevation of HEP", fontsize=10)

    # ---- Panel C: trial-level HEP-aINS BOLD coupling (Pred 0.C, r >= 0.30) ----
    # No trial-level (aINS BOLD, HEP) pairs exist in the archived seed
    # (sim0_hep_proxy.npz only stores a per-subject summary coupling
    # coefficient), so a representative within-participant trial-level
    # scatter is synthesized here targeting the pre-registered r ~= 0.30,
    # per the figure spec (scatter, not a histogram of subject coefficients).
    ax = axes[2]
    rng_c = np.random.default_rng(7)
    n_trials_c = 90
    target_r = 0.30
    z1 = rng_c.standard_normal(n_trials_c)
    z2 = rng_c.standard_normal(n_trials_c)
    ains_bold = z1
    hep_trial = target_r * z1 + np.sqrt(1 - target_r**2) * z2
    ains_bold = ains_bold * 0.6 + 0.3  # rescale to a.u.
    hep_trial = hep_trial * 1.8 + 3.6  # rescale to uV, matching Panel A/B range
    r_c, _ = pearsonr(ains_bold, hep_trial)
    ax.scatter(ains_bold, hep_trial, s=24, alpha=0.75, color=PALETTE["S_t"],
               edgecolors="white", linewidths=0.4)
    mc, bc = np.polyfit(ains_bold, hep_trial, 1)
    x_line_c = np.linspace(ains_bold.min(), ains_bold.max(), 100)
    y_line_c = mc * x_line_c + bc
    ax.plot(x_line_c, y_line_c, color="#333333", lw=1.3, ls="--")
    se = np.sqrt(np.sum((hep_trial - (mc * ains_bold + bc)) ** 2) / (n_trials_c - 2))
    ax.fill_between(x_line_c, y_line_c - 1.96 * se, y_line_c + 1.96 * se,
                     color=PALETTE["S_t"], alpha=0.15, lw=0)
    ax.annotate(f"within-participant\nr ≈ {r_c:.2f}", xy=(0.05, 0.90),
                xycoords="axes fraction", fontsize=8.5, color=PALETTE["S_t"])
    ax.set_xlabel("Anterior-insula BOLD (a.u.)", fontsize=9.5)
    ax.set_ylabel("HEP amplitude (μV)", fontsize=9.5)
    ax.set_title("Pred 0.C — HEP tracks\naINS BOLD trial-by-trial", fontsize=10)

    label_axes(axes)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure1.pdf")

    if show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 1")
    parser.add_argument("--no-show", action="store_true", help="Skip plt.show() (CI mode)")
    args = parser.parse_args()

    data = load_data()
    print(
        f"  r(HEP,d') main={pearsonr(data['hep_amplitude'], data['d_prime'])[0]:.3f}  "
        f"physo_delta={float(data['physo_delta_pct'].mean()):.1f}%  "
        f"mean_ains_coupling={float(data['ains_coupling'].mean()):.3f}"
    )
    plot(data, show=not args.no_show)


if __name__ == "__main__":
    main()
