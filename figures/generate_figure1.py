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

    # ---- Panel A: HEP amplitude vs interoceptive d' (Pred 0.A, r >= 0.35) ----
    ax = axes[0]
    hep = data["hep_amplitude"]
    dprime = data["d_prime"]
    labels = data["sample_label"]
    r, _ = pearsonr(hep, dprime)
    for lbl, marker, color in [
        ("main", "o", PALETTE["S_t"]),
        ("replication", "^", PALETTE["theta"]),
    ]:
        mask = labels == lbl
        ax.scatter(
            hep[mask], dprime[mask], s=26, alpha=0.75, color=color,
            edgecolors="white", linewidths=0.4, marker=marker, label=lbl.capitalize(),
        )
    m, b = np.polyfit(hep, dprime, 1)
    x_line = np.linspace(hep.min(), hep.max(), 100)
    ax.plot(x_line, m * x_line + b, color="#333333", lw=1.3, ls="--")
    ax.annotate(f"r = {r:.3f}\n(threshold r ≥ 0.35)", xy=(0.05, 0.86),
                xycoords="axes fraction", fontsize=8.5)
    ax.set_xlabel("HEP amplitude, 250–400 ms (μV)", fontsize=9.5)
    ax.set_ylabel("Heartbeat-discrimination d′", fontsize=9.5)
    ax.set_title("Pred 0.A — HEP tracks\ninteroceptive precision", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")

    # ---- Panel B: physostigmine vs placebo HEP effect (Pred 0.B, >=15%) ----
    ax = axes[1]
    placebo = data["hep_placebo"]
    physo = data["hep_physostigmine"]
    means = [placebo.mean(), physo.mean()]
    sems = [placebo.std() / np.sqrt(len(placebo)), physo.std() / np.sqrt(len(physo))]
    ax.bar(
        ["Placebo", "Physostigmine"], means, yerr=sems,
        color=[PALETTE["theta"], PALETTE["S_t"]], alpha=0.85, edgecolor="white",
        width=0.5, capsize=5,
    )
    delta_pct = float(data["physo_delta_pct"].mean())
    ax.annotate(
        f"Δ = {delta_pct:+.1f}% (threshold ≥ 15%)",
        xy=(0.5, 0.94), xycoords="axes fraction",
        ha="center", fontsize=8.5,
    )
    ax.set_ylabel("HEP amplitude (μV)", fontsize=9.5)
    ax.set_ylim(0, max(means) * 1.35)
    ax.set_title("Pred 0.B — Cholinergic\nelevation of HEP", fontsize=10)

    # ---- Panel C: HEP-aINS BOLD coupling (Pred 0.C, r >= 0.30) ----
    ax = axes[2]
    coupling = data["ains_coupling"]
    ax.hist(coupling, bins=16, color=PALETTE["S_t"], alpha=0.8, edgecolor="white")
    mean_coupling = float(coupling.mean())
    ax.axvline(mean_coupling, color="#333333", lw=1.5, ls="--",
               label=f"mean r = {mean_coupling:.3f}")
    ax.axvline(0.30, color=PALETTE["theta"], lw=1.3, ls=":",
               label="threshold r ≥ 0.30")
    ax.set_xlabel("Trial-level HEP–aINS BOLD coupling (r)", fontsize=9.5)
    ax.set_ylabel("Subjects", fontsize=9.5)
    ax.set_title("Pred 0.C — HEP tracks\naINS BOLD trial-by-trial", fontsize=10)
    ax.legend(fontsize=7)

    label_axes(axes)
    fig.suptitle(
        "Figure 1 — Protocol 0 — HEP Proxy Validation (Pred 0.A–Pred 0.C)",
        fontsize=11, y=1.02,
    )
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
