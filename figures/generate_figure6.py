"""Figure 6 — Protocol 4: DoC joint biomarker model (Pred 4.A–Pred 4.B).

Simulates HEP amplitude and PCI scores for four DoC groups from
protocol_4_disorders_of_consciousness.json and shows:
  A — HEP amplitude by DoC group (Pred 4.B)
  B — PCI by DoC group
  C — Joint model R² vs univariate R² for 3-month GCS-S recovery (Pred 4.A)

Run:
    python figures/generate_figure6.py
    python figures/generate_figure6.py --no-show   # CI mode
"""

import sys as _sys
import pathlib as _pathlib

_sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))

import argparse
import pathlib

import numpy as np
from scipy.stats import pearsonr

from apgi.core import compute_pi_i_eff
from figures.utils import (
    PALETTE,
    HALF_WIDTH,
    PANEL_HEIGHT,
    label_axes,
    make_figure,
    save_figure,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# Protocol 4 APGI parameters
PI_I_BY_GROUP = {
    "VS/UWS": 0.3,
    "MCS": 0.8,
    "Controls": 1.2,
}
N_PER_GROUP = {"VS/UWS": 15, "MCS": 20, "Controls": 15}
GROUP_COLORS = {
    "VS/UWS": PALETTE["theta"],
    "MCS": "#FFCC00",
    "Controls": PALETTE["S_t"],
}


def simulate(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    hep_by_group, pci_by_group = {}, {}

    for group, pi_i in PI_I_BY_GROUP.items():
        n = N_PER_GROUP[group]
        C = rng.uniform(0.5, 2.0, n)
        pi_i_eff = compute_pi_i_eff(pi_i, C, kappa=100.0)
        # HEP amplitude ~ Πⁱ_eff + noise
        hep_by_group[group] = pi_i_eff + rng.normal(0, 0.07, n)
        # PCI ~ ignition capacity (higher in MCS/controls)
        pci_base = {"VS/UWS": 0.18, "MCS": 0.35, "Controls": 0.52}[group]
        pci_by_group[group] = rng.normal(pci_base, 0.06, n)

    # Simulate 3-month GCS-S outcomes for regression (Pred 4.A)
    all_hep_list: list[float] = []
    all_pci_list: list[float] = []
    all_gcs_list: list[float] = []
    for group in PI_I_BY_GROUP:
        n = N_PER_GROUP[group]
        base_gcs = {"VS/UWS": 4.0, "MCS": 9.0, "Controls": 14.5}[group]
        gcs = base_gcs + 2.0 * hep_by_group[group] + 3.0 * pci_by_group[group]
        gcs += rng.normal(0, 1.2, n)
        all_hep_list.extend(hep_by_group[group])
        all_pci_list.extend(pci_by_group[group])
        all_gcs_list.extend(gcs)

    all_hep = np.array(all_hep_list)
    all_pci = np.array(all_pci_list)
    all_gcs = np.array(all_gcs_list)

    return {
        "hep_by_group": hep_by_group,
        "pci_by_group": pci_by_group,
        "all_hep": all_hep,
        "all_pci": all_pci,
        "all_gcs": all_gcs,
    }


def r_squared(y_true: np.ndarray, predictors: list[np.ndarray]) -> float:
    X = np.column_stack([np.ones(len(y_true))] + predictors)
    beta, *_ = np.linalg.lstsq(X, y_true, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def plot(data: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=3, width=HALF_WIDTH * 3, height=PANEL_HEIGHT)

    groups = list(PI_I_BY_GROUP.keys())
    x = np.arange(len(groups))

    # Panel A: HEP amplitude by DoC group (Pred 4.B)
    ax = axes[0]
    means = [data["hep_by_group"][g].mean() for g in groups]
    sems = [data["hep_by_group"][g].std() / np.sqrt(N_PER_GROUP[g]) for g in groups]
    ax.bar(
        x,
        means,
        yerr=sems,
        color=[GROUP_COLORS[g] for g in groups],
        alpha=0.85,
        edgecolor="white",
        width=0.5,
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("HEP amplitude (a.u.)", fontsize=10)
    ax.set_title("Pred 4.B — HEP discriminates\nMCS from VS/UWS", fontsize=10)

    # Panel B: PCI by DoC group
    ax = axes[1]
    means_pci = [data["pci_by_group"][g].mean() for g in groups]
    sems_pci = [data["pci_by_group"][g].std() / np.sqrt(N_PER_GROUP[g]) for g in groups]
    ax.bar(
        x,
        means_pci,
        yerr=sems_pci,
        color=[GROUP_COLORS[g] for g in groups],
        alpha=0.85,
        edgecolor="white",
        width=0.5,
        capsize=4,
    )
    ax.axhline(
        0.31, ls="--", lw=1, color="black", alpha=0.5, label="PCI threshold = 0.31"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("PCI", fontsize=10)
    ax.set_title("PCI by DoC group", fontsize=10)
    ax.legend(fontsize=7)

    # Panel C: Joint vs univariate R² (Pred 4.A)
    ax = axes[2]
    r2_hep = r_squared(data["all_gcs"], [data["all_hep"]])
    r2_pci = r_squared(data["all_gcs"], [data["all_pci"]])
    r2_joint = r_squared(data["all_gcs"], [data["all_hep"], data["all_pci"]])
    model_names = ["HEP only", "PCI only", "HEP + PCI\n(joint)"]
    r2_vals = [r2_hep, r2_pci, r2_joint]
    bar_colors = [PALETTE["S_t"], "#9966FF", "#00CC99"]
    bars = ax.bar(
        model_names, r2_vals, color=bar_colors, alpha=0.85, edgecolor="white", width=0.5
    )
    for bar, val in zip(bars, r2_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel(r"$R^2$ for GCS-S recovery", fontsize=10)
    ax.set_title("Pred 4.A — Joint model\noutperforms univariate", fontsize=10)
    ax.set_ylim(0, 1)

    label_axes(axes)
    fig.suptitle(
        "Figure 6 — Protocol 4: Disorders of Consciousness Biomarker Model",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure6.pdf")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 6")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    plot(simulate(), show=not args.no_show)


if __name__ == "__main__":
    main()
