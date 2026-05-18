"""Figure 4 — Protocol 2: TMS-induced disruption of Πⁱ_eff and PCI (P2a–P2c).

Simulates three TMS conditions from protocol_2_tms_insular_gating.json:
insula_active, thalamus_active, vertex_sham. Shows predicted PCI reduction
and HEP–P3b coupling abolition under insula TMS.

Run:
    python figures/generate_figure4.py
    python figures/generate_figure4.py --no-show   # CI mode
"""

import sys as _sys
import pathlib as _pathlib

_sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))

import argparse
import pathlib

import numpy as np
from scipy.stats import pearsonr

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

# Protocol 2 APGI parameters (from protocol_2_tms_insular_gating.json)
KAPPA = 100.0
ALPHA = 0.3
BETA = 0.7
DELTA_PI_INSULA = -0.4  # insula TMS reduces Πⁱ
DELTA_PI_THALAMUS = -0.3  # thalamic TMS reduces globally
DELTA_PI_SHAM = 0.0
N_TRIALS = 480


def simulate_condition(
    pi_i_base: float, delta_pi: float, seed: int, n: int = N_TRIALS
) -> dict:
    rng = np.random.default_rng(seed)
    pi_i = max(0.01, pi_i_base + delta_pi)
    C_metabolic = rng.uniform(0.5, 2.0, n)
    pi_i_eff = compute_pi_i_eff(pi_i, C_metabolic, kappa=KAPPA)
    pi_e = rng.uniform(0.8, 1.5, n)
    z_e = rng.uniform(0.2, 1.0, n)
    z_i = rng.uniform(0.1, 0.8, n)
    V_info = rng.uniform(0.1, 1.0, n)
    S_t = np.array(
        [
            compute_S_t(
                float(pi_e[i]), float(z_e[i]), float(pi_i_eff[i]), float(z_i[i])
            )
            for i in range(n)
        ]
    )
    theta_t = np.array(
        [
            compute_theta_t(float(C_metabolic[i]), float(V_info[i]), ALPHA, BETA)
            for i in range(n)
        ]
    )
    detected = np.array([ignition_criterion(S_t[i], theta_t[i]) for i in range(n)])
    hep = pi_i_eff + rng.normal(0, 0.08, n)
    p3b = S_t * detected + rng.normal(0, 0.05, n)
    # PCI ~ ignition rate scaled to 0–1 (simplified proxy)
    pci = detected.mean() * 0.8 + rng.uniform(0, 0.05)
    return {"S_t": S_t, "detected": detected, "hep": hep, "p3b": p3b, "pci": float(pci)}


def plot(show: bool = True) -> None:
    pi_i_base = 1.0
    conditions = {
        "Vertex\n(sham)": simulate_condition(pi_i_base, DELTA_PI_SHAM, seed=1),
        "Insula\n(active)": simulate_condition(pi_i_base, DELTA_PI_INSULA, seed=2),
        "Thalamus\n(active)": simulate_condition(pi_i_base, DELTA_PI_THALAMUS, seed=3),
    }

    fig, axes = make_figure(ncols=3, width=HALF_WIDTH * 3, height=PANEL_HEIGHT)

    labels = list(conditions.keys())
    colors = [PALETTE["S_t"], PALETTE["theta"], "#9966FF"]

    # Panel A: PCI by TMS condition (P2a)
    ax = axes[0]
    pcis = [conditions[k]["pci"] for k in labels]
    bars = ax.bar(labels, pcis, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    ax.axhline(
        0.31,
        ls="--",
        lw=1,
        color="black",
        alpha=0.6,
        label="PCI consciousness threshold",
    )
    ax.set_ylabel("PCI (ignition proxy)", fontsize=10)
    ax.set_title("P2a — PCI reduction\nby TMS site", fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=7)

    # Panel B: Ignition rate by TMS condition
    ax = axes[1]
    rates = [conditions[k]["detected"].mean() for k in labels]
    ax.bar(labels, rates, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    ax.set_ylabel("Ignition rate", fontsize=10)
    ax.set_title("P2b — Threshold elevation\nunder insula TMS", fontsize=10)
    ax.set_ylim(0, 1)

    # Panel C: HEP–P3b coupling by condition (P2b: abolished under insula TMS)
    ax = axes[2]
    r_vals, r_labels = [], []
    for (lbl, cond), col in zip(conditions.items(), colors):
        r, _ = pearsonr(cond["hep"], cond["p3b"])
        r_vals.append(r)
        r_labels.append(lbl.replace("\n", " "))
    ax.bar(r_labels, r_vals, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    ax.axhline(0, ls="--", lw=0.8, color="black", alpha=0.4)
    ax.set_ylabel(r"HEP–P3b coupling (r)", fontsize=10)
    ax.set_title("P2b — HEP–P3b coupling\nabolished by insula TMS", fontsize=10)

    label_axes(axes)
    fig.suptitle(
        "Figure 4 — Protocol 2: TMS Insular Gating of Πⁱ_eff", fontsize=11, y=1.02
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure4.pdf")
    if show:
        import matplotlib.pyplot as plt

        plt.show()
    import matplotlib.pyplot as plt

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 4")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    plot(show=not args.no_show)


if __name__ == "__main__":
    main()
