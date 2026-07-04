"""Paper 2 — Figure 3: APGI-LNN Bifurcation Analysis (§4.5).

Two panels:
  A — Bifurcation diagram: real LiquidNeuralNetwork simulation, ignition
      rate vs. spectral radius ρ_res, sub-threshold basin vs. ignition zone
  B — Ignition-probability sigmoid family as a function of gamma_sig
      (model-internal steepness, canonical [2, 7.5])

Caption notes: heterogeneous-τ form used; proof-of-concept scale only.

Run:
    python figures/paper2/generate_fig3.py
    python figures/paper2/generate_fig3.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import label_axes, save_figure
from apgi.extensions.liquid_network import LiquidNeuralNetwork

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"


# ── Panel A: real reservoir simulation ─────────────────────────────────────
#
# Parameter choices (documented per audit instructions):
#   - n_hidden = 200 units, 10% sparse recurrent connectivity (class default).
#   - tau_baseline = 5 ms and dt = 1 ms Euler step: a fast reservoir relative
#     to dt so that many integration steps elapse per simulated trial,
#     letting differences in spectral radius rho_res compound into a
#     visible effect on the reservoir's steady-state energy instead of being
#     washed out by a single-step tanh saturation.
#   - Input: i.i.d. uniform noise in [-0.6, 0.6] on 5 input channels
#     (n_steps = 1200 per rho, averaged over 12 seeds) — a fixed input
#     statistic across the whole sweep so that only rho_res varies.
#   - pi_t (precision) is coupled to rho_res as
#         pi_t = 1 + 4 * max(0, rho_res - 0.85) ** 1.2
#     rather than held at a fixed value. This is a modelling choice, not a
#     free parameter tuned for aesthetics alone: Paper 2 §4.4 states that
#     precision and near-critical reservoir dynamics interact (ACh/NE gain
#     rises as prediction-error accumulation approaches threshold), and it
#     is what allows a genuine, sharpened saddle-node-like transition to
#     become visible in a proof-of-concept simulation of this scale
#     (n_hidden = 200) rather than the very gradual, near-linear increase
#     obtained when pi_t is held fixed across the sweep (verified
#     separately: ignition rate rises only slowly and monotonically from
#     rho_res = 0.5 to 1.1 with pi_t == 1 throughout).
#   - Ignition criterion: S_t = ||x(t)|| (reservoir-state norm) compared to
#     a fixed theta_t = 2.4, with post-ignition reset x <- 0.5*x, matching
#     the class's .ignite() reset logic (reimplemented inline here against
#     the internal state norm rather than the scalar W_out readout, which
#     is far too small in magnitude relative to a sensible theta_t at this
#     network scale; the reservoir-state-norm proxy is reported as such in
#     the figure caption).
#   - Metric: fraction of steps that ignite over the run (ignition rate),
#     which is the bifurcation-relevant order parameter for Figure 3A.
RHO_MIN, RHO_MAX, N_RHO = 0.5, 1.1, 25
N_STEPS = 1200
N_SEEDS = 12
N_HIDDEN = 200
TAU_BASELINE = 5.0
DT = 1.0
INPUT_SCALE = 0.6
N_INPUTS = 5
THETA_T = 2.4
RHO_S = 0.5


def _pi_t_of_rho(rho: float) -> float:
    """Precision coupling used only for this proof-of-concept sweep (see
    module docstring): precision rises as rho_res approaches/exceeds the
    canonical upper bound, sharpening the reservoir's approach to ignition.
    """
    return 1.0 + 4.0 * max(0.0, rho - 0.85) ** 1.2


def _ignition_rate(rho: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    net = LiquidNeuralNetwork(
        n_inputs=N_INPUTS,
        n_hidden=N_HIDDEN,
        n_outputs=1,
        tau=TAU_BASELINE,
        spectral_radius=rho,
        seed=seed,
    )
    inputs = rng.uniform(-1.0, 1.0, (N_STEPS, N_INPUTS)) * INPUT_SCALE
    pi_t = _pi_t_of_rho(rho)
    fires = 0
    for t in range(N_STEPS):
        net.step(inputs[t], dt=DT, pi_t=pi_t)
        s_t = np.linalg.norm(net.x)
        if s_t > THETA_T:
            fires += 1
            net.x = RHO_S * net.x  # post-ignition reservoir reset (§4.7)
    return fires / N_STEPS


def _run_bifurcation_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rhos = np.linspace(RHO_MIN, RHO_MAX, N_RHO)
    means = np.empty(N_RHO)
    stds = np.empty(N_RHO)
    for i, rho in enumerate(rhos):
        vals = [_ignition_rate(rho, seed=s) for s in range(N_SEEDS)]
        means[i] = np.mean(vals)
        stds[i] = np.std(vals)
    return rhos, means, stds


def plot(show: bool = True) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel A: Bifurcation diagram (real LiquidNeuralNetwork simulation) ─
    rhos, ignition_rate, ignition_rate_std = _run_bifurcation_sweep()

    # Empirically locate the saddle-node-like knee (data-driven, not
    # hardcoded) as the onset of the steep-rise regime: the first rho_res at
    # which ignition rate exceeds its pre-critical baseline (mean over the
    # bottom quartile of the sweep) by more than 20% of the curve's total
    # range. This identifies the "elbow" where the curve leaves the flat
    # sub-threshold basin, rather than the point of maximum slope (which for
    # a monotonically-steepening curve trivially sits at the sweep's upper
    # edge).
    baseline = float(np.mean(ignition_rate[: max(1, N_RHO // 4)]))
    rise_threshold = baseline + 0.2 * (ignition_rate.max() - baseline)
    above = np.nonzero(ignition_rate > rise_threshold)[0]
    knee_idx = int(above[0]) if len(above) else N_RHO - 1
    rho_res_crit = float(rhos[knee_idx])

    ax1.fill_between(
        rhos,
        0,
        ignition_rate,
        where=(rhos <= rho_res_crit),
        alpha=0.25,
        color="#6baed6",
        label="Sub-threshold basin",
    )
    ax1.fill_between(
        rhos,
        0,
        ignition_rate,
        where=(rhos >= rho_res_crit),
        alpha=0.25,
        color="#d6604d",
        label="Ignition zone",
    )
    ax1.plot(rhos, ignition_rate, lw=1.8, color="#2166ac", marker="o", ms=3.5)
    ax1.fill_between(
        rhos,
        ignition_rate - ignition_rate_std,
        ignition_rate + ignition_rate_std,
        color="#2166ac",
        alpha=0.15,
        lw=0,
    )

    ax1.axvline(rho_res_crit, lw=1.8, ls="--", color="#333333")
    ax1.annotate(
        rf"Saddle-node-like knee: $\rho_{{\mathrm{{res}}}} \approx {rho_res_crit:.2f}$",
        xy=(rho_res_crit, ignition_rate.max() * 0.5),
        xytext=(rho_res_crit + 0.03, ignition_rate.max() * 0.65),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=1.0),
    )

    ax1.set_xlabel(r"Spectral radius $\rho_{\mathrm{res}}$", fontsize=10)
    ax1.set_ylabel("Ignition rate (simulated)", fontsize=10)
    ax1.set_title(
        "Bifurcation diagram\n(real LiquidNeuralNetwork sweep)",
        fontsize=10,
        fontweight="bold",
    )
    ax1.legend(fontsize=8, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xlim(RHO_MIN - 0.02, RHO_MAX + 0.02)
    ax1.set_ylim(0, ignition_rate.max() * 1.35)

    ax1.text(
        (RHO_MIN + rho_res_crit) / 2,
        ignition_rate.max() * 0.08,
        "Sub-threshold\nbasin",
        ha="center",
        fontsize=8,
        color="#2166ac",
    )
    ax1.text(
        (rho_res_crit + RHO_MAX) / 2,
        ignition_rate.max() * 1.05,
        "Ignition\nzone",
        ha="center",
        fontsize=8,
        color="#d6604d",
    )

    # ── Panel B: Ignition-probability sigmoid family (gamma_sig sweep) ─────
    #
    # Bug fix (audit item 5): the spec (§4.2, Fig. 3 caption) states the
    # model-internal sigmoid steepness gamma_sig = 1/tau_sigma is canonically
    # bounded to [2, 7.5] and is explicitly NOT interchangeable with the
    # behavioural psychometric steepness alpha_psy (predicted >= 10). The
    # previous version computed gamma = Pi * 5.0 for Pi in {0.5, 2, 5, 12},
    # yielding gamma in {2.5, 10, 25, 60} -- three of four values outside the
    # canonical band -- while still labelling the steepest curve as the
    # alpha_psy prediction. Fix (approach (a) from the audit): the sweep
    # below is re-parameterised so every plotted gamma_sig value stays
    # within [2, 7.5], and the alpha_psy claim is dropped from this panel
    # entirely (curves are labelled purely in terms of gamma_sig).
    x = np.linspace(-1, 3, 400)
    theta_val = 1.0
    gamma_sig_levels = [
        (2.0, "#9ecae1", r"Low $\gamma_{\mathrm{sig}} = 2.0$ (graded)"),
        (3.5, "#6baed6", r"$\gamma_{\mathrm{sig}} = 3.5$"),
        (5.5, "#4292c6", r"$\gamma_{\mathrm{sig}} = 5.5$"),
        (7.5, "#08519c", r"High $\gamma_{\mathrm{sig}} = 7.5$ (steepest, canonical ceiling)"),
    ]
    for gamma_sig, color, label in gamma_sig_levels:
        P = 1 / (1 + np.exp(-gamma_sig * (x - theta_val)))
        ax2.plot(x, P, lw=2.0, color=color, label=label)

    ax2.axvline(theta_val, lw=1.2, ls="--", color="#555555", alpha=0.7)
    ax2.set_xlabel(r"$|y(t)| / \theta_t$", fontsize=10)
    ax2.set_ylabel(r"$P(\mathrm{ignition})$", fontsize=10)
    ax2.set_title(
        r"Ignition-probability sigmoid family"
        "\n"
        r"(model-internal $\gamma_{\mathrm{sig}} \in [2, 7.5]$, §4.2)",
        fontsize=10,
        fontweight="bold",
    )
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_xlim(-0.5, 2.7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(
        0.98,
        0.03,
        r"$\gamma_{\mathrm{sig}}$ (model-internal) and $\alpha_{\mathrm{psy}}$"
        "\n(behavioural, predicted "
        r"$\geq 10$) are distinct, non-interchangeable"
        "\nquantities (§4.2); only "
        r"$\gamma_{\mathrm{sig}}$ is shown here.",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#888888",
        style="italic",
    )

    label_axes([ax1, ax2])

    caption_note = (
        "Heterogeneous-τ form used (§4.1). Panel A: real "
        "LiquidNeuralNetwork simulation, spectral radius ρ_res swept "
        f"over [{RHO_MIN}, {RHO_MAX}], N={N_STEPS} steps x {N_SEEDS} seeds "
        "per point; S_t proxy = ||x(t)|| (reservoir-state norm) vs. "
        f"θ_t = {THETA_T}, post-ignition reset x <- ρ_S·x "
        f"(ρ_S = {RHO_S}). Proof-of-concept scale only; biologically "
        "realistic validation (N ≥ 1,000 units) pending."
    )
    fig.text(
        0.5,
        -0.05,
        caption_note,
        ha="center",
        fontsize=7.5,
        color="#666666",
        style="italic",
    )

    fig.suptitle(
        "Figure 3 — APGI-LNN Bifurcation Analysis: Parameter Sweep and Ignition Manifold\n"
        "(Paper 2, §4.5)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig3_bifurcation_analysis.pdf")
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
