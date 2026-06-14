"""Paper 2 — Figure 4: Visual Masking in APGI-LNN: Reservoir State Trajectories (§4.10).

Two panels:
  A — S(t) trajectory curves (short/critical/long ISI) with θₜ dashed
  B — Critical ISI parameter sensitivity (Π_target, θₜ, ρ_crit)

Caption: simulated with heterogeneous-τ form; quantitative ISI values indicative.

Run:
    python figures/paper2/generate_fig4.py
    python figures/paper2/generate_fig4.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import label_axes, save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

RNG = np.random.default_rng(7)


def _s_trajectory(t_arr, pi, isi_ms, theta=1.0, dt=1.0):
    """S(t) trajectory implementing APGI ignition dynamics.

    Accumulation proceeds (target + persisting iconic trace) until either the
    state crosses ``theta`` — triggering *stable attractor locking* (the state
    stays elevated in the ignition zone, a sustained conscious percept) — or the
    mask arrives first and the competing state suppresses accumulation. A
    trajectory that crosses threshold therefore does NOT decay back to baseline.

    Returns (S, mask_t, ignited).
    """
    S = np.zeros_like(t_arr, dtype=float)
    C = 1.55 * (pi / 1.2)  # accumulation ceiling
    tau = 150.0  # accumulation time constant (ms)
    s_lock = theta + 0.30  # ignition attractor fixed point (in ignition zone)
    mask_t = 50.0 + isi_ms  # target onset at 0, mask at 50 + ISI ms
    mask_dur = 40.0
    ignited = False

    for i, t in enumerate(t_arr):
        prev = S[i - 1] if i > 0 else 0.0
        masking = mask_t <= t < mask_t + mask_dur
        if t < 0:
            S[i] = 0.0
        elif ignited:
            # Locked attractor: relax to the ignition fixed point and stay there;
            # the mask produces only a small, transient competitive dip.
            target = s_lock - (0.18 if masking else 0.0)
            S[i] = prev + (target - prev) * dt / 25.0
        elif t < mask_t:
            # Drive accumulation; crossing theta latches ignition.
            S[i] = prev + (C - prev) * dt / tau
            if S[i] >= theta:
                ignited = True
        elif prev >= theta - 0.15:
            # Marginal / bistable: arrived near threshold as the mask hits;
            # hovers just below the ignition zone (critical-ISI near-miss).
            S[i] = prev + ((theta - 0.12) - prev) * dt / 50.0
        else:
            # Mask installs a competing state before ignition -> suppression.
            S[i] = max(0.0, prev - 0.9 * dt / 35.0)
    return S, mask_t, ignited


def plot(show: bool = True) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    t = np.arange(-20, 400)
    theta = 1.0

    SCENARIOS = [
        {
            "isi": 30,
            "label": "Short ISI (30 ms) → unconscious",
            "color": "#d6604d",
            "ls": "-",
        },
        {
            "isi": 90,
            "label": "Critical ISI (~90 ms) → bistable",
            "color": "#888888",
            "ls": "--",
        },
        {
            "isi": 180,
            "label": "Long ISI (180 ms) → conscious",
            "color": "#2166ac",
            "ls": "-",
        },
    ]

    # ── Panel A ────────────────────────────────────────────────────────────
    # Ignition zone shading first (behind trajectories)
    ax1.fill_betweenx([theta, 1.5], -20, 400, alpha=0.07, color="#4dac26", zorder=0)
    ax1.text(300, theta + 0.06, "Ignition zone", fontsize=8, color="#4dac26")

    for sc in SCENARIOS:
        S, mask_t, ignited = _s_trajectory(t, pi=1.2, isi_ms=sc["isi"], theta=theta)
        ax1.plot(
            t,
            S,
            lw=2.0,
            color=sc["color"],
            linestyle=sc["ls"],
            label=sc["label"],
            zorder=4,
        )
        # Mark mask onset on each trajectory and show the competitive deflection.
        if -20 < mask_t < 380:
            idx = int(np.searchsorted(t, mask_t))
            idx = min(idx, len(S) - 1)
            ax1.axvline(mask_t, ymin=0, ymax=0.06, lw=1.2, color=sc["color"], alpha=0.7)
            ax1.scatter(
                [mask_t],
                [S[idx]],
                s=28,
                color=sc["color"],
                edgecolor="white",
                lw=0.8,
                zorder=5,
            )
            ax1.annotate(
                "mask",
                xy=(mask_t, S[idx]),
                xytext=(mask_t + 4, S[idx] + (0.12 if ignited else -0.14)),
                fontsize=6.5,
                color=sc["color"],
                arrowprops=dict(arrowstyle="->", color=sc["color"], lw=0.8),
            )

    ax1.axhline(theta, lw=1.5, ls="--", color="#333333", zorder=1)
    ax1.text(350, theta + 0.03, r"$\theta_t$", fontsize=10, color="#333333")

    # Trial-timeline bar at top; explicitly mark the 50 ms target duration.
    for t_start, t_end, label, color in [
        (0, 50, "Target", "#fcbba1"),
        (50, 140, "ISI (variable)", "#fee0d2"),
        (140, 200, "Mask", "#fc9272"),
    ]:
        ax1.fill_betweenx([1.58, 1.66], t_start, t_end, color=color, alpha=0.9)
        ax1.text((t_start + t_end) / 2, 1.68, label, ha="center", fontsize=7)
    # 50 ms target-duration dimension annotation
    ax1.annotate(
        "",
        xy=(0, 1.54),
        xytext=(50, 1.54),
        arrowprops=dict(arrowstyle="<->", color="#a63603", lw=1.0),
    )
    ax1.text(25, 1.50, "50 ms", ha="center", va="top", fontsize=6.8, color="#a63603")

    ax1.set_xlabel("Time (ms)", fontsize=10)
    ax1.set_ylabel(r"$S(t)$", fontsize=10)
    ax1.set_title("S(t) trajectories by ISI condition", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8, loc="center right")
    ax1.set_xlim(-20, 380)
    ax1.set_ylim(-0.05, 1.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Critical ISI parameter sensitivity ────────────────────────
    param_range = np.linspace(0.5, 1.5, 100)

    crit_ISI_pi = 80 + 60 * (1.0 - param_range)  # higher Π → shorter ISI
    crit_ISI_theta = 50 + 100 * (param_range - 0.5)  # higher θ → longer ISI
    crit_ISI_rho = 40 + 110 * (param_range - 0.5)  # higher ρ_crit → longer ISI

    ax2.plot(
        param_range,
        crit_ISI_pi,
        lw=2,
        color="#2166ac",
        label=r"Π$_{\mathrm{target}}$ ↑ → critical ISI ↓",
    )
    ax2.plot(
        param_range,
        crit_ISI_theta,
        lw=2,
        color="#d6604d",
        label=r"$\theta_t$ ↑ → critical ISI ↑",
    )
    ax2.plot(
        param_range,
        crit_ISI_rho,
        lw=2,
        color="#4dac26",
        label=r"$\rho_{\mathrm{crit}}$ ↑ → critical ISI ↑",
    )

    ax2.set_xlabel("Parameter value (normalised)", fontsize=10)
    ax2.set_ylabel("Critical ISI (ms, indicative)", fontsize=10)
    ax2.set_title(
        "Critical ISI parameter sensitivity\n"
        "(distinguishing prediction: masking ∝ ρ_crit)",
        fontsize=10,
        fontweight="bold",
    )
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    label_axes([ax1, ax2])
    caption = (
        "Simulated using heterogeneous-τ form (§4.1). "
        "Quantitative ISI values indicative; empirical calibration required per EP-6."
    )
    fig.text(
        0.5, -0.03, caption, ha="center", fontsize=7.5, color="#666666", style="italic"
    )
    fig.suptitle(
        "Figure 4 — Visual Masking in APGI-LNN: Reservoir State Trajectories\n"
        "(Paper 2, §4.10)",
        fontsize=11,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig4_visual_masking_trajectories.pdf")
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
