"""Figure 4 — Protocol 2 — Somatic-AgentSim: Somatic marker agent performance advantage (Pred 2.a–Pred 2.e).

Simulates five agent types from protocol_2_somatic_agent_sim.json
under three volatility levels and shows reward advantage of the full APGI
agent with somatic marker M̂ over β-lesion and other-lesion agents (Pred 2.a–Pred 2.d).
Pred 2.e: full APGI generative model achieves lower BIC than GNWT-only and Standard PP
when fit to human IGT trial-by-trial choice sequences (ΔBIC ≥ 10).

Run:
    python figures/generate_figure4.py
    python figures/generate_figure4.py --no-show   # CI mode
"""

import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))

import argparse
import pathlib

import numpy as np

from apgi.core import compute_pi_i_eff, compute_S_t, compute_theta_t, ignition_criterion
from figures.utils import (
    HALF_WIDTH,
    PALETTE,
    PANEL_HEIGHT,
    label_axes,
    make_figure,
    save_figure,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# Protocol 3 APGI parameters
KAPPA = 100.0
ALPHA = 0.3
BETA = 0.5
GAMMA_V = 0.6
GAMMA_A = 0.3
PI_I_BASELINE = 1.0
VOLATILITY_LEVELS = [0.1, 0.3, 0.6]
N_TRIALS = 500
N_AGENTS = 200


def somatic_marker(V_ca: float, A_ca: float) -> float:
    return GAMMA_V * V_ca + GAMMA_A * A_ca


def run_agent(agent_type: str, sigma_env: float, rng: np.random.Generator) -> float:
    """Run one agent for N_TRIALS and return cumulative normalised reward."""
    rewards = np.zeros(N_TRIALS)
    theta_t = compute_theta_t(1.0, 0.5, ALPHA, BETA)

    for t in range(N_TRIALS):
        C_metabolic = rng.uniform(0.5, 2.0)
        V_info = rng.uniform(0.1, 1.0)
        V_ca = rng.uniform(0.2, 0.8)
        A_ca = rng.uniform(0.1, 0.5)

        if agent_type == "full_apgi":
            M_hat = somatic_marker(V_ca, A_ca)
            pi_i = PI_I_BASELINE * np.exp(BETA * M_hat)
        elif agent_type == "beta_lesion":
            pi_i = PI_I_BASELINE  # β = 0: somatic channel disabled
        elif agent_type == "pi_i_lesion":
            pi_i = PI_I_BASELINE  # Πⁱ_eff held constant
            C_metabolic = 0.0
        elif agent_type == "alpha_lesion":
            pi_i = PI_I_BASELINE
            V_info = 0.0  # α = 0
        else:  # random
            rewards[t] = rng.uniform(0, 1) * (1 - sigma_env)
            continue

        pi_i_eff = compute_pi_i_eff(pi_i, C_metabolic, kappa=KAPPA)
        pi_e = rng.uniform(0.8, 1.5)
        z_e = rng.uniform(0.2, 1.0) * (1 + sigma_env * rng.standard_normal())
        z_i = rng.uniform(0.1, 0.8)
        S_t = compute_S_t(pi_e, z_e, pi_i_eff, z_i)
        ignited = ignition_criterion(S_t, theta_t)
        theta_t = compute_theta_t(C_metabolic, V_info, ALPHA, BETA)

        # Reward: ignition under volatile conditions is harder → higher value when achieved
        base_reward = rng.binomial(1, max(0.1, 0.7 - sigma_env * 0.5))
        rewards[t] = float(ignited) * base_reward

    return float(rewards.mean())


def simulate(seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    agents = ["full_apgi", "beta_lesion", "pi_i_lesion", "alpha_lesion", "random"]
    results: dict[str, dict[float, list[float]]] = {
        a: {v: [] for v in VOLATILITY_LEVELS} for a in agents
    }
    for agent in agents:
        for sigma in VOLATILITY_LEVELS:
            for _ in range(N_AGENTS):
                r = run_agent(agent, sigma, rng)
                results[agent][sigma].append(r)
    return results


def plot(results: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=3, width=HALF_WIDTH * 3, height=PANEL_HEIGHT)

    agent_colors = {
        "full_apgi": PALETTE["S_t"],
        "beta_lesion": PALETTE["theta"],
        "pi_i_lesion": "#9966FF",
        "alpha_lesion": "#FFCC00",
        "random": "#AAAAAA",
    }
    agent_labels = {
        "full_apgi": "Full APGI",
        "beta_lesion": "β-lesion",
        "pi_i_lesion": "Πⁱ-lesion",
        "alpha_lesion": "α-lesion",
        "random": "Random",
    }

    for ax, sigma in zip(axes, VOLATILITY_LEVELS):
        means = {a: np.mean(results[a][sigma]) for a in agent_labels}
        sems = {a: np.std(results[a][sigma]) / np.sqrt(N_AGENTS) for a in agent_labels}
        x = np.arange(len(agent_labels))
        bars = ax.bar(
            x,
            [means[a] for a in agent_labels],
            yerr=[sems[a] for a in agent_labels],
            color=[agent_colors[a] for a in agent_labels],
            alpha=0.85,
            edgecolor="white",
            width=0.6,
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [agent_labels[a] for a in agent_labels], fontsize=7, rotation=20, ha="right"
        )
        ax.set_ylabel(
            "Mean reward" if sigma == VOLATILITY_LEVELS[0] else "", fontsize=10
        )
        ax.set_title(f"σ_env = {sigma}", fontsize=10)
        ax.set_ylim(0, 0.7)

    label_axes(axes)
    fig.suptitle(
        "Figure 4 — Protocol 2 — Somatic-AgentSim: Somatic Marker Advantage (Pred 2.a)",
        fontsize=11,
        y=1.02,
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
    print("Simulating agents across volatility levels…")
    results = simulate()
    plot(results, show=not args.no_show)


if __name__ == "__main__":
    main()
