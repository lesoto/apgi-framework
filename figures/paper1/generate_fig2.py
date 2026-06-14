"""Paper 1 — Figure 2: Onset Ignition Timeline (0–500 ms).

Six-phase causal cascade from stimulus onset to global broadcast.
Not a waveform display — a phase-annotated causal cascade diagram.

Run:
    python figures/paper1/generate_fig2.py
    python figures/paper1/generate_fig2.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

PHASES = [
    {
        "name": "Phase 1\nError Generation",
        "t_start": 0,
        "t_end": 80,
        "substrate": "L2/3 mismatch",
        "variable": r"$\varepsilon^e, \varepsilon^i$",
        "eeg_marker": "MMN / N1\n(~80 ms)",
        "pharm": None,
        "color": "#6baed6",
    },
    {
        "name": "Phase 2\nPrecision Gating",
        "t_start": 80,
        "t_end": 160,
        "substrate": "PV+/SST+/VIP+",
        "variable": r"$\Pi^e, \Pi^i_{\mathrm{eff}}$",
        "eeg_marker": "P1 / N2\n(~150 ms)",
        "pharm": "scopolamine",
        "color": "#3182bd",
    },
    {
        "name": "Phase 3\nSomatic-Marker\nAmplification",
        "t_start": 160,
        "t_end": 230,
        "substrate": "vmPFC–VIP",
        "variable": r"$\beta_{\mathrm{SM}}$",
        "eeg_marker": "HEP peak\n(250–400 ms)",
        "pharm": None,
        "color": "#08519c",
    },
    {
        "name": r"Phase 4  $S_t$ Accumulation",
        "t_start": 230,
        "t_end": 350,
        "substrate": "L5 pyramidal",
        "variable": r"$S_t > \theta_t$",
        "eeg_marker": "P3b onset\n(~300 ms)",
        "pharm": "propranolol",
        "color": "#4dac26",
    },
    {
        "name": "Phase 5\nIgnition / Global\nBroadcast",
        "t_start": 350,
        "t_end": 450,
        "substrate": "frontoparietal–\nthalamic workspace",
        "variable": r"$P(B_t{=}1)$",
        "eeg_marker": "P3b plateau\n(300–600 ms)",
        "pharm": None,
        "color": "#d6604d",
    },
    {
        "name": "Phase 6\nRefractory\nReset",
        "t_start": 450,
        "t_end": 550,
        "substrate": "threshold re-elevation",
        "variable": r"$\theta_t \uparrow$",
        "eeg_marker": "alpha rebound\n(>500 ms)",
        "pharm": None,
        "color": "#a50f15",
    },
]


def plot(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-20, 580)
    ax.set_ylim(-2.2, 5.5)
    ax.axis("off")

    # ── Phase bars ─────────────────────────────────────────────────────────
    BAR_Y = 3.0
    BAR_H = 0.7

    for ph in PHASES:
        rect = mpatches.FancyBboxPatch(
            (ph["t_start"], BAR_Y),
            ph["t_end"] - ph["t_start"],
            BAR_H,
            boxstyle="round,pad=2",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor=ph["color"],
            alpha=0.85,
            zorder=3,
        )
        ax.add_patch(rect)

        mid = (ph["t_start"] + ph["t_end"]) / 2
        ax.text(
            mid,
            BAR_Y + BAR_H / 2,
            ph["name"],
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="white",
            zorder=4,
        )

        # Neural substrate annotation (above bar)
        ax.text(
            mid,
            BAR_Y + BAR_H + 0.15,
            ph["substrate"],
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#444444",
            style="italic",
        )

        # APGI variable (above substrate)
        ax.text(
            mid,
            BAR_Y + BAR_H + 0.70,
            ph["variable"],
            ha="center",
            va="bottom",
            fontsize=8,
            color=ph["color"],
        )

        # EEG/MEG marker below bar
        ax.text(
            mid,
            BAR_Y - 0.25,
            ph["eeg_marker"],
            ha="center",
            va="top",
            fontsize=6.5,
            color="#222222",
        )

        # Pharmacological probe (marginal)
        if ph["pharm"]:
            ax.text(
                mid,
                BAR_Y - 1.0,
                f"⊗ {ph['pharm']}",
                ha="center",
                va="top",
                fontsize=7,
                color="#7b3294",
                fontweight="bold",
            )

    # ── Axis / time line ───────────────────────────────────────────────────
    ax.annotate(
        "",
        xy=(560, BAR_Y - 1.6),
        xytext=(-10, BAR_Y - 1.6),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"),
    )
    for t in [0, 100, 200, 300, 400, 500]:
        ax.text(t, BAR_Y - 1.75, f"{t} ms", ha="center", va="top", fontsize=8)
        ax.plot([t, t], [BAR_Y - 1.6, BAR_Y - 1.55], lw=1, color="#333333")

    ax.text(280, BAR_Y - 2.1, "Time (ms)", ha="center", fontsize=10)

    # ── Layer labels ───────────────────────────────────────────────────────
    ax.text(
        -15, BAR_Y + BAR_H + 0.70, "APGI\nvariable", ha="right", fontsize=7, va="center", color="#666666"
    )
    ax.text(
        -15, BAR_Y + BAR_H + 0.15, "Neural\nsubstrate", ha="right", fontsize=7, va="center", color="#666666"
    )
    ax.text(-15, BAR_Y - 0.50, "EEG/MEG\nmarker", ha="right", fontsize=7, va="center", color="#666666")
    ax.text(-15, BAR_Y - 1.0, "Pharm.\nprobe", ha="right", fontsize=7, va="center", color="#7b3294")

    ax.set_title(
        "Figure 2 — Onset Ignition Timeline: Causal Cascade 0–500 ms\n"
        "(stimulus onset → global broadcast)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig2_onset_ignition_timeline.pdf")
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
