"""Paper 1 — Figure 4: Onset Ignition Timeline (0-500 ms).

Five-phase swim-lane causal cascade from stimulus onset to global broadcast,
matching the APGI-Figures.pdf Figure 4 spec: phase chevrons across the top,
three annotation rows beneath (neural/thalamo-cortical substrate, APGI
variable(s), EEG/MEG marker), and a vertical accent line marking the
threshold-crossing moment (t = 250 ms).

Run:
    python figures/paper1/generate_fig4.py
    python figures/paper1/generate_fig4.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# Five phases, matching the reference spec's timing exactly (0-50, 50-120,
# 120-250, 250-320, 320-500 ms). Threshold crossing (S_t > theta_t) happens
# at t = 250 ms, the boundary between phase 3 and phase 4.
PHASES = [
    {
        "name": "1. Error\nGeneration",
        "t_start": 0,
        "t_end": 50,
        "substrate": "Feedforward\nsensory input;\nerror in\nearly cortex",
        "variable": r"Pred. error $\varepsilon^i$" "\n(local)\nHigh " r"$\to$" " Low",
        "eeg_marker": "Early Gamma /\nN1\n~30-80 ms\n(sensory gain)",
        "color": "#2166ac",
    },
    {
        "name": "2. Precision\nWeighting",
        "t_start": 50,
        "t_end": 120,
        "substrate": "Neuromodulatory\ngating\n(interoceptive\nprecision signals)",
        "variable": r"Interoceptive $\Pi^i$" "\n(gain control)\nLow " r"$\to$" " High",
        "eeg_marker": "Alpha\nSuppression\n~80-150 ms\n(attentional gain)",
        "color": "#d6604d",
    },
    {
        "name": r"3. $S_t$" "\nAccumulation",
        "t_start": 120,
        "t_end": 250,
        "substrate": "Recurrent loops in\nassociation cortex",
        "variable": r"Salience evidence $S_t$" "\n(accumulation)\nRises over time",
        "eeg_marker": "N2 / Sustained Gamma\n~150-250 ms\n(accumulation phase)",
        "color": "#4dac26",
    },
    {
        "name": "4. Ignition\nDecision",
        "t_start": 250,
        "t_end": 320,
        "substrate": "Thalamic gate\nevaluates " r"$S_t$" "\nvs. threshold " r"$\theta_t$",
        "variable": r"Ignition if $S_t > \theta_t$" "\n(threshold\ncrossing)",
        "eeg_marker": "P3b Onset\n~250-320 ms\n(ignition)",
        "color": "#e6a817",
    },
    {
        "name": "5. Global\nBroadcast",
        "t_start": 320,
        "t_end": 500,
        "substrate": "Cortico-thalamo-\ncortical broadcast via\nGlobal Workspace",
        "variable": r"Global evidence $G_t$" "\n(workspace\nstabilization)\nSustained High",
        "eeg_marker": "Sustained Late\nPositivity\n~320-500 ms\n(global broadcast)",
        "color": "#7b3294",
    },
]

THRESHOLD_T = 250  # ms — S_t > theta_t crossing moment (phase 3 -> 4 boundary)

ROW_LABELS = [
    "Neural /\nThalamo-cortical\nSubstrate",
    r"$S_t, \theta_t$" "\nAPGI\nVariable(s)",
    "EEG / MEG\nMarker",
]


def plot(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.5))
    ax.set_xlim(-110, 560)
    ax.set_ylim(-3.7, 2.3)
    ax.axis("off")

    # ── Phase chevrons (top row) ────────────────────────────────────────────
    CHEVRON_Y = 1.0
    CHEVRON_H = 0.8

    for ph in PHASES:
        rect = mpatches.FancyBboxPatch(
            (ph["t_start"], CHEVRON_Y),
            ph["t_end"] - ph["t_start"],
            CHEVRON_H,
            boxstyle="round,pad=2",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor=ph["color"],
            alpha=0.88,
            zorder=3,
        )
        ax.add_patch(rect)
        mid = (ph["t_start"] + ph["t_end"]) / 2
        ax.text(
            mid,
            CHEVRON_Y + CHEVRON_H / 2,
            ph["name"],
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    # ── Three annotation rows beneath the chevrons ──────────────────────────
    ROW_Y = [0.35, -1.05, -2.45]  # substrate, APGI variable, EEG/MEG marker
    ROW_H = 1.15

    for row_y, label in zip(ROW_Y, ROW_LABELS):
        ax.text(
            -120,
            row_y + ROW_H / 2 - 0.05,
            label,
            ha="right",
            va="center",
            fontsize=7.5,
            color="#555555",
            fontweight="bold",
        )

    for ph in PHASES:
        mid = (ph["t_start"] + ph["t_end"]) / 2
        for row_y, key in zip(ROW_Y, ["substrate", "variable", "eeg_marker"]):
            ax.text(
                mid,
                row_y + ROW_H / 2 - 0.05,
                ph[key],
                ha="center",
                va="top",
                fontsize=6.0,
                color="#222222",
                linespacing=1.35,
            )
        # thin separators between phase columns across all rows
        ax.plot(
            [ph["t_start"], ph["t_start"]],
            [ROW_Y[-1] - 0.05, CHEVRON_Y],
            color="#dddddd",
            lw=0.8,
            zorder=1,
        )

    # ── Time axis ────────────────────────────────────────────────────────────
    AXIS_Y = ROW_Y[-1] - 0.15
    ax.annotate(
        "",
        xy=(540, AXIS_Y),
        xytext=(-10, AXIS_Y),
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"),
    )
    for t in [0, 50, 120, 250, 320, 500]:
        ax.plot([t, t], [AXIS_Y - 0.04, AXIS_Y + 0.04], lw=1, color="#333333")
        ax.text(t, AXIS_Y - 0.10, f"{t}", ha="center", va="top", fontsize=7.5)
    ax.text(250, AXIS_Y - 0.42, "Time (ms)", ha="center", fontsize=10)

    # ── Threshold-crossing accent line (S_t > theta_t at t = 250 ms) ────────
    ax.plot(
        [THRESHOLD_T, THRESHOLD_T],
        [CHEVRON_Y + CHEVRON_H, ROW_Y[-1] - 0.15],
        color="#333333",
        lw=1.6,
        linestyle=(0, (4, 2)),
        zorder=6,
    )
    ax.text(
        THRESHOLD_T + 8,
        CHEVRON_Y + CHEVRON_H + 0.55,
        r"$S_t > \theta_t$ threshold crossing",
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color="#333333",
    )

    # ── Legend: exactly one swatch per distinct colour actually used, each
    # with an explicit text label (no duplicate/unlabeled swatches). ────────
    for ph in PHASES:
        ax.plot([], [], color=ph["color"], lw=8, label=ph["name"].replace("\n", " "))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=5,
        fontsize=6.8,
        frameon=False,
        title="Phase (colour key)",
        title_fontsize=7.5,
    )

    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "fig4_onset_ignition_timeline.pdf")
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
