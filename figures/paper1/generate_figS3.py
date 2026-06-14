"""Paper 1 — Figure S3: Protocol 1 Trial-Timeline Schematic (Appendix D.1).

Cardiac-phase EEG paradigm: R-peak → cardiac-phase window → Gabor onset
→ response → HEP epoch → P3b epoch. Three condition tracks.

Run:
    python figures/paper1/generate_figS3.py
    python figures/paper1/generate_figS3.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import label_axes, save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

CONDITIONS = [
    {"name": "Interoceptive focus", "color": "#2166ac", "y": 2.6},
    {"name": "Exteroceptive focus",  "color": "#4dac26", "y": 1.6},
    {"name": "Dual-task",            "color": "#d6604d", "y": 0.6},
]

EVENTS = [
    {"label": "R-peak\ndetect", "t": 0,    "dur": 0.05},
    {"label": "Cardiac-\nphase\nwindow",   "t": 0.08, "dur": 0.22},
    {"label": "Gabor\nonset\n(QUEST)",      "t": 0.30, "dur": 0.05},
    {"label": "Response\nwindow",           "t": 0.35, "dur": 0.25},
    {"label": "HEP\nepoch\n250–400 ms",    "t": 0.25, "dur": 0.18},
    {"label": "P3b\nepoch\n300–600 ms",    "t": 0.60, "dur": 0.30},
]

EPOCHS = [
    {"label": "HEP epoch (250–400 ms post-R)", "t": 0.25, "dur": 0.18, "color": "#9ecae1"},
    {"label": "P3b epoch (300–600 ms post-stim)", "t": 0.60, "dur": 0.30, "color": "#a1d99b"},
]

PHASES = [
    {"label": "R-peak\ndetection", "t": 0.00, "dur": 0.08, "color": "#fee8c8"},
    {"label": "Cardiac-phase\nwindow assignment\n(systole vs diastole)", "t": 0.08, "dur": 0.22, "color": "#fdd49e"},
    {"label": "Gabor patch\nonset (QUEST)", "t": 0.30, "dur": 0.05, "color": "#fc8d59"},
    {"label": "Response\nwindow", "t": 0.35, "dur": 0.25, "color": "#d7301f"},
    {"label": "ITI", "t": 0.60, "dur": 0.40, "color": "#f0f0f0"},
]


def plot(show: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 4.0)
    ax.axis("off")

    BAR_H = 0.45

    # ── Condition tracks ───────────────────────────────────────────────────
    for cond in CONDITIONS:
        y = cond["y"]
        for ph in PHASES:
            rect = mpatches.FancyBboxPatch(
                (ph["t"], y),
                ph["dur"],
                BAR_H,
                boxstyle="round,pad=0.005",
                linewidth=1.0,
                edgecolor="#555555",
                facecolor=ph["color"],
                alpha=0.80,
                zorder=3,
            )
            ax.add_patch(rect)

        # Condition label
        ax.text(-0.04, y + BAR_H / 2, cond["name"], ha="right", va="center",
                fontsize=8, color=cond["color"], fontweight="bold")

        # Epoch overlays
        for ep in EPOCHS:
            rect2 = mpatches.FancyBboxPatch(
                (ep["t"], y - 0.03),
                ep["dur"],
                BAR_H + 0.06,
                boxstyle="round,pad=0.003",
                linewidth=2.0,
                edgecolor=ep["color"],
                facecolor="none",
                zorder=5,
            )
            ax.add_patch(rect2)

    # ── Phase labels below bottom track ───────────────────────────────────
    for ph in PHASES:
        mid = ph["t"] + ph["dur"] / 2
        ax.text(mid, 0.30, ph["label"], ha="center", va="top", fontsize=6.8,
                color="#444444", multialignment="center")

    # ── Epoch legend ───────────────────────────────────────────────────────
    for ep in EPOCHS:
        mid = ep["t"] + ep["dur"] / 2
        ax.text(mid, 3.65, ep["label"], ha="center", va="bottom",
                fontsize=7, color=ep["color"], fontweight="bold")

    # ── Timeline x-axis ───────────────────────────────────────────────────
    ax.annotate("", xy=(1.03, -0.05), xytext=(-0.02, -0.05),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    tick_labels = {0: "0 ms", 0.08: "~80", 0.30: "~300", 0.35: "~350",
                   0.60: "~600", 1.00: "~1000 ms"}
    for t, lbl in tick_labels.items():
        ax.text(t, -0.10, lbl, ha="center", va="top", fontsize=7.5)
        ax.plot([t, t], [-0.05, -0.02], lw=1, color="#333333")
    ax.text(0.5, -0.18, "Time post R-peak", ha="center", fontsize=9)

    # ── Systole/Diastole annotation ────────────────────────────────────────
    ax.annotate("systole", xy=(0.14, 3.45), fontsize=7.5, color="#555555",
                ha="center", style="italic")
    ax.annotate("vs. diastole", xy=(0.25, 3.45), fontsize=7.5, color="#555555",
                ha="center", style="italic")

    ax.set_title(
        "Figure S3 — Protocol 1 Trial-Timeline Schematic (Cardiac-Phase EEG)\n"
        "Appendix D.1, before Table S4",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figS3_protocol1_trial_timeline.pdf")
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
