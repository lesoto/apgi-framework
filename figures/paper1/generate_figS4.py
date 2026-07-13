"""Paper 1 — Figure S4: Protocol 3 fMRI Trial-Structure Schematic (Appendix D.3).

Block-event hybrid: foreperiod → decision → outcome → ITI.
Two tracks: anticipatory window (Πⁱ_eff via vmPFC–insula) vs. outcome (εⁱ).
SCR trace below event markers.

Run:
    python figures/paper1/generate_figS4.py
    python figures/paper1/generate_figS4.py --no-show
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
from figures.utils import label_axes, save_figure

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"


def plot(show: bool = True) -> None:
    fig, (ax_top, ax_scr) = plt.subplots(
        2, 1, figsize=(13, 6), gridspec_kw={"height_ratios": [2.5, 1]}
    )

    # ── Top: two-track trial schematic ────────────────────────────────────
    ax = ax_top
    ax.set_xlim(-0.1, 8.5)
    ax.set_ylim(-0.3, 3.2)
    ax.axis("off")

    # Track definitions
    TRACKS = [
        {
            "y": 2.1,
            "label": r"Anticipatory window ($\Pi^i_{\mathrm{eff}}$ via vmPFC–insula)",
            "color": "#2166ac",
            "phases": [
                {
                    "t": 0.0,
                    "dur": 2.0,
                    "label": "Foreperiod\n(0 ms jitter)",
                    "color": "#cce5ff",
                },
                {"t": 2.0, "dur": 1.0, "label": "Decision", "color": "#6baed6"},
                {
                    "t": 3.0,
                    "dur": 2.0,
                    "label": "Foreperiod\n(2–4 s jitter)",
                    "color": "#deebf7",
                },
                {"t": 5.0, "dur": 1.0, "label": "Decision", "color": "#6baed6"},
                {
                    "t": 6.0,
                    "dur": 1.0,
                    "label": "Outcome",
                    "color": "#08519c",
                    "fc": "white",
                },
                {"t": 7.0, "dur": 1.2, "label": "ITI", "color": "#f0f0f0"},
            ],
        },
        {
            "y": 0.6,
            "label": r"Outcome window ($\varepsilon^i$ encoding)",
            "color": "#d6604d",
            "phases": [
                {
                    "t": 0.0,
                    "dur": 3.0,
                    "label": "Foreperiod + Decision",
                    "color": "#fde0d9",
                },
                {
                    "t": 3.0,
                    "dur": 3.0,
                    "label": "Foreperiod + Decision",
                    "color": "#fde0d9",
                },
                {
                    "t": 6.0,
                    "dur": 1.0,
                    "label": "Outcome\n(εⁱ encoded)",
                    "color": "#d6604d",
                },
                {"t": 7.0, "dur": 1.2, "label": "ITI", "color": "#f0f0f0"},
            ],
        },
    ]

    def _text_color(fill: str) -> str:
        """Black text on light fills, white on dark — keeps every label
        (notably the light-grey ITI segment) legible at print resolution."""
        r, g, b = mcolors.to_rgb(fill)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.6 else "white"

    BAR_H = 0.55
    for track in TRACKS:
        y = track["y"]
        for ph in track["phases"]:
            fc = ph.get("fc", ph["color"])
            rect = mpatches.FancyBboxPatch(
                (ph["t"], y),
                ph["dur"],
                BAR_H,
                boxstyle="round,pad=0.03",
                linewidth=1.2,
                edgecolor=track["color"],
                facecolor=fc,
                alpha=0.85,
                zorder=3,
            )
            ax.add_patch(rect)
            ax.text(
                ph["t"] + ph["dur"] / 2,
                y + BAR_H / 2,
                ph["label"],
                ha="center",
                va="center",
                fontsize=7,
                color=_text_color(fc),
                zorder=4,
                multialignment="center",
            )

        ax.text(
            -0.08,
            y + BAR_H / 2,
            track["label"],
            ha="right",
            va="center",
            fontsize=7.5,
            color=track["color"],
            fontweight="bold",
            transform=ax.transData,
            wrap=True,
        )

    # x-axis
    ax.annotate(
        "",
        xy=(8.4, -0.1),
        xytext=(-0.05, -0.1),
        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333"),
    )
    for t, lbl in {0: "0 s", 2: "2", 3: "3", 5: "5", 6: "6", 7: "7", 8: "8 s"}.items():
        ax.text(t, -0.18, lbl, ha="center", fontsize=7.5)
        ax.plot([t, t], [-0.10, -0.07], lw=1, color="#333")
    ax.text(4.2, -0.28, "Time (s)", ha="center", fontsize=9)

    # ── Bottom: SCR trace ─────────────────────────────────────────────────
    ax_scr.set_xlim(-0.1, 8.5)
    ax_scr.set_ylim(-0.1, 1.3)
    t_scr = np.linspace(0, 8.2, 500)

    def scr_kernel(t_onset, amp=1.0):
        t_rel = t_scr - t_onset
        resp = np.where(t_rel > 0, amp * t_rel * np.exp(-t_rel / 1.5), 0)
        return resp

    scr = scr_kernel(2.0, amp=0.6) + scr_kernel(5.0, amp=0.6) + scr_kernel(6.0, amp=1.0)
    scr = scr / scr.max()
    ax_scr.plot(t_scr, scr, lw=1.8, color="#7b3294")
    ax_scr.fill_between(t_scr, scr, alpha=0.15, color="#7b3294")
    ax_scr.axvline(6.0, lw=1.2, ls="--", color="#d6604d", alpha=0.7)
    ax_scr.text(6.05, 0.9, "Outcome\nSCR peak", fontsize=7, color="#d6604d")
    ax_scr.set_ylabel("SCR\n(norm.)", fontsize=8)
    ax_scr.set_xlabel("Time (s)", fontsize=9)
    ax_scr.spines["top"].set_visible(False)
    ax_scr.spines["right"].set_visible(False)

    label_axes([ax_top, ax_scr])
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figS4_protocol3_fmri_trial.pdf")
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
