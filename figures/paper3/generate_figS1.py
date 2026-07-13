"""Paper 3 — Figure S1: Protocol 7 Clinical Study Design (supplementary).

Panel A: 2D scatter schematic (PCI vs HEP) with predicted group clusters.
Panel B: Longitudinal assessment timeline for four groups.

Run:
    python figures/paper3/generate_figS1.py
    python figures/paper3/generate_figS1.py --no-show
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

RNG = np.random.default_rng(99)

# Group PCI/HEP(uV) targets and APGI semantic-palette colours per the
# Protocol 7 figure spec's CRITICAL correction: PCI must stay within 0-0.8
# (never extend to/beyond 1.0) and HEP must be realistic microvolts (not a
# sub-1 normalized scale). Values match the calibrated sim5_doc_biomarker
# clinical-scale fields used by figures/generate_figure8.py (Protocol 7's
# other, "identical design", figure) so the two stay numerically consistent.
GROUPS = [
    {"name": "VS/UWS", "N": 30, "pci": 0.125, "hep": 1.5, "color": "#D7263D"},   # Interoceptive Red
    {"name": "MCS", "N": 30, "pci": 0.30, "hep": 3.75, "color": "#E8A400"},      # Amber
    {"name": "EMCS", "N": 20, "pci": 0.50, "hep": 7.0, "color": "#7B3FE4"},      # Workspace Purple
    {"name": "Controls", "N": 30, "pci": 0.65, "hep": 10.5, "color": "#2E9E5B"}, # Neuromodulator Green
]
PCI_STAR = 0.31  # canonical PCI consciousness threshold
PCI_AXIS_MAX = 0.8
HEP_AXIS_MAX_UV = 14.0

TIMEPOINTS = [
    {"label": "Baseline\n(wk 0)", "x": 0.15},
    {"label": "Follow-up 1\n(3 mo)", "x": 0.50},
    {"label": "Follow-up 2\n(6 mo)", "x": 0.85},
]
ASSESSMENTS = ["EEG/HEP", "TMS-EEG/PCI", "CRS-R"]


def _cov_ellipse(ax, mean, cov, color, nstd=1.0):
    """Draw an n-SD prediction ellipse for a Gaussian cluster."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nstd * np.sqrt(vals)
    ell = mpatches.Ellipse(
        mean,
        width,
        height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.22,
        lw=1.6,
        zorder=3,
    )
    ax.add_patch(ell)


# Per-group (PCI SD, HEP-uV SD) — same order of magnitude as the calibrated
# sim5_doc_biomarker fields; the two axes have very different numeric
# ranges (PCI 0-0.8 vs. HEP 0-14 uV) so a single shared covariance (as for
# a normalized 0-1x0-1 scale) is no longer appropriate.
GROUP_SD = {
    "VS/UWS": (0.02, 0.3),
    "MCS": (0.03, 0.4),
    "EMCS": (0.04, 0.6),
    "Controls": (0.04, 0.7),
}


def draw_scatter(ax):
    # Predicted (not observed) clusters: drawn as mean ± 1 SD ellipses rather
    # than simulated dots, which would imply empirical variability not yet
    # collected for this prospective Protocol 7 trial.
    legend_handles = []
    for g in GROUPS:
        pci_sd, hep_sd = GROUP_SD[g["name"]]
        cov = np.array([[pci_sd ** 2, 0.0], [0.0, hep_sd ** 2]])
        _cov_ellipse(ax, [g["pci"], g["hep"]], cov, g["color"], nstd=1.0)
        ax.plot(
            g["pci"],
            g["hep"],
            "*",
            ms=13,
            color=g["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=5,
        )
        legend_handles.append(
            mpatches.Patch(
                facecolor=g["color"],
                alpha=0.5,
                label=f"{g['name']} (N={g['N']}, mean ± 1 SD)",
            )
        )

    # Decision boundary: an illustrative diagonal separating low-consciousness
    # (VS/UWS, MCS) from high-consciousness (EMCS, Controls) biomarker space
    # (target joint AUC >= 0.80). Drawn in axes-fraction coordinates so it
    # renders as a clean corner-to-corner diagonal regardless of the very
    # different PCI vs. HEP(uV) numeric ranges.
    (bound,) = ax.plot(
        [0.05, 0.95],
        [0.05, 0.95],
        "k--",
        lw=1.5,
        alpha=0.6,
        transform=ax.transAxes,
        label="Joint AUC ≥ 0.80 decision boundary",
    )
    legend_handles.append(bound)

    # PCI* = 0.31 canonical consciousness threshold (vertical reference line).
    thresh = ax.axvline(
        PCI_STAR, ls=":", lw=1.3, color="#333333", alpha=0.7,
        label=f"PCI* = {PCI_STAR} threshold",
    )
    legend_handles.append(thresh)

    ax.set_xlabel("PCI (perturbational complexity index)", fontsize=10)
    ax.set_ylabel("HEP amplitude (µV)", fontsize=10)
    ax.set_xlim(0, PCI_AXIS_MAX)
    ax.set_ylim(0, HEP_AXIS_MAX_UV)
    ax.set_title(
        "A — Predicted group clusters\n(PCI × HEP joint biomarker)",
        fontsize=9,
        fontweight="bold",
    )
    ax.legend(handles=legend_handles, fontsize=7.5, loc="upper left", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Pre-registered quantitative falsification criterion stated on the figure face.
    ax.text(
        0.50,
        -0.02,
        "Pre-registered target: joint PCI×HEP AUC ≥ 0.80 for 3-month GCS-S prediction;\n"
        "formal test that HEP and PCI provide statistically independent predictive information.",
        ha="center",
        fontsize=6.2,
        color="#555555",
        style="italic",
        transform=ax.transAxes,
    )


def draw_timeline(ax):
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.1, 1.0)
    ax.axis("off")
    ax.set_title("B — Longitudinal assessment timeline", fontsize=9, fontweight="bold")

    # x positions for timepoints
    tp_x = [tp["x"] for tp in TIMEPOINTS]

    # Group rows
    ROW_Y = {g["name"]: 0.75 - i * 0.20 for i, g in enumerate(GROUPS)}

    # Header
    for tp in TIMEPOINTS:
        ax.text(
            tp["x"],
            0.96,
            tp["label"],
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="#333333",
        )
        ax.axvline(
            tp["x"], ymin=0.02, ymax=0.94, lw=0.8, color="#dddddd", ls="--", zorder=1
        )

    for g in GROUPS:
        y = ROW_Y[g["name"]]
        ax.text(
            0.02,
            y,
            f"{g['name']}\n(N={g['N']})",
            ha="left",
            va="center",
            fontsize=8,
            color=g["color"],
            fontweight="bold",
        )

        for x in tp_x:
            circ = mpatches.Circle(
                (x, y),
                0.025,
                facecolor=g["color"],
                edgecolor="white",
                lw=1.0,
                alpha=0.85,
                zorder=3,
            )
            ax.add_patch(circ)
            # Assessments
            for j, assess in enumerate(ASSESSMENTS):
                ax.text(
                    x,
                    y - 0.045 - j * 0.028,
                    f"• {assess}",
                    ha="center",
                    fontsize=5.5,
                    color="#555555",
                )

        # Horizontal connector
        ax.plot(tp_x, [y] * len(tp_x), lw=1.2, color=g["color"], alpha=0.40, zorder=2)

    # Assessment legend
    ax.text(
        0.50,
        0.05,
        "Assessments at each timepoint: " + " | ".join(ASSESSMENTS),
        ha="center",
        fontsize=7,
        color="#555555",
        style="italic",
    )


def plot(show: bool = True) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_scatter(ax1)
    draw_timeline(ax2)
    label_axes([ax1, ax2])
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figS1_protocol7_clinical_design.pdf")
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
