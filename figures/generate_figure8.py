"""Figure 8 — Protocol 7 (DoC-Biomarker) study design (Pred 7.A).

Per OUP-Protocols.txt Figure 8 caption:
  (A) Predicted joint distribution of the two biomarkers -- perturbational
      complexity index (PCI, x-axis) and heartbeat-evoked-potential
      amplitude (HEP, y-axis) -- for the four groups (VS/UWS, N=30; MCS,
      N=30; EMCS, N=20; controls, N=30); ellipses show group mean +/- 1 SD
      and the dashed diagonal is the pre-registered joint-classifier
      decision boundary (target AUC >= 0.80).
  (B) Longitudinal assessment timeline: every group receives EEG/HEP,
      TMS-EEG/PCI and CRS-R at baseline, 3 months and 6 months. EMCS
      occupies the intermediate position in joint HEP-PCI space, making
      the four-group gradient essential for full-gradient validation.

BUG FIX (this revision): the previous generate_figure8.py implemented only
THREE groups (VS/UWS, MCS, Controls; N=50 total) and mislabeled itself
"Protocol 6" in its docstring/title. This rewrite loads the archived
four-group seed dataset data/seeds/sim5_doc_biomarker.npz directly
(VS/UWS N=30, MCS N=30, EMCS N=20, Controls N=30; N=110 total, verified via
its group_labels/subject_id keys), consistent with the audited
Figure-N <-> Protocol-(N-1) numbering (Figure 8 = Protocol 7).

Judgment call / spec ambiguity: no distinct "Figure 8" caption exists
anywhere in OUP-OSF-Preregistration.txt or OUP-Protocols.txt beyond the
Protocol 7 content already assigned to Figure 8 above (OUP-Protocols.txt's
figure captions run Figure 1..Figure 8 in exact 1:1 correspondence with
Protocol 0..Protocol 7 -- there is no ninth figure or separate "Figure 8b").
The previous repo's generate_figure8.py duplicated Protocol 6 content
(mislabeled) rather than implementing Protocol 7; that duplication is
removed here. This script is now the sole implementation of Protocol 7 /
Figure 8, and there is no longer a separate Figure 7 vs Figure 8 collision.

Judgment call on Panel B: the archived sim5_doc_biomarker.npz seed contains
only a single (baseline) cross-sectional assessment per subject -- it does
not include simulated 3-month/6-month follow-up values. Since the spec
requires a longitudinal timeline panel and no seed data exists for the
follow-up timepoints, Panel B extrapolates plausible recovery trajectories
from each group's baseline PCI/HEP means (partial recovery for MCS/EMCS,
near-flat for VS/UWS and Controls) purely for illustrative study-design
purposes, clearly labeled as such -- consistent with the spec's own
"Values are illustrative pre-data predictions" caveat for this figure.

Run:
    python figures/generate_figure8.py
    python figures/generate_figure8.py --no-show   # CI mode
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
from matplotlib.patches import Ellipse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from figures.utils import (  # noqa: E402
    HALF_WIDTH,
    PALETTE,
    PANEL_HEIGHT,
    ensure_seed_dataset,
    label_axes,
    make_figure,
    save_figure,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"
DATA_DIR = pathlib.Path(
    os.environ.get("APGI_DATA_DIR", pathlib.Path(__file__).resolve().parent.parent / "data" / "seeds")
)

GROUP_ORDER = ["VS_UWS", "MCS", "EMCS", "Controls"]
GROUP_DISPLAY = {"VS_UWS": "VS/UWS", "MCS": "MCS", "EMCS": "EMCS", "Controls": "Controls"}
GROUP_N = {"VS_UWS": 30, "MCS": 30, "EMCS": 20, "Controls": 30}
GROUP_COLORS = {
    "VS_UWS": PALETTE["theta"],
    "MCS": "#FFCC00",
    "EMCS": "#FF8C42",
    "Controls": PALETTE["S_t"],
}
PCI_THRESHOLD = 0.31


def load_data(path: pathlib.Path | None = None) -> dict:
    npz_path = path or ensure_seed_dataset(
        DATA_DIR / "sim5_doc_biomarker.npz", "_gen_sim5_doc_biomarker"
    )
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def subject_level_summary(data: dict) -> dict:
    """Collapse trial-level sim5 data to one (hep, pci, group) row per
    subject (N=110).

    Uses the clinical-scale calibrated fields (`pci`, `hep_amplitude_uv`),
    NOT the raw `pci_proxy`/`hep_proxy` fields -- the raw proxies live on an
    internal, uncalibrated precision scale (VS/UWS PCI proxy is exactly 0.0;
    HEP proxy is sub-1 "a.u."), which is exactly the cross-tier scale
    conflation the Protocol 7 figure spec's CRITICAL correction warns
    against (PCI must never be 0 or exceed ~0.8; HEP must be in realistic
    microvolts, not a sub-1 scale).
    """
    sub = data["subject_id"]
    gl = data["group_labels"]
    hep = data["hep_amplitude_uv"]
    pci = data["pci"]

    subj_ids = np.unique(sub)
    out_group, out_hep, out_pci = [], [], []
    for sid in subj_ids:
        m = sub == sid
        out_group.append(gl[m][0])
        out_hep.append(float(hep[m].mean()))
        out_pci.append(float(pci[m].mean()))
    return {
        "subject_id": subj_ids,
        "group": np.array(out_group),
        "hep": np.array(out_hep),
        "pci": np.array(out_pci),
    }


def _draw_confidence_ellipse(ax, x: np.ndarray, y: np.ndarray, color: str, label: str) -> None:
    """Draw a mean +/- 1 SD ellipse (axis-aligned, using per-axis SD; a
    simple and transparent visual summary matching the spec's 'ellipses
    show group mean +/- 1 SD' description)."""
    mean_x, mean_y = x.mean(), y.mean()
    sd_x, sd_y = x.std(), y.std()
    ellipse = Ellipse(
        (mean_x, mean_y), width=2 * sd_x, height=2 * sd_y,
        facecolor=color, edgecolor=color, alpha=0.18, lw=1.8, zorder=2,
    )
    ax.add_patch(ellipse)
    ax.scatter([mean_x], [mean_y], marker="x", s=70, color=color, linewidths=2.2, zorder=4)
    ax.scatter(x, y, s=18, alpha=0.55, color=color, edgecolors="white",
               linewidths=0.3, label=label, zorder=3)


# Axis bounds fixed per the Protocol 7 figure spec's CRITICAL correction:
# PCI is a bounded 0-1 metric and must be shown over 0-0.8 (not extended to
# 1.0), and HEP amplitude must be shown in realistic microvolts (0-~14 uV),
# not compressed to a sub-1 scale.
PCI_AXIS_MAX = 0.8
HEP_AXIS_MAX_UV = 14.0
PCI_STAR = 0.31  # canonical PCI consciousness threshold


def plot_joint_scatter(ax, summary: dict) -> None:
    for g in GROUP_ORDER:
        mask = summary["group"] == g
        _draw_confidence_ellipse(
            ax, summary["pci"][mask], summary["hep"][mask], GROUP_COLORS[g],
            f"{GROUP_DISPLAY[g]} (N={GROUP_N[g]})",
        )

    ax.set_xlim(0, PCI_AXIS_MAX)
    ax.set_ylim(0, HEP_AXIS_MAX_UV)

    # PCI* = 0.31 canonical consciousness threshold (vertical reference line).
    ax.axvline(PCI_STAR, ls=":", lw=1.3, color="#333333", alpha=0.7,
               label=f"PCI* = {PCI_STAR} threshold")

    # Pre-registered joint-classifier decision boundary: an illustrative
    # diagonal separating low-consciousness (VS/UWS, MCS) from
    # high-consciousness (EMCS, Controls) biomarker space (target joint
    # AUC >= 0.80). Drawn in axes-fraction coordinates so it renders as a
    # clean corner-to-corner diagonal regardless of the very different PCI
    # vs. HEP(uV) numeric ranges.
    ax.plot(
        [0.05, 0.95], [0.05, 0.95], transform=ax.transAxes,
        ls="--", lw=1.5, color="#333333",
        label="Decision boundary\n(target joint AUC ≥ 0.80)", zorder=1,
    )

    ax.set_xlabel("PCI (perturbational complexity index)", fontsize=9.5)
    ax.set_ylabel("HEP amplitude (µV)", fontsize=9.5)
    ax.set_title(
        "Pred 7.A — Joint PCI–HEP space separates\nfour DoC groups (N=110)", fontsize=10
    )
    ax.legend(fontsize=6.5, loc="upper left")


def plot_longitudinal(ax, summary: dict) -> None:
    """Illustrative longitudinal recovery trajectory panel (see module
    docstring: no follow-up timepoints exist in the archived seed data;
    trajectories are extrapolated from baseline group means for
    study-design illustration only)."""
    timepoints = ["Baseline", "3 months", "6 months"]
    x = np.arange(len(timepoints))

    # Partial-recovery multipliers per group, applied to a joint composite
    # biomarker score = 0.5*(z-scored PCI) + 0.5*(z-scored HEP), purely
    # illustrative.
    recovery_factor = {"VS_UWS": [1.00, 1.03, 1.05], "MCS": [1.00, 1.18, 1.30],
                        "EMCS": [1.00, 1.10, 1.16], "Controls": [1.00, 1.00, 1.00]}

    pci_all, hep_all = summary["pci"], summary["hep"]
    pci_z = (pci_all - pci_all.mean()) / pci_all.std()
    hep_z = (hep_all - hep_all.mean()) / hep_all.std()
    composite = 0.5 * pci_z + 0.5 * hep_z

    for g in GROUP_ORDER:
        mask = summary["group"] == g
        baseline = composite[mask].mean()
        baseline_sem = composite[mask].std() / np.sqrt(mask.sum())
        trajectory = np.array(recovery_factor[g]) * baseline
        ax.errorbar(
            x, trajectory, yerr=baseline_sem, marker="o", ms=6, lw=1.8,
            color=GROUP_COLORS[g], capsize=4, label=GROUP_DISPLAY[g],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(timepoints, fontsize=9.5)
    ax.set_ylabel("Composite HEP–PCI score (z, illustrative)", fontsize=9.5)
    ax.set_title(
        "Longitudinal assessment timeline\n(EEG/HEP, TMS-EEG/PCI, CRS-R at each timepoint)",
        fontsize=10,
    )
    ax.annotate(
        "Trajectories are illustrative extrapolations from\n"
        "baseline-only seed data (no 3/6-month follow-up\n"
        "measurements exist in the archived dataset).",
        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=6.5, color="#777777",
        style="italic",
    )
    ax.legend(fontsize=7, loc="upper left")


def plot(data: dict, show: bool = True) -> None:
    summary = subject_level_summary(data)
    fig, axes = make_figure(ncols=2, width=HALF_WIDTH * 2.3, height=PANEL_HEIGHT * 1.15)

    plot_joint_scatter(axes[0], summary)
    plot_longitudinal(axes[1], summary)

    label_axes(axes)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure8.pdf")

    if show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 8")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    data = load_data()
    summary = subject_level_summary(data)
    n_total = len(summary["subject_id"])
    counts = {g: int((summary["group"] == g).sum()) for g in GROUP_ORDER}
    print(f"  N={n_total} subjects; group counts: {counts}")
    for g in GROUP_ORDER:
        mask = summary["group"] == g
        print(
            f"    {GROUP_DISPLAY[g]}: PCI={summary['pci'][mask].mean():.3f}  "
            f"HEP={summary['hep'][mask].mean():.3f}"
        )
    plot(data, show=not args.no_show)


if __name__ == "__main__":
    main()
