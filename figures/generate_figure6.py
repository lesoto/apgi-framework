"""Figure 6 — Protocol 5 (Insula-TMS): causal-neuromodulation targets and
predicted dissociation (Pred 5.A-Pred 5.D).

Per OUP-Protocols.txt Figure 6 caption:
  (A) Stimulation montage on a lateral cortical surface: anterior insula
      (aINS; tFUS or deep TMS), dorsolateral prefrontal cortex (dlPFC),
      posterior parietal cortex (PPC) and vertex sham, with MNI
      coordinates.
  (B) Perturbational Complexity Index (PCI) reduced by aINS and dlPFC/PPC
      stimulation relative to vertex (Pred 5.A).
  (C) HEP-P3b interoceptive coupling abolished under aINS stimulation but
      preserved under dlPFC -- the core double dissociation (Pred 5.B).
  (D) Baseline interoceptive-accuracy x site interaction: high-IA
      participants show the largest aINS-driven PCI reduction, absent for
      dlPFC/PPC (Pred 5.C-D).

This content replaces/moves the previous generate_figure6.py (which was
mislabeled "Protocol 4" in its own docstring/title, and used a degenerate
hand-rolled simulation in which every condition -- including vertex sham --
sat far below the PCI = 0.31 consciousness threshold). Fix: load the
archived seed dataset data/seeds/sim8_tms_pci.npz directly. In the seed
data, vertex sham correctly sits ABOVE the 0.31 threshold (normal waking
consciousness baseline; mean PCI ~= 0.47), while aINS stimulation reduces
PCI below/near threshold (~0.28) and dlPFC sits in between (~0.39) --
consistent with Pred 5.A's "aINS ~20%, dlPFC/PPC 15-25%" reduction pattern
relative to vertex, and with the requirement that only active stimulation,
not sham, reduces PCI below the consciousness threshold.

Run:
    python figures/generate_figure6.py
    python figures/generate_figure6.py --no-show   # CI mode
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import matplotlib.patches as mpatches
import numpy as np

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

PCI_THRESHOLD = 0.31
SITE_ORDER = ["vertex", "dlPFC", "aINS"]
SITE_LABELS = {"vertex": "Vertex\n(sham)", "dlPFC": "dlPFC", "aINS": "aINS"}
SITE_COLORS = {"vertex": "#AAAAAA", "dlPFC": PALETTE["theta"], "aINS": PALETTE["S_t"]}

# MNI coordinates for the montage panel (Protocol 5 spec)
MONTAGE_SITES = {
    "dlPFC": {"mni": "(-44, 36, 24)", "xy": (0.30, 0.72), "color": PALETTE["theta"]},
    "PPC": {"mni": "(-40, -60, 44)", "xy": (0.58, 0.66), "color": "#9966FF"},
    "aINS": {"mni": "(±34, 14, 0)", "xy": (0.46, 0.42), "color": PALETTE["S_t"]},
    "Vertex\n(sham)": {"mni": "(0, 0, 80)", "xy": (0.44, 0.94), "color": "#AAAAAA"},
}


def load_data(path: pathlib.Path | None = None) -> dict:
    npz_path = path or ensure_seed_dataset(DATA_DIR / "sim8_tms_pci.npz", "_gen_sim8_tms_pci")
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _draw_brain_outline(ax) -> None:
    """Simplified lateral cortical-surface silhouette for the montage panel."""
    brain = mpatches.Ellipse(
        (0.45, 0.55), 0.65, 0.55, angle=-8, facecolor="#f2f2f2",
        edgecolor="#666666", lw=1.5, zorder=1,
    )
    ax.add_patch(brain)
    # Simple sulcal accents for visual context (not anatomically precise).
    for cx, cy, w, h, ang in [
        (0.35, 0.60, 0.18, 0.05, 20), (0.5, 0.50, 0.22, 0.05, -15),
        (0.55, 0.65, 0.15, 0.04, 30),
    ]:
        ax.add_patch(mpatches.Ellipse((cx, cy), w, h, angle=ang, facecolor="none",
                                       edgecolor="#cccccc", lw=1.0, zorder=2))
    # Nose/front indicator
    ax.annotate("anterior", xy=(0.08, 0.55), fontsize=7, color="#888888", style="italic")
    ax.annotate("posterior", xy=(0.80, 0.55), fontsize=7, color="#888888", style="italic")


def plot_montage(ax) -> None:
    _draw_brain_outline(ax)
    for label, info in MONTAGE_SITES.items():
        x, y = info["xy"]
        ax.scatter([x], [y], s=140, color=info["color"], edgecolor="black",
                   linewidths=1.2, zorder=5)
        ax.annotate(
            f"{label}\nMNI {info['mni']}", xy=(x, y), xytext=(x, y - 0.18),
            ha="center", va="top", fontsize=7.2, fontweight="bold",
            color=info["color"],
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Stimulation montage (lateral view)\naINS via tFUS/deep TMS; vertex sham control",
        fontsize=10,
    )


def plot(data: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=4, width=HALF_WIDTH * 4, height=PANEL_HEIGHT)

    site = data["site"]
    pci = data["pci"]
    coupling = data["hep_pci_coupling"]
    tertile = data["pi_i_tertile"]

    # Panel A: stimulation montage
    plot_montage(axes[0])

    # Panel B: PCI by site (Pred 5.A) — vertex sham sits AT/ABOVE the 0.31
    # consciousness threshold; only active stimulation reduces PCI.
    ax = axes[1]
    means = [pci[site == s].mean() for s in SITE_ORDER]
    sems = [pci[site == s].std() / np.sqrt((site == s).sum()) for s in SITE_ORDER]
    bars = ax.bar(
        [SITE_LABELS[s] for s in SITE_ORDER], means, yerr=sems,
        color=[SITE_COLORS[s] for s in SITE_ORDER], alpha=0.85, edgecolor="white",
        width=0.5, capsize=5,
    )
    ax.axhline(PCI_THRESHOLD, ls="--", lw=1.2, color="black", alpha=0.7,
               label=f"PCI consciousness\nthreshold = {PCI_THRESHOLD}")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("PCI", fontsize=9.5)
    ax.set_ylim(0, 0.65)
    ax.set_title("Pred 5.A — PCI reduced by aINS,\ndlPFC vs. vertex sham", fontsize=10)
    ax.legend(fontsize=7)

    # Panel C: HEP-P3b/PCI coupling by site (Pred 5.B double dissociation)
    ax = axes[2]
    coupling_means = [coupling[site == s].mean() for s in SITE_ORDER]
    coupling_sems = [coupling[site == s].std() / np.sqrt((site == s).sum()) for s in SITE_ORDER]
    bars = ax.bar(
        [SITE_LABELS[s] for s in SITE_ORDER], coupling_means, yerr=coupling_sems,
        color=[SITE_COLORS[s] for s in SITE_ORDER], alpha=0.85, edgecolor="white",
        width=0.5, capsize=5,
    )
    ax.axhline(0, ls="-", lw=0.8, color="black", alpha=0.3)
    ax.set_ylabel("HEP–PCI coupling (r)", fontsize=9.5)
    ax.set_title(
        "Pred 5.B — Coupling abolished by aINS,\npreserved under dlPFC/vertex", fontsize=10
    )

    # Panel D: PCI reduction (vs vertex) by pi_i tertile x site (Pred 5.C-D)
    ax = axes[3]
    vertex_by_tertile = {t: pci[(site == "vertex") & (tertile == t)].mean() for t in np.unique(tertile)}
    tertile_labels = ["Low Πⁱ", "Mid Πⁱ", "High Πⁱ"]
    x = np.arange(len(tertile_labels))
    width = 0.35
    for i, s in enumerate(["dlPFC", "aINS"]):
        reductions = [
            vertex_by_tertile[t] - pci[(site == s) & (tertile == t)].mean()
            for t in np.unique(tertile)
        ]
        ax.bar(x + (i - 0.5) * width, reductions, width, color=SITE_COLORS[s],
               alpha=0.85, edgecolor="white", label=SITE_LABELS[s].replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(tertile_labels, fontsize=9)
    ax.set_ylabel("PCI reduction vs. vertex sham", fontsize=9.5)
    ax.set_title(
        "Pred 5.C–D — aINS PCI reduction scales\nwith baseline Πⁱ; absent for dlPFC", fontsize=10
    )
    ax.legend(fontsize=7)

    label_axes(axes)
    fig.suptitle(
        "Figure 6 — Protocol 5 — Insula-TMS: Causal Dissociation of Interoceptive and Workspace Gating (Pred 5.A–Pred 5.D)",
        fontsize=11,
        y=1.03,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "figure6.pdf")

    if show:
        import matplotlib.pyplot as plt
        plt.show()
    import matplotlib.pyplot as plt
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 6")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    data = load_data()
    site = data["site"]
    pci = data["pci"]
    print(
        "  PCI by site: "
        + ", ".join(f"{s}={pci[site == s].mean():.3f}" for s in SITE_ORDER)
        + f"  (threshold={PCI_THRESHOLD})"
    )
    plot(data, show=not args.no_show)


if __name__ == "__main__":
    main()
