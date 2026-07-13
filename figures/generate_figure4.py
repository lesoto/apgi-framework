"""Figure 4 — Protocol 3 (Anticipation-fMRI): predicted vmPFC-posterior-insula
anticipatory coupling (Pred 3.B-Pred 3.D).

Per OUP-Protocols.txt Figure 4 caption:
  (A) vmPFC-pIC PPI coupling is stronger during anticipation than at outcome
      (Pred 3.B).
  (B) vmPFC BOLD is driven by option expected value (valence), not by
      sensory contrast (Pred 3.C).
  (C) Coupling is present under a long foreperiod (2000-4000 ms) but
      abolished without one (Pred 3.D) -- dissociating anticipatory
      Πⁱ_eff retrieval from outcome-locked εⁱ encoding.
  Bars show group means +/- SEM; values are illustrative pre-data
  predictions (no dedicated seed dataset exists for Protocol 3; this
  protocol's fMRI/PPI measures are simulated from the protocol's specified
  effect sizes, consistent with the "illustrative pre-data prediction"
  framing in the spec).

This content was moved here (unchanged in substance) from the previous
generate_figure5.py to match the Figure-N <-> Protocol-(N-1) numbering
audited against OUP-Protocols.txt.

Run:
    python figures/generate_figure4.py
    python figures/generate_figure4.py --no-show   # CI mode
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from figures.utils import (  # noqa: E402
    HALF_WIDTH,
    PALETTE,
    PANEL_HEIGHT,  # noqa: E402
    label_axes,
    make_figure,
    save_figure,
)

OUTPUT_DIR = pathlib.Path(__file__).parent / "output"

# Protocol 3 parameters (protocol_3_anticipation_fmri.json)
N_SUBJECTS = 36

WORKSPACE_PURPLE = "#7B3FE4"  # APGI semantic palette (shared rendering spec)


def simulate_ppi(n_subjects: int = N_SUBJECTS, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)

    # PPI coefficients: anticipation window > outcome window (Pred 3.b)
    ppi_anticipation = rng.normal(0.42, 0.12, n_subjects)
    ppi_outcome = rng.normal(0.11, 0.10, n_subjects)

    # vmPFC BOLD: EV-parametric (valence) vs. contrast-parametric (Pred 3.c)
    bold_valence = rng.normal(0.55, 0.14, n_subjects)  # sensitive to option EV
    bold_contrast = rng.normal(0.08, 0.11, n_subjects)  # insensitive to contrast

    # Foreperiod manipulation (Pred 3.d): long vs. 0 ms foreperiod
    ppi_long_fp = rng.normal(0.44, 0.11, n_subjects)
    ppi_no_fp = rng.normal(0.09, 0.10, n_subjects)

    return {
        "ppi_anticipation": ppi_anticipation,
        "ppi_outcome": ppi_outcome,
        "bold_valence": bold_valence,
        "bold_contrast": bold_contrast,
        "ppi_long_fp": ppi_long_fp,
        "ppi_no_fp": ppi_no_fp,
    }


def plot(data: dict, show: bool = True) -> None:
    fig, axes = make_figure(ncols=3, width=HALF_WIDTH * 3, height=PANEL_HEIGHT)

    # Panel A: PPI anticipation vs outcome (Pred 3.b)
    ax = axes[0]
    means = [data["ppi_anticipation"].mean(), data["ppi_outcome"].mean()]
    sems = [
        data["ppi_anticipation"].std() / np.sqrt(N_SUBJECTS),
        data["ppi_outcome"].std() / np.sqrt(N_SUBJECTS),
    ]
    ax.bar(
        ["Anticipation", "Outcome"],
        means,
        yerr=sems,
        color=[PALETTE["S_t"], WORKSPACE_PURPLE],
        alpha=0.85,
        edgecolor="white",
        width=0.4,
        capsize=5,
    )
    ax.axhline(0, ls="--", lw=0.8, color="black", alpha=0.4)
    ax.set_ylabel("vmPFC–pIC PPI coefficient", fontsize=10)
    ax.set_title("Pred 3.b — Anticipatory coupling\npeaks before outcome", fontsize=10)

    # Panel B: vmPFC BOLD — valence vs contrast (Pred 3.c)
    ax = axes[1]
    means_b = [data["bold_valence"].mean(), data["bold_contrast"].mean()]
    sems_b = [
        data["bold_valence"].std() / np.sqrt(N_SUBJECTS),
        data["bold_contrast"].std() / np.sqrt(N_SUBJECTS),
    ]
    ax.bar(
        ["Option EV\n(valence)", "Sensory\ncontrast"],
        means_b,
        yerr=sems_b,
        color=[PALETTE["S_t"], PALETTE["theta"]],
        alpha=0.85,
        edgecolor="white",
        width=0.4,
        capsize=5,
    )
    ax.axhline(0, ls="--", lw=0.8, color="black", alpha=0.4)
    ax.set_ylabel("vmPFC BOLD β (a.u.)", fontsize=10)
    ax.set_title("Pred 3.c — vmPFC sensitive to\nvalence, not contrast", fontsize=10)

    # Panel C: Foreperiod manipulation (Pred 3.d)
    ax = axes[2]
    means_fp = [data["ppi_long_fp"].mean(), data["ppi_no_fp"].mean()]
    sems_fp = [
        data["ppi_long_fp"].std() / np.sqrt(N_SUBJECTS),
        data["ppi_no_fp"].std() / np.sqrt(N_SUBJECTS),
    ]
    ax.bar(
        ["Long foreperiod\n(2000–4000 ms)", "No foreperiod\n(0 ms)"],
        means_fp,
        yerr=sems_fp,
        color=[PALETTE["S_t"], "#AAAAAA"],
        alpha=0.85,
        edgecolor="white",
        width=0.4,
        capsize=5,
    )
    ax.axhline(0, ls="--", lw=0.8, color="black", alpha=0.4)
    ax.set_ylabel("vmPFC–pIC PPI coefficient", fontsize=10)
    ax.set_title("Pred 3.d — Anticipation drives\nvmPFC–insula coupling", fontsize=10)

    label_axes(axes)
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
    plot(simulate_ppi(), show=not args.no_show)


if __name__ == "__main__":
    main()
