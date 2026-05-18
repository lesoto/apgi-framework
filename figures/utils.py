"""Shared plotting utilities for APGI figures.

Defines the APGI Neural Glow colour palette, axis formatters, and
figure-size constants so all figure scripts maintain visual consistency.
"""

from __future__ import annotations

import logging
import pathlib
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Suppress font-not-found chatter from matplotlib's font cache rebuild when
# LaTeX/CM fonts are not installed.  DejaVu Sans is the correct fallback and
# the figures look identical; the messages are purely informational noise.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="findfont:.*")

mpl.rcParams.update(
    {
        "mathtext.fontset": "dejavusans",
        "font.family": "DejaVu Sans",
    }
)

# ---------------------------------------------------------------------------
# APGI Neural Glow palette
# ---------------------------------------------------------------------------

PALETTE = {
    "S_t": "#2166ac",  # deep blue — global integration signal
    "theta": "#d6604d",  # warm red  — ignition threshold
    "ignition": "#f4a582",  # light salmon — ignition event markers
    "beta": "#2166ac",  # parameter scatter: β
    "pi_i": "#d6604d",  # parameter scatter: Πⁱ
    "identity": "#333333",  # identity line in scatter plots
}

# Standard figure widths (inches)
FULL_WIDTH = 10.0
HALF_WIDTH = 4.8
PANEL_HEIGHT = 4.0


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------


def despine(ax: Axes) -> None:
    """Remove top and right spines from *ax*."""
    ax.spines[["top", "right"]].set_visible(False)


def label_axes(
    axes: list[Axes], labels: list[str] | None = None, fontsize: int = 12
) -> None:
    """Add panel labels (A, B, C, …) to each axis in the list."""
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(axes))]
    for ax, lbl in zip(axes, labels):
        ax.text(
            -0.12,
            1.05,
            lbl,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="top",
            ha="right",
        )


def annotate_pearson_r(
    ax: Axes, r: float, xy: tuple[float, float] = (0.05, 0.92)
) -> None:
    """Annotate a scatter axis with a Pearson r value."""
    ax.annotate(f"r = {r:.3f}", xy=xy, xycoords="axes fraction", fontsize=10)


def add_identity_line(ax: Axes, lo: float, hi: float) -> None:
    """Draw a dashed identity (y = x) reference line."""
    ax.plot([lo, hi], [lo, hi], "--", color=PALETTE["identity"], lw=0.8, alpha=0.5)


def vlines_ignition(ax: Axes, t: np.ndarray, ignition: np.ndarray) -> None:
    """Draw vertical shading at ignition time-points."""
    ignition_t = t[ignition]
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 3
    ax.vlines(
        ignition_t,
        ymin=0,
        ymax=ymax,
        colors=PALETTE["ignition"],
        alpha=0.4,
        linewidth=0.8,
    )


# ---------------------------------------------------------------------------
# Figure factory
# ---------------------------------------------------------------------------


def make_figure(
    ncols: int = 1,
    nrows: int = 1,
    width: float = FULL_WIDTH,
    height: float = PANEL_HEIGHT,
) -> tuple[Figure, list[Axes]]:
    """Create a figure and flatten the axes list."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    flat: list[Axes] = np.asarray(axes).ravel().tolist()
    for ax in flat:
        despine(ax)
    return fig, flat


def save_figure(fig: Figure, path: pathlib.Path | str, dpi: int = 300) -> None:
    """Save *fig* to *path* and print a confirmation message."""

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
