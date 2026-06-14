"""Shared plotting utilities for APGI figures."""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.modules["utils"] = sys.modules[__name__]

mpl.rcParams.update(
    {
        "mathtext.fontset": "dejavusans",
        "font.family": "DejaVu Sans",
    }
)

PALETTE = {
    "S_t": "#2166ac",
    "theta": "#d6604d",
    "ignition": "#f4a582",
    "beta": "#2166ac",
    "pi_i": "#d6604d",
    "identity": "#333333",
}

FULL_WIDTH = 10.0
HALF_WIDTH = 4.8
PANEL_HEIGHT = 4.0


def despine(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label_axes(
    axes: list[Axes], labels: list[str] | None = None, fontsize: int = 12
) -> None:
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
    ax.annotate(f"r = {r:.3f}", xy=xy, xycoords="axes fraction", fontsize=10)


def add_identity_line(ax: Axes, lo: float, hi: float) -> None:
    # Prominent y = x reference so systematic recovery bias is visually
    # assessable against the scatter (drawn on top of the points).
    ax.plot(
        [lo, hi], [lo, hi],
        "--", color="black", lw=1.2, alpha=0.9, zorder=6, label="y = x (identity)",
    )


def vlines_ignition(ax: Axes, t: np.ndarray, ignition: np.ndarray) -> None:
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


def make_figure(
    ncols: int = 1,
    nrows: int = 1,
    width: float = FULL_WIDTH,
    height: float = PANEL_HEIGHT,
) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    flat: list[Axes] = np.asarray(axes).ravel().tolist()
    for ax in flat:
        despine(ax)
    return fig, flat


def save_figure(fig: Figure, path: pathlib.Path | str, dpi: int = 300) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
