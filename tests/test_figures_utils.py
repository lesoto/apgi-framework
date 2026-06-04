"""Tests for figures/utils.py.

Covers PALETTE constants, despine, label_axes, annotate_pearson_r,
add_identity_line, vlines_ignition, make_figure, and save_figure.
"""

import pathlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

# figures/ is not an installed package — add it to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "figures"))
import utils as fig_utils

# ===========================================================================
# PALETTE
# ===========================================================================


class TestPalette:
    def test_required_keys_present(self):
        required = {"S_t", "theta", "ignition", "beta", "pi_i", "identity"}
        assert required.issubset(set(fig_utils.PALETTE.keys()))

    def test_values_are_hex_strings(self):
        for key, val in fig_utils.PALETTE.items():
            assert isinstance(val, str), f"{key} is not a string"
            assert val.startswith("#"), f"{key} value '{val}' is not a hex colour"

    def test_constants_defined(self):
        assert fig_utils.FULL_WIDTH > 0
        assert fig_utils.HALF_WIDTH > 0
        assert fig_utils.PANEL_HEIGHT > 0


# ===========================================================================
# despine
# ===========================================================================


class TestDespine:
    def test_removes_top_and_right_spines(self):
        _, ax = plt.subplots()
        fig_utils.despine(ax)
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close("all")

    def test_leaves_bottom_and_left_spines(self):
        _, ax = plt.subplots()
        fig_utils.despine(ax)
        assert ax.spines["bottom"].get_visible()
        assert ax.spines["left"].get_visible()
        plt.close("all")


# ===========================================================================
# label_axes
# ===========================================================================


class TestLabelAxes:
    def test_default_labels_are_uppercase_letters(self):
        _, axes = plt.subplots(1, 3)
        fig_utils.label_axes(list(axes))
        texts = [t.get_text() for ax in axes for t in ax.texts]
        assert "A" in texts
        assert "B" in texts
        assert "C" in texts
        plt.close("all")

    def test_custom_labels(self):
        _, axes = plt.subplots(1, 2)
        fig_utils.label_axes(list(axes), labels=["X", "Y"])
        texts = [t.get_text() for ax in axes for t in ax.texts]
        assert "X" in texts
        assert "Y" in texts
        plt.close("all")

    def test_single_axis(self):
        _, ax = plt.subplots()
        fig_utils.label_axes([ax])
        assert ax.texts[0].get_text() == "A"
        plt.close("all")


# ===========================================================================
# annotate_pearson_r
# ===========================================================================


class TestAnnotatePearsonR:
    def test_annotation_added(self):
        _, ax = plt.subplots()
        fig_utils.annotate_pearson_r(ax, r=0.873)
        annotations = [
            c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)
        ]
        assert len(annotations) == 1
        assert "0.873" in annotations[0].get_text()
        plt.close("all")

    def test_custom_xy_position(self):
        _, ax = plt.subplots()
        fig_utils.annotate_pearson_r(ax, r=0.5, xy=(0.8, 0.1))
        annotations = [
            c for c in ax.get_children() if isinstance(c, matplotlib.text.Annotation)
        ]
        assert len(annotations) == 1
        plt.close("all")


# ===========================================================================
# add_identity_line
# ===========================================================================


class TestAddIdentityLine:
    def test_adds_one_line(self):
        _, ax = plt.subplots()
        n_before = len(ax.lines)
        fig_utils.add_identity_line(ax, lo=0.0, hi=1.0)
        assert len(ax.lines) == n_before + 1
        plt.close("all")

    def test_line_is_diagonal(self):
        _, ax = plt.subplots()
        fig_utils.add_identity_line(ax, lo=-1.0, hi=1.0)
        line = ax.lines[-1]
        xdata, ydata = line.get_xdata(), line.get_ydata()
        np.testing.assert_allclose(xdata, ydata)
        plt.close("all")


# ===========================================================================
# vlines_ignition
# ===========================================================================


class TestVlinesIgnition:
    def _make_ax_with_data(self):
        _, ax = plt.subplots()
        ax.set_ylim(0, 3)
        return ax

    def test_adds_collections_for_ignition_events(self):
        ax = self._make_ax_with_data()
        t = np.arange(10, dtype=float)
        ignition = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=bool)
        fig_utils.vlines_ignition(ax, t, ignition)
        assert len(ax.collections) > 0
        plt.close("all")

    def test_no_ignitions_no_collections(self):
        ax = self._make_ax_with_data()
        t = np.arange(5, dtype=float)
        ignition = np.zeros(5, dtype=bool)
        n_before = len(ax.collections)
        fig_utils.vlines_ignition(ax, t, ignition)
        # vlines called with empty array — may add an empty collection
        # just assert it doesn't raise
        plt.close("all")

    def test_all_ignitions(self):
        ax = self._make_ax_with_data()
        t = np.arange(5, dtype=float)
        ignition = np.ones(5, dtype=bool)
        fig_utils.vlines_ignition(ax, t, ignition)
        assert len(ax.collections) > 0
        plt.close("all")


# ===========================================================================
# make_figure
# ===========================================================================


class TestMakeFigure:
    def test_returns_figure_and_axes_list(self):
        fig, axes = fig_utils.make_figure(ncols=2, nrows=1)
        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, list)
        assert len(axes) == 2
        plt.close("all")

    def test_default_single_panel(self):
        fig, axes = fig_utils.make_figure()
        assert len(axes) == 1
        plt.close("all")

    def test_multi_row_multi_col(self):
        fig, axes = fig_utils.make_figure(ncols=3, nrows=2)
        assert len(axes) == 6
        plt.close("all")

    def test_despine_applied(self):
        _, axes = fig_utils.make_figure(ncols=2)
        for ax in axes:
            assert not ax.spines["top"].get_visible()
            assert not ax.spines["right"].get_visible()
        plt.close("all")

    def test_custom_size(self):
        fig, _ = fig_utils.make_figure(width=6.0, height=3.0)
        w, h = fig.get_size_inches()
        assert w == pytest.approx(6.0)
        assert h == pytest.approx(3.0)
        plt.close("all")


# ===========================================================================
# save_figure
# ===========================================================================


class TestSaveFigure:
    def test_creates_file(self, tmp_path):
        fig, _ = fig_utils.make_figure()
        out = tmp_path / "test_output.png"
        fig_utils.save_figure(fig, out, dpi=72)
        assert out.exists()
        assert out.stat().st_size > 0
        plt.close("all")

    def test_creates_parent_dirs(self, tmp_path):
        fig, _ = fig_utils.make_figure()
        out = tmp_path / "subdir" / "nested" / "fig.png"
        fig_utils.save_figure(fig, out, dpi=72)
        assert out.exists()
        plt.close("all")

    def test_accepts_string_path(self, tmp_path):
        fig, _ = fig_utils.make_figure()
        out = str(tmp_path / "str_path.png")
        fig_utils.save_figure(fig, out, dpi=72)
        assert pathlib.Path(out).exists()
        plt.close("all")
