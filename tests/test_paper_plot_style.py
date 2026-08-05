"""Tests for paper_plot_scripts/style.py (A&A Matplotlib configuration)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from paper_plot_scripts import style


def test_aa_plot_single_column_width():
    style.aa_plot(column="single", use_tex=False)
    try:
        fig, ax = plt.subplots()
        width, height = fig.get_size_inches()

        np.testing.assert_allclose(width, style.AA_COLUMN_WIDTH)
        np.testing.assert_allclose(
            style.AA_COLUMN_WIDTH_MM, 90.0, rtol=1e-12
        )
        np.testing.assert_allclose(
            height, width * style.GOLDEN_RATIO, rtol=1e-12
        )
        plt.close(fig)
    finally:
        style.revert_params()


def test_aa_plot_double_column_width():
    style.aa_plot(column="double", use_tex=False)
    try:
        fig, ax = plt.subplots()
        width, height = fig.get_size_inches()

        np.testing.assert_allclose(width, style.AA_TEXT_WIDTH)
        np.testing.assert_allclose(
            style.AA_TEXT_WIDTH_MM, 184.0, rtol=1e-12
        )
        plt.close(fig)
    finally:
        style.revert_params()


def test_aa_plot_runs_and_saves_figure(tmp_path):
    style.aa_plot(column="single", use_tex=False)
    try:
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        out = tmp_path / "figure.pdf"
        fig.savefig(out)
        plt.close(fig)

        assert out.exists()
        assert out.stat().st_size > 0
    finally:
        style.revert_params()


def test_aa_plot_invalid_column_raises():
    with pytest.raises(ValueError):
        style.aa_plot(column="triple", use_tex=False)


def test_aa_plot_custom_dimensions():
    style.aa_plot(fig_width=5.0, fig_height=3.0, use_tex=False)
    try:
        fig, ax = plt.subplots()
        width, height = fig.get_size_inches()

        np.testing.assert_allclose(width, 5.0)
        np.testing.assert_allclose(height, 3.0)
        plt.close(fig)
    finally:
        style.revert_params()


def test_aa_style_context_manager_reverts_afterwards():
    original_figsize = matplotlib.rcParams["figure.figsize"]

    with style.aa_style(column="double", use_tex=False):
        assert matplotlib.rcParams["figure.figsize"][0] == pytest.approx(
            style.AA_TEXT_WIDTH
        )

    assert matplotlib.rcParams["figure.figsize"] == original_figsize


def test_revert_params_restores_original_config():
    original = dict(style.ORIGINAL_MATPLOTLIB_CONFIG)

    style.aa_plot(column="single", use_tex=False)
    style.revert_params()

    assert matplotlib.rcParams["figure.figsize"] == original["figure.figsize"]
    assert matplotlib.rcParams["font.size"] == original["font.size"]
