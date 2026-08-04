r"""
Matplotlib configuration for Astronomy & Astrophysics figures.

The dimensions are derived from the A&A ``aa.cls`` layout:

* full text width: 184 mm
* column separation: 4 mm
* single-column width: 90 mm

Typical use
-----------

    import matplotlib.pyplot as plt
    from schurcorr.plotting import aa_plot

    aa_plot(column="single")
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(r"Time ($\mathrm{s}$)")
    ax.set_ylabel(r"Flux ($\mathrm{Jy}$)")
    fig.savefig("figure.pdf")

Use ``revert_params()`` to restore the Matplotlib configuration that was
active when this module was imported. For temporary settings, use
``aa_style()`` as a context manager.

This module is used by :mod:`paper_plot_scripts` to reproduce the three
figures of the paper's numerical-demonstrations section.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Literal

import matplotlib

MM_PER_INCH = 25.4
AA_TEXT_WIDTH_MM = 184.0
AA_COLUMN_SEP_MM = 4.0
AA_COLUMN_WIDTH_MM = (AA_TEXT_WIDTH_MM - AA_COLUMN_SEP_MM) / 2.0

AA_TEXT_WIDTH = AA_TEXT_WIDTH_MM / MM_PER_INCH
AA_COLUMN_WIDTH = AA_COLUMN_WIDTH_MM / MM_PER_INCH

GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0

# Store the configuration that was active when this module was imported.
ORIGINAL_MATPLOTLIB_CONFIG = matplotlib.rcParams.copy()


def _figure_width(column: Literal["single", "double"]) -> float:
    """Return the A&A figure width in inches."""
    if column == "single":
        return AA_COLUMN_WIDTH
    if column == "double":
        return AA_TEXT_WIDTH
    raise ValueError("column must be either 'single' or 'double'")


def aa_plot(
    column: Literal["single", "double"] = "single",
    *,
    fig_width: float | None = None,
    fig_height: float | None = None,
    height_ratio: float = GOLDEN_RATIO,
    fontsize: float = 8.0,
    use_tex: bool = True,
    constrained_layout: bool = True,
    dpi: int = 600,
) -> None:
    """Configure Matplotlib for homogeneous A&A publication figures.

    Call this function before creating a figure. The settings are applied
    globally through :data:`matplotlib.rcParams`.

    Parameters
    ----------
    column
        ``"single"`` for a 90 mm wide figure or ``"double"`` for a
        184 mm wide figure.
    fig_width
        Optional custom figure width in inches. This overrides ``column``.
    fig_height
        Optional custom figure height in inches. If omitted, it is calculated
        from ``fig_width * height_ratio``.
    height_ratio
        Figure height divided by figure width. The default is the golden ratio.
    fontsize
        Base font size in points. Eight points is a practical default for A&A
        figures at their final printed size.
    use_tex
        Use an external LaTeX installation for all text. Set this to ``False``
        for faster interactive work or when LaTeX is unavailable.
    constrained_layout
        Enable Matplotlib's constrained-layout engine.
    dpi
        Resolution used for raster output. PDF and SVG output remains vector
        based for lines and text.

    Notes
    -----
    Figure widths should normally be specified at their final publication size.
    Avoid resizing figures in LaTeX, because resizing also changes the apparent
    font size and line width.
    """
    if fig_width is None:
        fig_width = _figure_width(column)

    if fig_width <= 0:
        raise ValueError("fig_width must be positive")
    if fig_height is not None and fig_height <= 0:
        raise ValueError("fig_height must be positive")
    if height_ratio <= 0:
        raise ValueError("height_ratio must be positive")
    if fontsize <= 0:
        raise ValueError("fontsize must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    if fig_height is None:
        fig_height = fig_width * height_ratio

    latex_preamble = "\n".join(
        [
            r"\usepackage[T1]{fontenc}",
            r"\usepackage{amsmath}",
            r"\usepackage{siunitx}",
            r"\sisetup{detect-all}",
        ]
    )

    params: dict[str, object] = {
        # Figure geometry
        "figure.figsize": (fig_width, fig_height),
        "figure.constrained_layout.use": constrained_layout,
        "figure.dpi": 120,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Typography
        "font.family": "serif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "axes.titlepad": 4.0,
        "legend.fontsize": fontsize - 1.0,
        "xtick.labelsize": fontsize - 1.0,
        "ytick.labelsize": fontsize - 1.0,
        "mathtext.fontset": "cm",
        "axes.unicode_minus": True,
        # Lines, markers, and axes
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.8,
        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4,
        "legend.columnspacing": 1.0,
        # Vector output and font embedding when LaTeX is disabled
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # LaTeX rendering
        "text.usetex": use_tex,
    }

    if use_tex:
        params["text.latex.preamble"] = latex_preamble

    matplotlib.rcParams.update(params)


def revert_params() -> None:
    """Restore the Matplotlib configuration active at module import time."""
    matplotlib.rcParams.update(ORIGINAL_MATPLOTLIB_CONFIG)


@contextmanager
def aa_style(*args: object, **kwargs: object) -> Iterator[None]:
    """Temporarily apply the A&A plotting style.

    All positional and keyword arguments are passed to :func:`aa_plot`.

    Example
    -------

        with aa_style(column="double", fontsize=8, use_tex=False):
            fig, ax = plt.subplots()
            ax.plot(x, y)
    """
    previous = matplotlib.rcParams.copy()
    aa_plot(*args, **kwargs)  # type: ignore[arg-type]
    try:
        yield
    finally:
        matplotlib.rcParams.update(previous)


__all__ = [
    "AA_COLUMN_SEP_MM",
    "AA_COLUMN_WIDTH",
    "AA_COLUMN_WIDTH_MM",
    "AA_TEXT_WIDTH",
    "AA_TEXT_WIDTH_MM",
    "GOLDEN_RATIO",
    "aa_plot",
    "aa_style",
    "revert_params",
]
