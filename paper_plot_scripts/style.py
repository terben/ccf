r"""Matplotlib configuration used by the paper figure scripts.

This module provides a generalized framework for scientific plots in astronomy,
supporting multiple journal layouts (e.g., A&A, OJA) via a central configuration.

Typical use
-----------

    import matplotlib.pyplot as plt
    from style import set_plot_style

    # Set style for Astronomy & Astrophysics single column
    set_plot_style(journal="aa", column="single")
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(r"Time ($\mathrm{s}$)")
    ax.set_ylabel(r"Flux ($\mathrm{Jy}$)")
    fig.savefig("figure.pdf")

Use ``revert_params()`` to restore the Matplotlib configuration that was
active when this module was imported. For temporary settings, use
``style_context()`` as a context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import matplotlib

MM_PER_INCH = 25.4
GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0

@dataclass(frozen=True)
class JournalConfig:
    """Encapsulates journal-specific layout dimensions."""
    name: str
    text_width_in: float
    column_sep_in: float
    default_fontsize: float = 8.0

# Central Registry for journal configurations
JOURNAL_REGISTRY: dict[str, JournalConfig] = {
    "aa": JournalConfig(
        name="Astronomy & Astrophysics",
        text_width_in=184.0 / MM_PER_INCH,
        column_sep_in=4.0 / MM_PER_INCH,
        default_fontsize=8.0,
    ),
    "oja": JournalConfig(
        name="Open Journal of Astrophysics",
        text_width_in=7.1014,
        column_sep_in=0.3125,
        default_fontsize=8.0,
    ),
}

# Store the configuration that was active when this module was imported.
ORIGINAL_MATPLOTLIB_CONFIG = matplotlib.rcParams.copy()


def _calculate_figure_width(config: JournalConfig, column: Literal["single", "double"]) -> float:
    """Calculate the figure width in inches based on journal config and column choice."""
    if column == "double":
        return config.text_width_in
    if column == "single":
        return (config.text_width_in - config.column_sep_in) / 2.0
    raise ValueError("column must be either 'single' or 'double'")


def set_plot_style(
    journal: str = "aa",
    column: Literal["single", "double"] = "single",
    *,
    fig_width: float | None = None,
    fig_height: float | None = None,
    height_ratio: float = GOLDEN_RATIO,
    fontsize: float | None = None,
    use_tex: bool = True,
    constrained_layout: bool = True,
    dpi: int = 600,
) -> None:
    """Configure Matplotlib for homogeneous publication figures.

    Call this function before creating a figure. The settings are applied
    globally through :data:`matplotlib.rcParams`.

    Parameters
    ----------
    journal
        Key of the journal in the registry (e.g., "aa", "oja"). Defaults to "aa".
    column
        ``"single"`` for a single-column figure or ``"double"`` for a full-width figure.
    fig_width
        Optional custom figure width in inches. This overrides ``column``.
    fig_height
        Optional custom figure height in inches. If omitted, it is calculated
        from ``fig_width * height_ratio``.
    height_ratio
        Figure height divided by figure width. The default is the golden ratio.
    fontsize
        Base font size in points. If None, the journal's default is used.
    use_tex
        Use an external LaTeX installation for all text. Set this to ``False``
        for faster interactive work or when LaTeX is unavailable.
    constrained_layout
        Enable Matplotlib's constrained-layout engine.
    dpi
        Resolution used for raster output. PDF and SVG output remains vector
        based for lines and text.
    """
    if journal not in JOURNAL_REGISTRY:
        raise KeyError(f"Journal '{journal}' not found in registry. Available: {list(JOURNAL_REGISTRY.keys())}")

    config = JOURNAL_REGISTRY[journal]

    if fig_width is None:
        fig_width = _calculate_figure_width(config, column)

    if fig_width <= 0:
        raise ValueError("fig_width must be positive")
    if fig_height is not None and fig_height <= 0:
        raise ValueError("fig_height must be positive")
    if height_ratio <= 0:
        raise ValueError("height_ratio must be positive")
    if fontsize is not None and fontsize <= 0:
        raise ValueError("fontsize must be positive")
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    if fontsize is None:
        fontsize = config.default_fontsize

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
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
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
def style_context(
    journal: str = "aa",
    column: Literal["single", "double"] = "single",
    **kwargs: object
) -> Iterator[None]:
    """Temporarily apply a journal's plotting style.

    All keyword arguments are passed to :func:`set_plot_style`.

    Example
    -------

        with style_context(journal="oja", column="double", fontsize=8, use_tex=False):
            fig, ax = plt.subplots()
            ax.plot(x, y)
    """
    previous = matplotlib.rcParams.copy()
    set_plot_style(journal=journal, column=column, **kwargs)
    try:
        yield
    finally:
        matplotlib.rcParams.update(previous)


def aa_plot(**kwargs: object) -> None:
    """Wrapper for A&A plots.

    Passes all arguments to set_plot_style with journal="aa".
    """
    set_plot_style(journal="aa", **kwargs)

def oja_plot(**kwargs: object) -> None:
    """Wrapper for OJA plots.

    Passes all arguments to set_plot_style with journal="oja".
    """
    set_plot_style(journal="oja", **kwargs)


__all__ = [
    "JournalConfig",
    "JOURNAL_REGISTRY",
    "set_plot_style",
    "style_context",
    "aa_plot",
    "oja_plot",
    "revert_params",
]
