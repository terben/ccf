#!/usr/bin/env python3
"""Figure 3 (\\label{fig:qg_transport} in CCF.tex), Sect. 5.3.

The script performs the structural closure test described there:

1. Simulate M realizations of the normalized correlation coefficients r_n
   of a periodic one-dimensional Gaussian field with N=32 grid points and
   a Gaussian power spectrum, L k0 = 80 (Wilking & Schneider 2013).
2. Convert r -> alpha with the Levinson-Durbin recursion and then
   y_n = atanh(alpha_n).
3. Fit a multivariate Gaussian to the complete y-vector (mean and full
   covariance), draw a transported sample, and map y -> alpha -> r.
4. Compare the direct and transported r-marginals for selected lags.

By default the program uses M=400000, matching WS2013.  It writes a PDF
and, unless --no-png is passed, a PNG preview.  The only dependencies are
NumPy, SciPy, and Matplotlib.

r_to_alpha / alpha_to_r below are self-contained, batched
Levinson-Durbin recursions -- the same recursion as
schurcorr.pacf / schurcorr.from_pacf, but vectorized directly across
the sample axis via NumPy reductions (`np.sum(..., axis=1)`) rather
than schurcorr's row-by-row loop (chosen there to keep the 1-D and
batched code paths on identical floating-point operations, see
schurcorr/levinson.py). The two vectorization strategies are
mathematically equivalent but not bit-for-bit identical, and at
M=400000 samples the difference is not always below floating-point
noise; these two functions are therefore kept local rather than
replaced by the library calls, to reproduce the published figure and
table exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt

from schurcorr.plotting import AA_TEXT_WIDTH, aa_plot
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis, ks_2samp, skew

# Colour palette matching Fig. 1
TRANSPORT_COLOR = "#c0392b"
DIRECT_COLOR = "black"
RESIDUAL_COLOR = "#6a3d9a"

# Configure Matplotlib for a two-column Astronomy & Astrophysics figure.
# The full figure width is taken from aa_plot; only the height ratio is set here.
aa_plot(
    column="double",
    height_ratio=0.52,
    fontsize=8.0,
    use_tex=True,
    constrained_layout=False,
)

# Preserve the exact A&A text width when saving.  The layout is controlled
# explicitly below because this 2 x 3 panel arrangement benefits from tighter
# spacing than Matplotlib's automatic layout engines provide.
plt.rcParams["savefig.bbox"] = None

# Line-width hierarchy matched to Fig. 1:
#   2.2 pt: principal comparison curves (Fig. 1 boundaries)
#   1.3 pt: secondary/diagnostic curves (Fig. 1 guide curves)
#   0.8 pt: neutral reference lines and axes (aa_plot default)
MAIN_LINEWIDTH = 2.2
SECONDARY_LINEWIDTH = 1.3
REFERENCE_LINEWIDTH = 0.8

FIGDIR = Path(__file__).resolve().parent / "../figs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", "-M", type=int, default=400_000,
                   help="number of realizations (default: 400000)")
    p.add_argument("--grid-points", "-N", type=int, default=32,
                   help="number of periodic real-space grid points (default: 32)")
    p.add_argument("--Lk0", type=float, default=80.0,
                   help="dimensionless power-spectrum width L*k0 (default: 80)")
    p.add_argument("--lags", type=int, nargs=3, default=(1, 2, 3),
                   metavar=("N1", "N2", "N3"),
                   help="three displayed lags (default: 1 2 3)")
    p.add_argument("--seed", type=int, default=20250304,
                   help="random seed (default: 20250304)")
    p.add_argument("--batch-size", type=int, default=50_000,
                   help="simulation batch size (default: 50000)")
    p.add_argument("--bins", type=int, default=180,
                   help="number of common density bins per panel (default: 180)")
    p.add_argument("--smooth", type=float, default=1.35,
                   help="Gaussian smoothing width in histogram bins (default: 1.35)")
    p.add_argument("--diagonal-cov", action="store_true",
                   help="use only diagonal y covariance (diagnostic variant)")
    p.add_argument("--output", "-o", type=Path,
                   default=FIGDIR / "fig_4_qg_transport.pdf",
                   help="output PDF path (default: ../figs/fig_4_qg_transport.pdf)")
    p.add_argument("--no-png", action="store_true",
                   help="do not also write a PNG preview")
    p.add_argument(
        "--stats-table", type=Path, default=None, metavar="FILE",
        help=("write the numerical inset information to a standalone LaTeX "
              "table source and omit the insets from the figure")
    )
    return p.parse_args()


def simulate_direct_r(
    samples: int,
    n_grid: int,
    lk0: float,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    """Draw direct r samples using the WS2013 power-spectrum construction.

    For a Gaussian random field each independent positive-frequency power
    mode is exponentially distributed.  The periodic correlation estimator
    of SH2009 Eq. (63) is exactly the inverse discrete Fourier transform of
    these non-negative realized powers.  As in WS2013, the zero mode and the
    Nyquist mode are omitted, leaving N/2-1 modes.
    """
    if n_grid < 4 or n_grid % 2:
        raise ValueError("N must be an even integer >= 4")
    n_lags = n_grid // 2 - 1
    mode = np.arange(1, n_grid // 2, dtype=float)  # 1,...,N/2-1
    mean_power = np.exp(-((2.0 * np.pi * mode / lk0) ** 2))
    lag = np.arange(0, n_lags + 1, dtype=float)
    cosine = np.cos(2.0 * np.pi * np.outer(mode, lag) / n_grid)

    result = np.empty((samples, n_lags), dtype=np.float64)
    start = 0
    while start < samples:
        stop = min(start + batch_size, samples)
        # Exponential powers are equivalent to squared complex-Gaussian modes.
        powers = rng.exponential(scale=mean_power, size=(stop - start, n_lags))
        xi = powers @ cosine
        result[start:stop] = xi[:, 1:] / xi[:, [0]]
        start = stop
    return result


def r_to_alpha(r: np.ndarray) -> np.ndarray:
    """Inverse Levinson-Durbin map from autocorrelations to PACF alpha."""
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("r must have shape (samples, lags)")
    m, nmax = r.shape
    alpha = np.empty_like(r)
    phi = np.empty((m, 0), dtype=np.float64)
    sigma2 = np.ones(m, dtype=np.float64)

    for n in range(1, nmax + 1):
        if n == 1:
            pred = np.zeros(m)
        else:
            pred = np.sum(phi * r[:, n - 2 :: -1], axis=1)
        a = (r[:, n - 1] - pred) / sigma2
        alpha[:, n - 1] = a
        if n == 1:
            phi = a[:, None]
        else:
            phi = np.concatenate((phi - a[:, None] * phi[:, ::-1], a[:, None]), axis=1)
        sigma2 *= 1.0 - a * a
    return alpha


def alpha_to_r(alpha: np.ndarray) -> np.ndarray:
    """Forward Levinson-Durbin map from PACF alpha to autocorrelations."""
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.ndim != 2:
        raise ValueError("alpha must have shape (samples, lags)")
    m, nmax = alpha.shape
    r = np.empty_like(alpha)
    phi = np.empty((m, 0), dtype=np.float64)
    sigma2 = np.ones(m, dtype=np.float64)

    for n in range(1, nmax + 1):
        a = alpha[:, n - 1]
        if n == 1:
            pred = np.zeros(m)
        else:
            pred = np.sum(phi * r[:, n - 2 :: -1], axis=1)
        r[:, n - 1] = pred + a * sigma2
        if n == 1:
            phi = a[:, None]
        else:
            phi = np.concatenate((phi - a[:, None] * phi[:, ::-1], a[:, None]), axis=1)
        sigma2 *= 1.0 - a * a
    return r


def gaussian_transport(
    y_direct: np.ndarray,
    rng: np.random.Generator,
    diagonal_cov: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit N(mu,C) in y-space and draw one equally sized joint sample."""
    mu = np.mean(y_direct, axis=0)
    cov = np.cov(y_direct, rowvar=False, ddof=1)
    if diagonal_cov:
        cov = np.diag(np.diag(cov))

    # Tiny adaptive jitter protects Cholesky against round-off without changing
    # the empirical covariance at visible precision.
    scale = float(np.max(np.diag(cov)))
    jitter = max(scale, 1.0) * 1.0e-13
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))

    z = rng.standard_normal(y_direct.shape)
    y_transport = mu + z @ chol.T
    return y_transport, mu, cov


def smooth_hist_density(x: np.ndarray, edges: np.ndarray, smooth: float) -> np.ndarray:
    h, _ = np.histogram(x, bins=edges, density=True)
    if smooth > 0:
        h = gaussian_filter1d(h.astype(float), smooth, mode="nearest")
    # Re-normalize after edge smoothing.
    widths = np.diff(edges)
    norm = np.sum(h * widths)
    return h / norm if norm > 0 else h


def moments(x: np.ndarray) -> Tuple[float, float]:
    return float(skew(x, bias=False)), float(kurtosis(x, fisher=True, bias=False))


def robust_edges(a: np.ndarray, b: np.ndarray, bins: int) -> np.ndarray:
    both = np.concatenate((a, b))
    lo, hi = np.quantile(both, [2.5e-4, 1.0 - 2.5e-4])
    pad = 0.045 * (hi - lo)
    lo = max(-1.0, lo - pad)
    hi = min(1.0, hi + pad)
    return np.linspace(lo, hi, bins + 1)


def compute_statistics(
    r_direct: np.ndarray,
    r_transport: np.ndarray,
    y_direct: np.ndarray,
    y_transport: np.ndarray,
    lags: Iterable[int],
) -> list[dict[str, float]]:
    """Return the diagnostics formerly shown in the numerical insets."""
    rows: list[dict[str, float]] = []
    for lag in lags:
        idx = lag - 1
        g1d, g2d = moments(r_direct[:, idx])
        g1t, g2t = moments(r_transport[:, idx])
        yg1d, yg2d = moments(y_direct[:, idx])
        yg1t, yg2t = moments(y_transport[:, idx])
        rows.append({
            "lag": float(lag),
            "dks": float(ks_2samp(
                r_direct[:, idx], r_transport[:, idx], method="asymp"
            ).statistic),
            "r_g1_dir": g1d, "r_g1_trs": g1t,
            "r_g2_dir": g2d, "r_g2_trs": g2t,
            "y_g1_dir": yg1d, "y_g1_trs": yg1t,
            "y_g2_dir": yg2d, "y_g2_trs": yg2t,
        })
    return rows


def write_latex_statistics_table(
    path: Path, statistics: list[dict[str, float]],
) -> None:
    """Write a complete LaTeX table containing the closure diagnostics."""
    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \caption{Numerical diagnostics for the quasi-Gaussian closure test. "
        r"The symbols $\gamma_1$ and $\gamma_2$ denote skewness and excess kurtosis, respectively.}",
        r"  \label{tab:fig_5_3}",
        r"  \begin{tabular}{c c cc cc cc cc}",
        r"    \hline",
        r"    $n$ & $D_{\rm KS}$ & "
        r"$\gamma_{1,r}^{\rm dir}$ & $\gamma_{1,r}^{\rm trs}$ & "
        r"$\gamma_{2,r}^{\rm dir}$ & $\gamma_{2,r}^{\rm trs}$ & "
        r"$\gamma_{1,y}^{\rm dir}$ & $\gamma_{1,y}^{\rm trs}$ & "
        r"$\gamma_{2,y}^{\rm dir}$ & $\gamma_{2,y}^{\rm trs}$ \\",
        r"    \hline",
    ]
    for row in statistics:
        lines.append(
            "    "
            f"{int(row['lag'])} & {row['dks']:.4f} & "
            f"{row['r_g1_dir']:+.3f} & {row['r_g1_trs']:+.3f} & "
            f"{row['r_g2_dir']:+.3f} & {row['r_g2_trs']:+.3f} & "
            f"{row['y_g1_dir']:+.3f} & {row['y_g1_trs']:+.3f} & "
            f"{row['y_g2_dir']:+.3f} & {row['y_g2_trs']:+.3f} \\\\"
        )
    lines.extend([
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(
    r_direct: np.ndarray,
    r_transport: np.ndarray,
    y_direct: np.ndarray,
    y_transport: np.ndarray,
    lags: Iterable[int],
    bins: int,
    smooth: float,
    diagonal_cov: bool,
    show_insets: bool = True,
    statistics: list[dict[str, float]] | None = None,
) -> plt.Figure:
    lags = tuple(lags)
    if statistics is None:
        statistics = compute_statistics(
            r_direct, r_transport, y_direct, y_transport, lags
        )
    if len(statistics) != len(lags):
        raise ValueError("statistics and lags must have the same length")

    # The version with numerical insets needs a taller density row.  Set the
    # height explicitly while keeping the full A&A two-column width.
    height_ratio = 0.55 if show_insets else 0.47
    height_ratios = [2.2, 1.0] if show_insets else [1.62, 1.0]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(AA_TEXT_WIDTH, AA_TEXT_WIDTH * height_ratio),
        sharex="col",
        gridspec_kw={
            "height_ratios": height_ratios,
            "hspace": 0.10,
            "wspace": 0.16,
        },
    )

    for col, lag in enumerate(lags):
        idx = lag - 1
        rd = r_direct[:, idx]
        rt = r_transport[:, idx]
        edges = robust_edges(rd, rt, bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        dd = smooth_hist_density(rd, edges, smooth)
        dt = smooth_hist_density(rt, edges, smooth)
        residual = (dt - dd) / np.max(dd)

        row = statistics[col]
        g1d, g1t = row["r_g1_dir"], row["r_g1_trs"]
        g2d, g2t = row["r_g2_dir"], row["r_g2_trs"]
        yg1d, yg1t = row["y_g1_dir"], row["y_g1_trs"]
        yg2d, yg2t = row["y_g2_dir"], row["y_g2_trs"]
        dks = row["dks"]

        ax = axes[0, col]
        ax.plot(centers, dt, lw=MAIN_LINEWIDTH, label="transport")
        ax.plot(centers, dd, lw=MAIN_LINEWIDTH, ls="--", label="direct")
        ax.set_title(rf"lag $n={lag}$")
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", top=True, right=True)
        if col == 0:
            ax.set_ylabel(r"density $p(r_n)$")
            ax.legend(loc="upper left", frameon=False)

        r_text = (
            rf"$D_{{\rm KS}}={dks:.4f}$" "\n"
            rf"$r$: $\gamma_1^{{\rm dir}}={g1d:+.3f}$, $\gamma_1^{{\rm trs}}={g1t:+.3f}$" "\n"
            rf"$\quad\gamma_2^{{\rm dir}}={g2d:+.3f}$, $\gamma_2^{{\rm trs}}={g2t:+.3f}$" "\n"
            rf"$y$: $\gamma_1^{{\rm dir}}={yg1d:+.3f}$, $\gamma_1^{{\rm trs}}={yg1t:+.3f}$" "\n"
            rf"$\quad\gamma_2^{{\rm dir}}={yg2d:+.3f}$, $\gamma_2^{{\rm trs}}={yg2t:+.3f}$"
        )
        if show_insets:
            ax.text(
                0.98, 0.97, r_text, transform=ax.transAxes, ha="right", va="top",
                linespacing=1.18,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.86,
                          edgecolor="0.75", linewidth=0.55),
            )

        axr = axes[1, col]
        axr.axhline(0.0, lw=REFERENCE_LINEWIDTH, color="0.35")
        axr.plot(centers, residual, lw=SECONDARY_LINEWIDTH)
        axr.tick_params(direction="in", top=True, right=True)
        axr.set_xlabel(rf"$r_{lag}$")
        lim = max(0.012, 1.08 * np.max(np.abs(residual)))
        axr.set_ylim(-lim, lim)
        if col == 0:
            axr.set_ylabel(r"$(p_{\rm trs}-p_{\rm dir})/\max p_{\rm dir}$")

    # Manual margins give a compact and stable 3-column layout.  A global
    # title is intentionally omitted; the caption carries that information.
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        bottom=0.115,
        top=0.955,
        wspace=0.16,
        hspace=0.10,
    )
    return fig


def main() -> None:
    args = parse_args()
    if args.samples < 10:
        raise ValueError("--samples must be at least 10")
    max_lag = args.grid_points // 2 - 1
    if any(lag < 1 or lag > max_lag for lag in args.lags):
        raise ValueError(f"lags must lie between 1 and {max_lag}")

    seed_sequence = np.random.SeedSequence(args.seed)
    rng_direct, rng_transport = [np.random.default_rng(s) for s in seed_sequence.spawn(2)]

    print(f"Simulating {args.samples:,} direct realizations ...")
    r_direct = simulate_direct_r(
        args.samples, args.grid_points, args.Lk0, rng_direct, args.batch_size
    )

    alpha_direct = r_to_alpha(r_direct)
    max_abs_alpha = float(np.max(np.abs(alpha_direct)))
    if max_abs_alpha > 1.0 + 5e-10:
        raise RuntimeError(f"direct sample left PACF cube: max |alpha|={max_abs_alpha:.16g}")
    eps = 8.0 * np.finfo(float).eps
    alpha_direct = np.clip(alpha_direct, -1.0 + eps, 1.0 - eps)
    y_direct = np.arctanh(alpha_direct)

    print("Fitting and drawing the multivariate Gaussian in y-space ...")
    y_transport, _, _ = gaussian_transport(y_direct, rng_transport, args.diagonal_cov)
    alpha_transport = np.tanh(y_transport)
    r_transport = alpha_to_r(alpha_transport)

    # Internal roundtrip check catches sign/indexing mistakes in the recursion.
    check = alpha_to_r(alpha_direct[: min(2000, args.samples)])
    roundtrip = float(np.max(np.abs(check - r_direct[: check.shape[0]])))
    print(f"Levinson-Durbin roundtrip max error: {roundtrip:.3e}")

    statistics = compute_statistics(
        r_direct, r_transport, y_direct, y_transport, args.lags
    )
    for row in statistics:
        print(f"lag {int(row['lag']):2d}: D_KS = {row['dks']:.6f}")

    if args.stats_table is not None:
        write_latex_statistics_table(args.stats_table, statistics)
        print(f"Wrote {args.stats_table}")

    fig = make_figure(
        r_direct, r_transport, y_direct, y_transport,
        args.lags, args.bins, args.smooth, args.diagonal_cov,
        show_insets=args.stats_table is None, statistics=statistics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"Wrote {args.output}")
    if not args.no_png:
        png = args.output.with_suffix(".png")
        fig.savefig(png, dpi=300)
        print(f"Wrote {png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
