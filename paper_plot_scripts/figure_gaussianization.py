#!/usr/bin/env python3
"""Reproduce the quasi-Gaussian closure test from Sect. 5.3.

Simulates direct correlation samples from a periodic 1-D Gaussian field,
transports them through a multivariate Gaussian fit in Fisher (y) space,
and compares direct vs. transported r-marginals at selected lags.

Run:
    python paper_plot_scripts/figure_gaussianization.py --quick
    python paper_plot_scripts/figure_gaussianization.py

--quick uses a small sample count for a fast functional check; the default
matches the publication sample size (M=400000) and is significantly
slower.

Writes a PDF and, if --png is passed, a PNG preview.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ks_2samp, kurtosis, skew

import ccf
from style import AA_TEXT_WIDTH, aa_plot

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


QUICK_SAMPLES = 2_000
QUICK_BATCH_SIZE = 500


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true",
                   help="small M for a fast functional check "
                        f"(samples={QUICK_SAMPLES}, batch-size={QUICK_BATCH_SIZE}), "
                        "unless --samples/--batch-size override it")
    p.add_argument("--samples", "-M", type=int, default=None,
                   help="number of realizations (default: 400000, or "
                        f"{QUICK_SAMPLES} with --quick)")
    p.add_argument("--grid-points", "-N", type=int, default=32,
                   help="number of periodic real-space grid points (default: 32)")
    p.add_argument("--Lk0", type=float, default=80.0,
                   help="dimensionless power-spectrum width L*k0 (default: 80)")
    p.add_argument("--lags", type=int, nargs=3, default=(1, 2, 3),
                   metavar=("N1", "N2", "N3"),
                   help="three displayed lags (default: 1 2 3)")
    p.add_argument("--seed", type=int, default=20250304,
                   help="random seed (default: 20250304)")
    p.add_argument("--batch-size", type=int, default=None,
                   help="simulation batch size (default: 50000, or "
                        f"{QUICK_BATCH_SIZE} with --quick)")
    p.add_argument("--bins", type=int, default=180,
                   help="number of common density bins per panel (default: 180)")
    p.add_argument("--smooth", type=float, default=1.35,
                   help="Gaussian smoothing width in histogram bins (default: 1.35)")
    p.add_argument("--diagonal-cov", action="store_true",
                   help="use only diagonal y covariance (diagnostic variant)")
    p.add_argument("--output", "-o", type=Path,
                   default=FIGDIR / "fig_3_gaussianisation.pdf",
                   help="output PDF path (default: ../figs/fig_3_gaussianisation.pdf)")
    p.add_argument("--png", action="store_true",
                   help="also write a PNG preview")
    p.add_argument(
        "--stats-table", type=Path, default=None, metavar="FILE",
        help=("write the numerical inset information to a standalone LaTeX "
              "table source and omit the insets from the figure")
    )
    args = p.parse_args()

    if args.samples is None:
        args.samples = QUICK_SAMPLES if args.quick else 400_000
    if args.batch_size is None:
        args.batch_size = QUICK_BATCH_SIZE if args.quick else 50_000

    return args


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

def residual_sampling_band(
    p_dir: np.ndarray,
    p_trs: np.ndarray,
    edges: np.ndarray,
    n_samples: int,
    smooth: float,
) -> np.ndarray:
    """Pointwise ±1σ sampling band of the plotted density residual.

    The residual plotted in the lower row is
        (tilde p_trs - tilde p_dir) / max(tilde p_dir),
    where the tilde denotes Gaussian-smoothed histogram density
    estimates from M i.i.d. samples in bins of width Delta.

    For a smoothing kernel of width sigma_s bins, the variance of a
    single smoothed estimate at density value p(r) is approximately
        Var(tilde p(r)) ~= p(r) / (M * Delta) * 1/(2 sqrt(pi) sigma_s),
    (binomial bin variance reduced by the sum of squared Gaussian
    kernel weights).  The direct and transport histograms are drawn
    from independent RNGs, so their variances add, giving
        sigma_res(r) = sqrt((p_dir(r) + p_trs(r)) /
                             (2 sqrt(pi) sigma_s M Delta)).
    The returned band is sigma_res(r) / max(p_dir), matching the
    normalisation of the plotted residual.
    """
    delta = float(np.mean(np.diff(edges)))
    reduction = 1.0 / (2.0 * np.sqrt(np.pi) * smooth) if smooth > 0 else 1.0
    var_sum = (np.clip(p_dir, 0.0, None) + np.clip(p_trs, 0.0, None)) \
              * reduction / (n_samples * delta)
    sigma_res = np.sqrt(var_sum)
    peak = float(np.max(p_dir))
    return sigma_res / peak if peak > 0 else sigma_res


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
                0.98, 0.97, r_text, transform=ax.transAxes,
                ha="right", va="top",
                linespacing=1.18,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", alpha=0.86,
                          edgecolor="0.75", linewidth=0.55),
            )

        axr = axes[1, col]
        band = residual_sampling_band(
            dd, dt, edges, r_direct.shape[0], smooth,
        )
        axr.fill_between(
            centers, -band, band,
            color="0.80", alpha=0.55, linewidth=0,
            label=(r"$\pm 1\sigma$ sampling" if col == 0 else None),
        )
        axr.axhline(0.0, lw=REFERENCE_LINEWIDTH, color="0.35")
        axr.plot(centers, residual, lw=SECONDARY_LINEWIDTH)
        axr.tick_params(direction="in", top=True, right=True)
        axr.set_xlabel(rf"$r_{lag}$")
        lim = max(0.012,
                  1.08 * max(np.max(np.abs(residual)), float(np.max(band))))
        axr.set_ylim(-lim, lim)
        if col == 0:
            axr.set_ylabel(r"$(p_{\rm trs}-p_{\rm dir})/\max p_{\rm dir}$")
            axr.legend(loc="upper left", frameon=False, fontsize=7)

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
    rng_direct, rng_transport = [np.random.default_rng(s)
                                 for s in seed_sequence.spawn(2)]

    print(f"Simulating {args.samples:,} direct realizations ...")
    r_direct = simulate_direct_r(
        args.samples, args.grid_points, args.Lk0, rng_direct, args.batch_size
    )

    print("Converting r -> alpha via ccf.pacf() ...")
    t0 = time.time()
    try:
        alpha_direct = ccf.pacf(r_direct)
    except (ccf.SingularToeplitzError, ValueError) as error:
        raise RuntimeError(
            "ccf.pacf() could not recover an admissible alpha for "
            "the full batch -- at least one simulated row left the PACF "
            f"cube. Original error: {error}"
        ) from error
    print(f"  ccf.pacf() on {r_direct.shape}: {time.time() - t0:.2f}s")

    max_abs_alpha = float(np.max(np.abs(alpha_direct)))
    print(f"  max |alpha_direct| = {max_abs_alpha:.16g} "
          "(ccf.pacf() guarantees this is < 1 by construction, "
          "or it would already have raised above)")
    eps = 8.0 * np.finfo(float).eps
    alpha_direct = np.clip(alpha_direct, -1.0 + eps, 1.0 - eps)
    y_direct = np.arctanh(alpha_direct)

    print("Fitting and drawing the multivariate Gaussian in y-space ...")
    y_transport, _, _ = gaussian_transport(y_direct, rng_transport,
                                           args.diagonal_cov)
    alpha_transport = np.tanh(y_transport)

    print("Converting alpha -> r via ccf.from_pacf() ...")
    t0 = time.time()
    r_transport = ccf.from_pacf(alpha_transport)
    print(f"  ccf.from_pacf() on {alpha_transport.shape}: "
          f"{time.time() - t0:.2f}s")

    # Internal roundtrip check catches sign/indexing mistakes in the recursion.
    check = ccf.from_pacf(alpha_direct[: min(2000, args.samples)])
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

    if args.png:
        png = args.output.with_suffix(".png")
        fig.savefig(png, dpi=300)
        print(f"Wrote {png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
