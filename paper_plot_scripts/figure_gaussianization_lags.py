#!/usr/bin/env python3
"""Revised Fig. 3 (Sect. 5.3): closure test at representative lags.

Compared with figure_gaussianization.py (whose code is imported unchanged,
including its OJA style settings), this figure

* shows representative lags spanning the full range (default n = 1, 8, 15:
  the first lag, the commensurate lag N/4, and the last lag), and
* adds the naive baseline of a multivariate Gaussian in r-space (mean and
  covariance of the direct sample) as a third branch, so that the closure
  of the y-space transport is judged against the standard Gaussian
  likelihood assumption rather than in isolation.

All three branches are evaluated identically: equally sized samples,
common bins, the same Gaussian histogram smoothing.

Run (conda env ``ccf``):
    python paper_plot_scripts/figure_gaussianization_lags.py --quick
    python paper_plot_scripts/figure_gaussianization_lags.py --png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Importing figure_gaussianization applies its OJA double-column style.
from figure_gaussianization import (
    FIGDIR, MAIN_LINEWIDTH, REFERENCE_LINEWIDTH, SECONDARY_LINEWIDTH,
    residual_sampling_band, robust_edges, smooth_hist_density,
)
from m2_closure_robustness import closure_run, ks, moments
from style import figure_path

plt.rcParams["savefig.bbox"] = None

C_TRS, C_DIR, C_GR = "C0", "C1", "C2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="M = 20000 functional check")
    p.add_argument("--samples", "-M", type=int, default=None)
    p.add_argument("--grid-points", "-N", type=int, default=32)
    p.add_argument("--Lk0", type=float, default=80.0)
    p.add_argument("--lags", type=int, nargs=3, default=(1, 8, 15))
    p.add_argument("--seed", type=int, default=20250304)
    p.add_argument("--bins", type=int, default=180)
    p.add_argument("--smooth", type=float, default=1.35)
    p.add_argument("--nyquist", action="store_true")
    p.add_argument("--output", "-o", type=Path,
                   default=FIGDIR / "fig_3_gaussianisation_lags.pdf")
    p.add_argument("--stats", type=Path, default=None, metavar="FILE",
                   help="also write the per-lag statistics as CSV")
    p.add_argument("--png", action="store_true")
    args = p.parse_args()
    if args.samples is None:
        args.samples = 20_000 if args.quick else 400_000
    return args


def make_figure(run, lags, bins, smooth) -> plt.Figure:
    width = plt.rcParams["figure.figsize"][0]
    fig, axes = plt.subplots(
        2, 3, figsize=(width, width * 0.47), sharex="col",
        gridspec_kw={"height_ratios": [1.62, 1.0], "hspace": 0.10, "wspace": 0.16},
    )
    m = run["r_a"].shape[0]
    res_lim = 0.012
    for col, lag in enumerate(lags):
        i = lag - 1
        rd, rt, rg = run["r_a"][:, i], run["r_t"][:, i], run["r_g"][:, i]
        edges = robust_edges(rd, np.concatenate((rt, rg)), bins)
        x = 0.5 * (edges[:-1] + edges[1:])
        dd = smooth_hist_density(rd, edges, smooth)
        dt = smooth_hist_density(rt, edges, smooth)
        dg = smooth_hist_density(rg, edges, smooth)
        peak = np.max(dd)

        ax = axes[0, col]
        ax.plot(x, dt, lw=MAIN_LINEWIDTH, color=C_TRS, label=r"transport ($y$-Gaussian)")
        ax.plot(x, dd, lw=MAIN_LINEWIDTH, ls="--", color=C_DIR, label="direct")
        ax.plot(x, dg, lw=SECONDARY_LINEWIDTH, ls=":", color=C_GR,
                label=r"Gaussian in $r$")
        ax.set_title(rf"lag $n={lag}$")
        # Headroom above the peak keeps the legend clear of the curves.
        ax.set_ylim(0, 1.42 * max(np.max(dd), np.max(dt), np.max(dg)))
        if col == 0:
            ax.set_ylabel(r"density $p(r_n)$")
            ax.legend(loc="upper left", frameon=False)

        axr = axes[1, col]
        band = residual_sampling_band(dd, dt, edges, m, smooth)
        axr.fill_between(x, -band, band, color="0.80", alpha=0.55, linewidth=0,
                         label=(r"$\pm 1\sigma$ sampling" if col == 0 else None))
        axr.axhline(0.0, lw=REFERENCE_LINEWIDTH, color="0.35")
        res_t = (dt - dd) / peak
        res_g = (dg - dd) / peak
        axr.plot(x, res_t, lw=SECONDARY_LINEWIDTH, color=C_TRS)
        axr.plot(x, res_g, lw=SECONDARY_LINEWIDTH, ls=":", color=C_GR)
        axr.set_xlabel(rf"$r_{{{lag}}}$")
        res_lim = max(res_lim, 1.08 * max(np.max(np.abs(res_t)),
                                          np.max(np.abs(res_g)),
                                          float(np.max(band))))
        if col == 0:
            axr.set_ylabel(r"$(p-p_{\rm dir})/\max p_{\rm dir}$")
            axr.legend(loc="lower left", frameon=False, fontsize=7)

    # A common residual scale makes the three lags directly comparable;
    # tick labels are kept only on the leftmost panel.
    for col, axr in enumerate(axes[1]):
        axr.set_ylim(-res_lim, res_lim)
        if col > 0:
            axr.tick_params(labelleft=False)

    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.115, top=0.955,
                        wspace=0.16, hspace=0.10)
    return fig


def main() -> None:
    args = parse_args()
    max_lag = args.grid_points // 2 - 1
    if any(lag < 1 or lag > max_lag for lag in args.lags):
        raise ValueError(f"lags must lie between 1 and {max_lag}")
    run = closure_run(args.samples, args.grid_points, args.Lk0, args.seed,
                      nyquist=args.nyquist)
    rows = []
    for lag in args.lags:
        i = lag - 1
        rd, rt, rg = run["r_a"][:, i], run["r_t"][:, i], run["r_g"][:, i]
        g_d, k_d = moments(rd)
        g_t, k_t = moments(rt)
        g_yd, k_yd = moments(run["y_a"][:, i])
        g_yt, k_yt = moments(run["y_t"][:, i])
        rows.append({"lag": lag, "dks_trs": ks(rd, rt), "dks_gr": ks(rd, rg),
                     "dks_null": ks(rd, run["r_b"][:, i]),
                     "g_r_dir": g_d, "g_r_trs": g_t, "k_r_dir": k_d, "k_r_trs": k_t,
                     "g_y_dir": g_yd, "g_y_trs": g_yt, "k_y_dir": k_yd, "k_y_trs": k_yt})
        print("lag {lag:2d}: D_KS trs={dks_trs:.4f} gauss-r={dks_gr:.4f} "
              "null={dks_null:.4f}".format(**rows[-1]))
    if args.stats is not None:
        from m2_closure_robustness import write_csv
        write_csv(args.stats, rows)

    fig = make_figure(run, args.lags, args.bins, args.smooth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path(args.output))
    print(f"Wrote {figure_path(args.output)}")
    if args.png:
        fig.savefig(figure_path(args.output, ".png"), dpi=300)
        print(f"Wrote {figure_path(args.output, '.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
