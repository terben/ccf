#!/usr/bin/env python3
"""New Fig. 4 (Sect. 5.3): closure diagnostics as a function of lag.

Reads the multi-seed output of ``m2_closure_robustness.py null`` and shows,
for all lags n = 1, ..., N/2 - 1 (mean and seed-to-seed standard deviation):

(a) Kolmogorov--Smirnov distances to the direct branch for the y-space
    transport and for the naive Gaussian in r, together with the sampling
    null D_KS between two independent direct samples;
(b) skewness and (c) excess kurtosis of r_n in the direct and transport
    branches, and of y_n in the direct branch.

Faint vertical lines mark the commensurate lags n = N/4 and n ~ N/3.
The OJA style is inherited by importing figure_gaussianization.

Run (conda env ``ccf``):
    python paper_plot_scripts/m2_closure_robustness.py null
    python paper_plot_scripts/figure_closure_vs_lag.py --png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_gaussianization import FIGDIR, REFERENCE_LINEWIDTH, SECONDARY_LINEWIDTH
from m2_closure_robustness import OUTDIR
from style import figure_path

plt.rcParams["savefig.bbox"] = None

C_TRS, C_DIR, C_GR, C_Y = "C0", "C1", "C2", "0.45"


def load(path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    """Return {column: {lag: (mean, std over seeds)}}."""
    raw: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as fh:
        for row in csv.DictReader(fh):
            lag = int(row["lag"])
            for key, val in row.items():
                if key not in ("lag", "seed", "M", "N", "Lk0"):
                    raw[key][lag].append(float(val))
    return {k: {lag: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                for lag, v in d.items()} for k, d in raw.items()}


def series(stats, key):
    lags = np.array(sorted(stats[key]))
    mean = np.array([stats[key][n][0] for n in lags])
    std = np.array([stats[key][n][1] for n in lags])
    return lags, mean, std


def errplot(ax, stats, key, offset, **kw):
    lags, mean, std = series(stats, key)
    ax.errorbar(lags + offset, mean, yerr=std, lw=SECONDARY_LINEWIDTH,
                elinewidth=0.8, capsize=0, ms=3.2, **kw)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=OUTDIR / "null.csv")
    p.add_argument("--grid-points", "-N", type=int, default=32,
                   help="only used to place the commensurate-lag markers")
    p.add_argument("--output", "-o", type=Path,
                   default=FIGDIR / "fig_4_closure_vs_lag.pdf")
    p.add_argument("--png", action="store_true")
    args = p.parse_args()

    stats = load(args.input)
    width = plt.rcParams["figure.figsize"][0]
    fig, axes = plt.subplots(1, 3, figsize=(width, width * 0.31))

    ax = axes[0]
    lags, null_m, null_s = series(stats, "dks_ab")
    ax.fill_between(lags, 0, null_m + null_s, color="0.80", alpha=0.55,
                    linewidth=0, label="sampling null")
    errplot(ax, stats, "dks_at", 0.0, color=C_TRS, marker="o",
            label=r"transport ($y$-Gaussian)")
    errplot(ax, stats, "dks_ag", 0.0, color=C_GR, marker="s", ls=":",
            label=r"Gaussian in $r$")
    ax.set_ylabel(r"$D_{\rm KS}$")
    ax.set_ylim(0, 1.3 * float(np.max(series(stats, "dks_ag")[1])))
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    errplot(ax, stats, "g_r_dir", -0.12, color=C_DIR, marker="o", ls="--",
            label=r"$r_n$ direct")
    errplot(ax, stats, "g_r_trs", +0.12, color=C_TRS, marker="o",
            label=r"$r_n$ transport")
    errplot(ax, stats, "g_y_dir", 0.0, color=C_Y, marker="^", ls="-",
            mfc="white", label=r"$y_n$ direct")
    ax.set_ylabel(r"skewness $\gamma$")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax.legend(loc="upper left", frameon=False)

    ax = axes[2]
    errplot(ax, stats, "k_r_dir", -0.12, color=C_DIR, marker="o", ls="--")
    errplot(ax, stats, "k_r_trs", +0.12, color=C_TRS, marker="o")
    errplot(ax, stats, "k_y_dir", 0.0, color=C_Y, marker="^", ls="-", mfc="white")
    ax.set_ylabel(r"excess kurtosis $\kappa$")

    special = [args.grid_points / 4.0, round(args.grid_points / 3.0)]
    for ax in axes:
        ax.axhline(0.0, lw=REFERENCE_LINEWIDTH, color="0.35", zorder=0)
        for s in special:
            ax.axvline(s, lw=REFERENCE_LINEWIDTH, color="0.75", ls="-", zorder=0)
        ax.set_xlabel(r"lag $n$")
        ax.set_xlim(0.3, lags.max() + 0.7)
        ax.set_xticks([1, 4, 8, 12, 15] if lags.max() == 15 else None)

    fig.subplots_adjust(left=0.068, right=0.992, bottom=0.165, top=0.975,
                        wspace=0.30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path(args.output))
    print(f"Wrote {figure_path(args.output)}")
    if args.png:
        fig.savefig(figure_path(args.output, ".png"), dpi=300)
        print(f"Wrote {figure_path(args.output, '.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
