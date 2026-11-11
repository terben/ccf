#!/usr/bin/env python3
"""Robustness diagnostics for the quasi-Gaussian closure test (Sect. 5.3).

Referee point M2 asks whether the closure seen at lags n = 1, 2, 3 holds at
all simulated lags.  This script re-uses the simulation and transport code of
``figure_gaussianization.py`` (imported, not modified) and runs three
experiments over *all* lags n = 1, ..., N/2 - 1:

``null``
    Several seeds at fixed (M, N, L k0).  For every seed we draw a direct
    sample A, an independent direct sample B of the same size, and the
    transport sample T fitted to A.  We record per lag

    * D_KS(A, T)  -- the in-sample closure statistic of the paper,
    * D_KS(B, T)  -- out-of-sample closure statistic,
    * D_KS(A, B)  -- the pure sampling null (two direct samples),

    together with skewness / excess kurtosis of r and y in A and T.

``mscan``
    D_KS(A, T) and D_KS(A, B) as a function of M.  A sampling-limited
    statistic scales as M^{-1/2}; a systematic mismatch saturates.

``config``
    One seed per configuration on a grid of N and L k0 (and with the
    diagonal-covariance variant), to see whether deviations are tied to
    specific lags, to lags relative to N (e.g. N/4), or to the spectrum.

Results are written as CSV files to ``m2_output/`` next to this script.

Run (from the repository root, conda env ``ccf``):
    python paper_plot_scripts/m2_closure_robustness.py null
    python paper_plot_scripts/m2_closure_robustness.py mscan
    python paper_plot_scripts/m2_closure_robustness.py config
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

import ccf
from figure_gaussianization import gaussian_transport, moments, simulate_direct_r

OUTDIR = Path(__file__).resolve().parent / "m2_output"
BATCH_SIZE = 50_000
EPS = 8.0 * np.finfo(float).eps


def r_to_y(r: np.ndarray) -> np.ndarray:
    alpha = np.clip(ccf.pacf(r), -1.0 + EPS, 1.0 - EPS)
    return np.arctanh(alpha)


def closure_run(
    samples: int, n_grid: int, lk0: float, seed: int, diagonal_cov: bool = False,
) -> dict[str, np.ndarray]:
    """Direct sample A, independent direct sample B, transport T fitted to A.

    The A and T streams use the same seed-spawning scheme as
    figure_gaussianization.main(), so seed 20250304 reproduces Table 3.
    """
    ss_a, ss_t = np.random.SeedSequence(seed).spawn(2)
    rng_a, rng_t = np.random.default_rng(ss_a), np.random.default_rng(ss_t)
    rng_b = np.random.default_rng(np.random.SeedSequence([seed, 777]))

    r_a = simulate_direct_r(samples, n_grid, lk0, rng_a, BATCH_SIZE)
    r_b = simulate_direct_r(samples, n_grid, lk0, rng_b, BATCH_SIZE)
    y_a = r_to_y(r_a)
    y_t, _, _ = gaussian_transport(y_a, rng_t, diagonal_cov)
    r_t = ccf.from_pacf(np.tanh(y_t))
    return {"r_a": r_a, "r_b": r_b, "r_t": r_t, "y_a": y_a, "y_t": y_t}


def ks(a: np.ndarray, b: np.ndarray) -> float:
    return float(ks_2samp(a, b, method="asymp").statistic)


def lag_rows(run: dict[str, np.ndarray], **meta) -> list[dict]:
    rows = []
    for idx in range(run["r_a"].shape[1]):
        ra, rb, rt = run["r_a"][:, idx], run["r_b"][:, idx], run["r_t"][:, idx]
        ya, yt = run["y_a"][:, idx], run["y_t"][:, idx]
        g_ra, k_ra = moments(ra)
        g_rt, k_rt = moments(rt)
        g_ya, k_ya = moments(ya)
        g_yt, k_yt = moments(yt)
        rows.append({
            **meta, "lag": idx + 1,
            "dks_at": ks(ra, rt), "dks_bt": ks(rb, rt), "dks_ab": ks(ra, rb),
            "g_r_dir": g_ra, "g_r_trs": g_rt, "k_r_dir": k_ra, "k_r_trs": k_rt,
            "g_y_dir": g_ya, "g_y_trs": g_yt, "k_y_dir": k_ya, "k_y_trs": k_yt,
            "mean_r": float(ra.mean()), "sd_r": float(ra.std()),
            "mean_y": float(ya.mean()), "sd_y": float(ya.std()),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"Wrote {path}")


def exp_null(args) -> None:
    rows = []
    for k in range(args.seeds):
        seed = args.seed + k
        t0 = time.time()
        run = closure_run(args.samples, args.grid_points, args.Lk0, seed)
        rows += lag_rows(run, seed=seed, M=args.samples,
                         N=args.grid_points, Lk0=args.Lk0)
        print(f"seed {seed}: {time.time() - t0:.1f}s")
    write_csv(OUTDIR / "null.csv", rows)


def exp_mscan(args) -> None:
    rows = []
    for m in (25_000, 50_000, 100_000, 200_000, 400_000, 800_000):
        for k in range(args.seeds):
            run = closure_run(m, args.grid_points, args.Lk0, args.seed + k)
            rows += lag_rows(run, seed=args.seed + k, M=m,
                             N=args.grid_points, Lk0=args.Lk0)
        print(f"M = {m:,} done")
    write_csv(OUTDIR / "mscan.csv", rows)


def exp_config(args) -> None:
    rows = []
    grid = [(n, lk0, False) for n in (16, 32, 64) for lk0 in (20.0, 40.0, 80.0, 160.0)]
    grid.append((32, 80.0, True))
    for n, lk0, diag in grid:
        try:
            run = closure_run(args.samples, n, lk0, args.seed, diagonal_cov=diag)
        except (ccf.SingularToeplitzError, ValueError) as err:
            print(f"N={n} Lk0={lk0} diag={diag}: skipped ({err})")
            continue
        rows += lag_rows(run, seed=args.seed, M=args.samples, N=n, Lk0=lk0,
                         diag=int(diag))
        print(f"N={n} Lk0={lk0:g} diag={diag} done")
    write_csv(OUTDIR / "config.csv", rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=("null", "mscan", "config"))
    p.add_argument("--samples", "-M", type=int, default=400_000)
    p.add_argument("--grid-points", "-N", type=int, default=32)
    p.add_argument("--Lk0", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=20250304)
    p.add_argument("--seeds", type=int, default=8,
                   help="number of consecutive seeds (null: 8, mscan: use 4)")
    args = p.parse_args()
    {"null": exp_null, "mscan": exp_mscan, "config": exp_config}[args.experiment](args)


if __name__ == "__main__":
    main()
