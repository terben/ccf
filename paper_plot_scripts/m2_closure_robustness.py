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

``fieldcheck``
    Validates the power-spectrum route against explicit periodic Gaussian
    field realisations (FFT), for both mode conventions below.

Every experiment also records D_KS(A, G), where G is drawn from a
multivariate Gaussian *in r-space* with the mean and covariance of A -- the
naive Gaussian-likelihood baseline against which the y-space transport is
to be judged.

Mode conventions (``--nyquist``)
    Default ("ws"): zero and Nyquist modes omitted, as in WS2013 and in
    figure_gaussianization.py.  On the periodic grid this enforces the exact
    linear constraint r_1 + r_3 + ... + r_{N/2-1} = 0 (difference of the
    zero-mode and Nyquist-mode sum rules), so the direct r-distribution is
    supported on a hyperplane: N/2-1 lags but only N/2-2 degrees of freedom.
    With ``--nyquist`` the (real-valued) Nyquist mode is kept; its power is
    lambda * chi^2_1 and it enters xi_n with half weight and sign (-1)^n.
    The constraint then involves xi_{N/2}, which is not analysed, and the
    r_1, ..., r_{N/2-1} distribution is non-degenerate.

Results are written as CSV files to ``m2_output/`` next to this script
(suffix ``_nyq`` for the Nyquist convention).

Run (from the repository root, conda env ``ccf``):
    python paper_plot_scripts/m2_closure_robustness.py null [--nyquist]
    python paper_plot_scripts/m2_closure_robustness.py mscan [--nyquist]
    python paper_plot_scripts/m2_closure_robustness.py config [--nyquist]
    python paper_plot_scripts/m2_closure_robustness.py fieldcheck
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


def mean_power(mode: np.ndarray, lk0: float) -> np.ndarray:
    """Gaussian-shape power spectrum, same convention as figure_gaussianization."""
    return np.exp(-((2.0 * np.pi * mode / lk0) ** 2))


def simulate_direct_r_nyquist(
    samples: int, n_grid: int, lk0: float, rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    """Power-spectrum route with the Nyquist mode kept (zero mode omitted).

    Paired modes m = 1, ..., N/2-1 carry exponential powers lambda_m E_m and
    enter xi_n as 2 lambda_m E_m cos(2 pi m n / N); the real Nyquist mode
    carries lambda_{N/2} chi^2_1 and enters as (-1)^n.  The common factor 2
    of the paired modes is kept explicitly here because the relative weight
    of the Nyquist term matters.
    """
    if n_grid < 4 or n_grid % 2:
        raise ValueError("N must be an even integer >= 4")
    n_lags = n_grid // 2 - 1
    mode = np.arange(1, n_grid // 2, dtype=float)
    lam = mean_power(mode, lk0)
    lam_nyq = float(mean_power(np.array([n_grid / 2.0]), lk0)[0])
    lag = np.arange(0, n_lags + 1, dtype=float)
    cosine = 2.0 * np.cos(2.0 * np.pi * np.outer(mode, lag) / n_grid)
    sign = (-1.0) ** lag

    result = np.empty((samples, n_lags), dtype=np.float64)
    start = 0
    while start < samples:
        stop = min(start + batch_size, samples)
        powers = rng.exponential(scale=lam, size=(stop - start, n_lags))
        p_nyq = lam_nyq * rng.standard_normal(stop - start) ** 2
        xi = powers @ cosine + np.outer(p_nyq, sign)
        result[start:stop] = xi[:, 1:] / xi[:, [0]]
        start = stop
    return result


def simulate_field_r(
    samples: int, n_grid: int, lk0: float, rng: np.random.Generator,
    nyquist: bool, batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Explicit periodic Gaussian fields via FFT, periodic xi estimator.

    Reference for ``fieldcheck``: complex Gaussian amplitudes with
    <|g_m|^2> = lambda_m for 0 < m < N/2, zero mean (g_0 = 0), and a real
    Gaussian Nyquist amplitude with variance lambda_{N/2} if ``nyquist``.
    """
    half = n_grid // 2
    mode = np.arange(0, half + 1, dtype=float)
    lam = mean_power(mode, lk0)
    lam[0] = 0.0
    if not nyquist:
        lam[half] = 0.0
    result = np.empty((samples, half - 1), dtype=np.float64)
    start = 0
    while start < samples:
        stop = min(start + batch_size, samples)
        b = stop - start
        amp = (rng.standard_normal((b, half + 1))
               + 1j * rng.standard_normal((b, half + 1))) * np.sqrt(lam / 2.0)
        amp[:, half] = rng.standard_normal(b) * np.sqrt(lam[half])
        g = np.fft.irfft(amp, n=n_grid, axis=1)
        spec = np.abs(np.fft.rfft(g, axis=1)) ** 2
        xi = np.fft.irfft(spec, n=n_grid, axis=1)  # periodic estimator
        result[start:stop] = xi[:, 1:half] / xi[:, [0]]
        start = stop
    return result


def simulate(samples, n_grid, lk0, rng, nyquist):
    if nyquist:
        return simulate_direct_r_nyquist(samples, n_grid, lk0, rng, BATCH_SIZE)
    return simulate_direct_r(samples, n_grid, lk0, rng, BATCH_SIZE)


def closure_run(
    samples: int, n_grid: int, lk0: float, seed: int, diagonal_cov: bool = False,
    nyquist: bool = False,
) -> dict[str, np.ndarray]:
    """Direct sample A, independent direct sample B, transport T fitted to A,
    and the r-space Gaussian baseline G fitted to A.

    The A and T streams use the same seed-spawning scheme as
    figure_gaussianization.main(), so seed 20250304 (default mode
    convention) reproduces Table 3.
    """
    ss_a, ss_t = np.random.SeedSequence(seed).spawn(2)
    rng_a, rng_t = np.random.default_rng(ss_a), np.random.default_rng(ss_t)
    rng_b = np.random.default_rng(np.random.SeedSequence([seed, 777]))
    rng_g = np.random.default_rng(np.random.SeedSequence([seed, 999]))

    r_a = simulate(samples, n_grid, lk0, rng_a, nyquist)
    r_b = simulate(samples, n_grid, lk0, rng_b, nyquist)
    y_a = r_to_y(r_a)
    y_t, _, _ = gaussian_transport(y_a, rng_t, diagonal_cov)
    r_t = ccf.from_pacf(np.tanh(y_t))
    # r-space Gaussian baseline.  In the default convention its covariance is
    # singular (see module docstring); an eigen-decomposition handles that.
    cov_r = np.cov(r_a, rowvar=False)
    w, v = np.linalg.eigh(cov_r)
    z = rng_g.standard_normal(r_a.shape)
    r_g = r_a.mean(axis=0) + (z * np.sqrt(np.clip(w, 0.0, None))) @ v.T
    return {"r_a": r_a, "r_b": r_b, "r_t": r_t, "r_g": r_g,
            "y_a": y_a, "y_t": y_t}


def ks(a: np.ndarray, b: np.ndarray) -> float:
    return float(ks_2samp(a, b, method="asymp").statistic)


def lag_rows(run: dict[str, np.ndarray], **meta) -> list[dict]:
    rows = []
    for idx in range(run["r_a"].shape[1]):
        ra, rb, rt = run["r_a"][:, idx], run["r_b"][:, idx], run["r_t"][:, idx]
        rg = run["r_g"][:, idx]
        ya, yt = run["y_a"][:, idx], run["y_t"][:, idx]
        g_ra, k_ra = moments(ra)
        g_rt, k_rt = moments(rt)
        g_ya, k_ya = moments(ya)
        g_yt, k_yt = moments(yt)
        rows.append({
            **meta, "lag": idx + 1,
            "dks_at": ks(ra, rt), "dks_bt": ks(rb, rt), "dks_ab": ks(ra, rb),
            "dks_ag": ks(ra, rg),
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


def suffix(args) -> str:
    return "_nyq" if args.nyquist else ""


def exp_null(args) -> None:
    rows = []
    for k in range(args.seeds):
        seed = args.seed + k
        t0 = time.time()
        run = closure_run(args.samples, args.grid_points, args.Lk0, seed,
                          nyquist=args.nyquist)
        rows += lag_rows(run, seed=seed, M=args.samples,
                         N=args.grid_points, Lk0=args.Lk0)
        print(f"seed {seed}: {time.time() - t0:.1f}s")
    write_csv(OUTDIR / f"null{suffix(args)}.csv", rows)


def exp_mscan(args) -> None:
    rows = []
    for m in (25_000, 50_000, 100_000, 200_000, 400_000, 800_000):
        for k in range(args.seeds):
            run = closure_run(m, args.grid_points, args.Lk0, args.seed + k,
                              nyquist=args.nyquist)
            rows += lag_rows(run, seed=args.seed + k, M=m,
                             N=args.grid_points, Lk0=args.Lk0)
        print(f"M = {m:,} done")
    write_csv(OUTDIR / f"mscan{suffix(args)}.csv", rows)


def exp_config(args) -> None:
    rows = []
    grid = [(n, lk0, False) for n in (16, 32, 64) for lk0 in (20.0, 40.0, 80.0, 160.0)]
    grid.append((32, 80.0, True))
    for n, lk0, diag in grid:
        try:
            run = closure_run(args.samples, n, lk0, args.seed, diagonal_cov=diag,
                              nyquist=args.nyquist)
        except (ccf.SingularToeplitzError, ValueError) as err:
            print(f"N={n} Lk0={lk0} diag={diag}: skipped ({err})")
            continue
        rows += lag_rows(run, seed=args.seed, M=args.samples, N=n, Lk0=lk0,
                         diag=int(diag))
        print(f"N={n} Lk0={lk0:g} diag={diag} done")
    write_csv(OUTDIR / f"config{suffix(args)}.csv", rows)


def exp_fieldcheck(args) -> None:
    """Power route vs. explicit fields, both conventions, all lags."""
    rows = []
    for nyq in (False, True):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 31]))
        r_pow = simulate(args.samples, args.grid_points, args.Lk0, rng, nyq)
        r_fld = simulate_field_r(args.samples, args.grid_points, args.Lk0,
                                 rng, nyq)
        odd_pow = np.abs(r_pow[:, 0::2].sum(axis=1)).max()
        odd_fld = np.abs(r_fld[:, 0::2].sum(axis=1)).max()
        print(f"nyquist={nyq}: max |sum_odd r_n| power={odd_pow:.2e} "
              f"field={odd_fld:.2e}")
        for idx in range(r_pow.shape[1]):
            rows.append({
                "nyquist": int(nyq), "lag": idx + 1,
                "dks_power_field": ks(r_pow[:, idx], r_fld[:, idx]),
                "mean_power": float(r_pow[:, idx].mean()),
                "mean_field": float(r_fld[:, idx].mean()),
                "sd_power": float(r_pow[:, idx].std()),
                "sd_field": float(r_fld[:, idx].std()),
            })
    write_csv(OUTDIR / "fieldcheck.csv", rows)


def exp_decompose(args) -> None:
    """Which part of the y-model limits the closure: marginals or dependence?

    Per lag, D_KS against the direct sample A for
      gauss_y : the paper's transport (Gaussian in y, mean + covariance),
      copula  : Gaussian copula in y with the *exact* empirical y-marginals,
      indep   : exact y-marginals, all cross-lag dependence destroyed,
      gauss_r : Gaussian in r (baseline),  null: independent direct sample.
    If ``copula`` is not better than ``gauss_y``, the limitation lies in
    non-Gaussian dependence between the y_n, not in their marginals.
    Also writes corr((y_n-<y_n>)^2, (y_k-<y_k>)^2) for direct and transport;
    for any Gaussian this equals rho_nk^2, so excess values measure
    variance coupling between lags.
    """
    from scipy.stats import norm, rankdata
    rows, crows = [], []
    for nyq in (False, True):
        run = closure_run(args.samples, args.grid_points, args.Lk0, args.seed,
                          nyquist=nyq)
        ra, ya, yt = run["r_a"], run["y_a"], run["y_t"]
        m, d = ya.shape
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, 4242]))
        u = (np.apply_along_axis(rankdata, 0, ya) - 0.5) / m
        corr = np.corrcoef(norm.ppf(u), rowvar=False)
        zz = rng.standard_normal((m, d)) @ np.linalg.cholesky(corr).T
        idx = np.clip((norm.cdf(zz) * m).astype(int), 0, m - 1)
        y_cop = np.take_along_axis(np.sort(ya, axis=0), idx, axis=0)
        r_cop = ccf.from_pacf(np.tanh(y_cop))
        y_ind = np.column_stack([rng.permutation(ya[:, j]) for j in range(d)])
        r_ind = ccf.from_pacf(np.tanh(y_ind))
        for n in range(d):
            rows.append({
                "nyquist": int(nyq), "lag": n + 1,
                "dks_gauss_y": ks(ra[:, n], run["r_t"][:, n]),
                "dks_copula": ks(ra[:, n], r_cop[:, n]),
                "dks_indep": ks(ra[:, n], r_ind[:, n]),
                "dks_gauss_r": ks(ra[:, n], run["r_g"][:, n]),
                "dks_null": ks(ra[:, n], run["r_b"][:, n]),
            })
        sq_a = (ya - ya.mean(axis=0)) ** 2
        sq_t = (yt - yt.mean(axis=0)) ** 2
        c_a = np.corrcoef(sq_a, rowvar=False)
        c_t = np.corrcoef(sq_t, rowvar=False)
        rho2 = np.corrcoef(ya, rowvar=False) ** 2
        for i in range(d):
            for j in range(i + 1, d):
                crows.append({"nyquist": int(nyq), "n": i + 1, "k": j + 1,
                              "sqcorr_dir": float(c_a[i, j]),
                              "sqcorr_trs": float(c_t[i, j]),
                              "rho2_dir": float(rho2[i, j])})
        print(f"nyquist={nyq} done")
    write_csv(OUTDIR / "decompose.csv", rows)
    write_csv(OUTDIR / "decompose_sqcorr.csv", crows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment",
                   choices=("null", "mscan", "config", "fieldcheck", "decompose"))
    p.add_argument("--nyquist", action="store_true",
                   help="keep the Nyquist mode (non-degenerate r-distribution)")
    p.add_argument("--samples", "-M", type=int, default=400_000)
    p.add_argument("--grid-points", "-N", type=int, default=32)
    p.add_argument("--Lk0", type=float, default=80.0)
    p.add_argument("--seed", type=int, default=20250304)
    p.add_argument("--seeds", type=int, default=8,
                   help="number of consecutive seeds (null: 8, mscan: use 4)")
    args = p.parse_args()
    {"null": exp_null, "mscan": exp_mscan, "config": exp_config,
     "fieldcheck": exp_fieldcheck, "decompose": exp_decompose}[args.experiment](args)


if __name__ == "__main__":
    main()
