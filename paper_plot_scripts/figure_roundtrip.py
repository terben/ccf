"""Figure 2 (fig:roundtrip in CCF.tex) -- numerical behaviour of the
Levinson-Durbin recursion, Sect. 5.2.

Panel A: float64 failure rate (SingularToeplitzError or ValueError from
schurcorr.pacf) vs. N, for alpha drawn independently and uniformly from
(-b, b)^N, b in {0.9, 0.95}.

Panel B: arbitrary-precision (mpmath) roundtrip error
log10(max_n |alpha_n - alpha_n'|) vs. N, median and worst-trial band,
same alpha bounds.

Run:
    python paper_plot_scripts/figure_roundtrip.py --quick
    python paper_plot_scripts/figure_roundtrip.py --paper

--quick uses small sample counts for a fast functional check; --paper
(the default) uses the publication sample counts. See
docs/development_notes.md for why a third, O(N^2)-timing panel is not
included.

Writes: figs/fig_3_roundtrip.pdf, figs/fig_3_roundtrip.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import schurcorr as sc
from style import aa_plot

SEED = 20260714

FIGDIR = Path(__file__).resolve().parent / "../figs"
FIGDIR.mkdir(exist_ok=True)

BOUNDS = (0.9, 0.95)
BOUND_COLORS = {0.9: "tab:blue", 0.95: "tab:red"}

# Panel A trial chunk size; see failure_rate_scan and docs/development_notes.md.
TRIAL_BATCH_SIZE = 1_000

# Mirrors schurcorr.levinson._ROUNDING_TOL (not imported -- see
# _roundtrip_failure_mask below); tests/test_figure_roundtrip_panel_a.py
# cross-checks this diagnostic against schurcorr.pacf's own exceptions,
# so a future change to the library constant would surface as a test
# failure here rather than as a silent mismatch.
_ROUNDING_TOL = 1.0e-12

# Publication sample sizes (--paper, the default).
PAPER_FAILRATE_N_VALUES = (
    8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 112, 128,
)
PAPER_FAILRATE_TRIALS = 3_000
PAPER_PRECISION_N_VALUES = (16, 64, 256)
PAPER_PRECISION_TRIALS = {16: 5_000, 64: 2_500, 256: 250}
PAPER_HERO_N = 1024
PAPER_HERO_TRIALS = 15

# Small sample sizes for a fast functional check (--quick).
QUICK_FAILRATE_N_VALUES = (8, 16, 32, 64)
QUICK_FAILRATE_TRIALS = 50
QUICK_PRECISION_N_VALUES = (16, 64)
QUICK_PRECISION_TRIALS = {16: 20, 64: 10}
QUICK_HERO_N = 128
QUICK_HERO_TRIALS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", action="store_true",
        help="small sample counts for a fast functional check",
    )
    mode.add_argument(
        "--paper", action="store_true",
        help="publication sample counts (default)",
    )
    return p.parse_args()


def _roundtrip_failure_mask(r: np.ndarray) -> np.ndarray:
    """Per-row boolean mask: does the r -> alpha recursion fail for this row?

    sc.pacf's batch mode is, by design, strict for library users: for a
    batch with several problematic rows it raises for only the smallest
    offending row index (see its docstring), not a mask over all rows.
    That is the wrong shape of answer for this panel, which needs a
    pass/fail count over thousands of trials per (bound, N), and no
    public schurcorr function returns a per-row
    interior/boundary/invalid classification for a 2-D batch (see
    docs/development_notes.md for why, and for why a Python loop calling
    a per-row function defeats the batching this exists for).

    This therefore reimplements only the minimal r -> alpha arithmetic
    needed to reproduce sc.pacf's own failure criterion for every row at
    once -- not the alpha/sigma2/phi bookkeeping the library keeps for
    its public return values. tests/test_figure_roundtrip_panel_a.py
    cross-checks the resulting mask against sc.pacf's actual per-row
    exceptions, so this is not an independent claim about the recursion.
    """
    n_samples, n_max = r.shape
    phi = np.zeros((n_samples, n_max), dtype=np.float64)
    sigma2 = np.ones(n_samples, dtype=np.float64)
    active = np.ones(n_samples, dtype=bool)
    failed = np.zeros(n_samples, dtype=bool)

    for n in range(1, n_max + 1):
        idx = n - 1
        if not np.any(active):
            break

        # A row whose sigma2 already rounded to the boundary on the
        # previous step fails here too, without a new alpha_n to check.
        pre_boundary = active & (sigma2 <= _ROUNDING_TOL)
        failed[pre_boundary] = True
        active[pre_boundary] = False
        computing = active

        if n == 1:
            pred = np.zeros(n_samples, dtype=np.float64)
        else:
            pred = np.sum(phi[:, :idx] * r[:, idx - 1 :: -1], axis=1)

        sigma2_safe = np.where(computing, sigma2, 1.0)
        alpha_n = (r[:, idx] - pred) / sigma2_safe
        abs_alpha_n = np.abs(alpha_n)

        invalid_now = computing & (abs_alpha_n > 1.0 + _ROUNDING_TOL)
        boundary_now = computing & ~invalid_now & (abs_alpha_n >= 1.0)
        continue_now = computing & ~(invalid_now | boundary_now)

        failed[invalid_now | boundary_now] = True
        active[invalid_now | boundary_now] = False

        update_mask = continue_now | boundary_now
        alpha_step = np.where(boundary_now, np.sign(alpha_n), alpha_n)

        if n == 1:
            phi[:, 0] = np.where(update_mask, alpha_step, phi[:, 0])
        else:
            phi_prev = phi[:, :idx].copy()
            new_head = phi_prev - alpha_step[:, None] * phi_prev[:, ::-1]
            phi[:, :idx] = np.where(update_mask[:, None], new_head, phi_prev)
            phi[:, idx] = np.where(update_mask, alpha_step, phi[:, idx])

        sigma2_next = sigma2 * (1.0 - alpha_step * alpha_step)
        clamp = continue_now & (sigma2_next < 0.0) & (sigma2_next > -_ROUNDING_TOL)
        sigma2_next = np.where(clamp, 0.0, sigma2_next)
        sigma2 = np.where(update_mask, sigma2_next, sigma2)

    return failed


def failure_rate_scan(
    n_values: tuple[int, ...], trials: int, rng: np.random.Generator,
) -> dict[tuple[float, int], float]:
    """Panel A data: fraction of admissibility failures over `trials`
    draws, for each (bound, N).

    Trials are processed in chunks because schurcorr's NumPy recursion is
    vectorized across samples; calling sc.from_pacf/sc.pacf once per
    trial would spend most of the runtime in per-call Python/NumPy
    dispatch overhead rather than the O(N) arithmetic itself.
    """
    results = {}
    for bound in BOUNDS:
        for N in n_values:
            fails = 0
            for start in range(0, trials, TRIAL_BATCH_SIZE):
                m = min(TRIAL_BATCH_SIZE, trials - start)
                alpha_batch = rng.uniform(-bound, bound, size=(m, N))
                r_batch = sc.from_pacf(alpha_batch)
                failed_mask = _roundtrip_failure_mask(r_batch)
                fails += int(np.count_nonzero(failed_mask))
            results[(bound, N)] = fails / trials
    return results


def mp_roundtrip_error(alpha: np.ndarray, dps: int) -> float:
    """max_n |alpha_n - alpha_n'| at arbitrary precision, compared in
    the mp domain (not after downcasting to float64)."""
    r_mp = sc.from_pacf_mp(alpha, dps=dps)
    rec_mp = sc.pacf_mp(r_mp, dps=dps)
    err = max(abs(mp.mpf(a) - b) for a, b in zip(alpha, rec_mp))
    return float(err)


def precision_scan(
    n_values: tuple[int, ...],
    trials_by_n: dict[int, int],
    hero_n: int,
    hero_trials: int,
    rng: np.random.Generator,
) -> dict[tuple[float, int], dict[str, float]]:
    """Panel B data: median / worst arbitrary-precision roundtrip error
    for each (bound, N), including the hero point at `hero_n`."""
    results = {}
    for bound in BOUNDS:
        for N in (*n_values, hero_n):
            trials = trials_by_n.get(N, hero_trials)
            dps = sc.recommended_dps(N)
            errs = np.empty(trials)
            for i in range(trials):
                alpha = rng.uniform(-bound, bound, size=N)
                errs[i] = mp_roundtrip_error(alpha, dps)
            results[(bound, N)] = dict(
                median=float(np.median(errs)), worst=float(np.max(errs))
            )
    return results


def make_figure(
    failrate: dict,
    failrate_n_values: tuple[int, ...],
    precision: dict,
    precision_n_values: tuple[int, ...],
    hero_n: int,
    path_stub: Path,
) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2)

    for ax in (ax_a, ax_b):
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlabel(ax.get_xlabel(), labelpad=6)
        ax.set_ylabel(ax.get_ylabel(), labelpad=8)

    # Panel A
    for bound in BOUNDS:
        c = BOUND_COLORS[bound]
        y = [100 * failrate[(bound, N)] for N in failrate_n_values]
        ax_a.plot(
            failrate_n_values, y,
            marker="o", linestyle="-",
            color=c, label=f"b={bound}",
            lw=2.4, ms=5.2, alpha=0.95
        )
    ax_a.set_xscale("log")
    ax_a.set_xlabel(r"$N$")
    ax_a.set_ylabel("failed trials [%]")
    ax_a.set_title("A: float64 failure rate")
    ax_a.legend(framealpha=0.92, loc="best")

    # Panel B
    precision_Ns = (*precision_n_values, hero_n)
    for bound in BOUNDS:
        c = BOUND_COLORS[bound]
        med = [np.log10(precision[(bound, N)]["median"]) for N in precision_Ns]
        worst = [np.log10(precision[(bound, N)]["worst"]) for N in precision_Ns]
        ax_b.plot(
            precision_Ns, med,
            marker="s", linestyle="--",
            color=c, label=f"b={bound}",
            lw=2.4, ms=5.2, alpha=0.95
        )
        ax_b.fill_between(precision_Ns, med, worst, color=c, alpha=0.12, linewidth=0)
    ax_b.axvline(hero_n, color="gray", linestyle=":", linewidth=1.3, alpha=0.9)
    ax_b.set_xscale("log")
    ax_b.set_xticks(list(precision_Ns))
    ax_b.set_xticklabels([str(N) for N in precision_Ns])
    ax_b.set_xlabel(r"$N$")
    ax_b.set_ylabel(r"$\log_{10}\,\max_n|\alpha_n-\alpha_n'|$")
    ax_b.set_title("B: arbitrary precision (mpmath)")
    ax_b.legend(framealpha=0.92, loc="best")

    # constrained_layout is enabled globally by aa_plot.
    fig.savefig(path_stub.with_suffix(".pdf"))
    fig.savefig(path_stub.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    quick = args.quick

    aa_plot(column="double", height_ratio=0.25, fontsize=8.0, use_tex=True)
    rng = np.random.default_rng(SEED)

    failrate_n_values = QUICK_FAILRATE_N_VALUES if quick else PAPER_FAILRATE_N_VALUES
    failrate_trials = QUICK_FAILRATE_TRIALS if quick else PAPER_FAILRATE_TRIALS
    precision_n_values = QUICK_PRECISION_N_VALUES if quick else PAPER_PRECISION_N_VALUES
    precision_trials = QUICK_PRECISION_TRIALS if quick else PAPER_PRECISION_TRIALS
    hero_n = QUICK_HERO_N if quick else PAPER_HERO_N
    hero_trials = QUICK_HERO_TRIALS if quick else PAPER_HERO_TRIALS

    t0 = time.time()
    failrate = failure_rate_scan(failrate_n_values, failrate_trials, rng)
    t1 = time.time()
    print(f"Panel A (failure rate) done in {t1 - t0:.1f}s")

    precision = precision_scan(
        precision_n_values, precision_trials, hero_n, hero_trials, rng
    )
    t2 = time.time()
    print(f"Panel B (mp precision, incl. N={hero_n} hero) done in {t2 - t1:.1f}s")

    make_figure(
        failrate, failrate_n_values,
        precision, precision_n_values, hero_n,
        FIGDIR / "fig_3_roundtrip",
    )
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
