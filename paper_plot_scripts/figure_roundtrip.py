"""Figure 2 (fig:roundtrip in CCF.tex) -- numerical behaviour of the
Levinson-Durbin recursion, Sect. 5.2.

Panel A: float64 failure rate (SingularToeplitzError or ValueError from
ccf.pacf) vs. N, for alpha drawn independently and uniformly from
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
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import ccf
from style import aa_plot

DEFAULT_JOBS = min(8, os.cpu_count() or 1)

SEED = 20260714

FIGDIR = Path(__file__).resolve().parent / "../figs"
FIGDIR.mkdir(exist_ok=True)

BOUNDS = (0.9, 0.95)
BOUND_COLORS = {0.9: "tab:blue", 0.95: "tab:red"}

# Panel A trial chunk size; see failure_rate_scan and docs/development_notes.md.
TRIAL_BATCH_SIZE = 1_000

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
    p.add_argument(
        "-j", "--jobs", type=int, default=None,
        help=(
            "number of worker processes for Panel B arbitrary-precision "
            f"trials (default: min(8, available CPUs) = {DEFAULT_JOBS}; "
            "use 1 for serial execution; --quick defaults to 1 unless set "
            "explicitly)"
        ),
    )
    args = p.parse_args()
    if args.jobs is not None and args.jobs < 1:
        p.error("--jobs must be >= 1")
    return args


def failure_rate_scan(
    n_values: tuple[int, ...], trials: int, rng: np.random.Generator,
) -> dict[tuple[float, int], float]:
    """Panel A data: fraction of admissibility failures over `trials`
    draws, for each (bound, N).

    Trials are processed in chunks because ccf's NumPy recursion is
    vectorized across samples; calling the NumPy Levinson recursions once
    per trial would spend most of the runtime in per-call Python/NumPy
    dispatch overhead rather than the O(N) arithmetic itself.
    """
    results = {}
    for bound in BOUNDS:
        for N in n_values:
            fails = 0
            for start in range(0, trials, TRIAL_BATCH_SIZE):
                m = min(TRIAL_BATCH_SIZE, trials - start)
                alpha_batch = rng.uniform(-bound, bound, size=(m, N))
                r_batch = ccf.from_pacf(alpha_batch)
                status = ccf.pacf_status(r_batch)
                failed = status.boundary | status.invalid
                fails += int(np.count_nonzero(failed))
            results[(bound, N)] = fails / trials
    return results


def mp_roundtrip_error(alpha: np.ndarray, dps: int) -> float:
    """max_n |alpha_n - alpha_n'| at arbitrary precision, compared in
    the mp domain (not after downcasting to float64)."""
    r_mp = ccf.from_pacf_mp(alpha, dps=dps)
    rec_mp = ccf.pacf_mp(r_mp, dps=dps)
    err = max(abs(mp.mpf(a) - b) for a, b in zip(alpha, rec_mp))
    return float(err)


def _mp_roundtrip_error_chunk(alpha_chunk: np.ndarray, dps: int) -> np.ndarray:
    """Compute mp_roundtrip_error for each row of a chunk (worker entry point)."""
    return np.array([mp_roundtrip_error(alpha, dps) for alpha in alpha_chunk])


def precision_scan(
    n_values: tuple[int, ...],
    trials_by_n: dict[int, int],
    hero_n: int,
    hero_trials: int,
    rng: np.random.Generator,
    jobs: int = 1,
) -> dict[tuple[float, int], dict[str, float]]:
    """Panel B data: median / worst arbitrary-precision roundtrip error
    for each (bound, N), including the hero point at `hero_n`.

    Trials are independent; for jobs > 1 they are computed in a
    reused ProcessPoolExecutor. Random draws always happen in this
    process first, so the Monte Carlo sample is identical regardless
    of `jobs`.
    """
    results = {}

    with (ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else nullcontext(None)) as pool:
        for bound in BOUNDS:
            for N in (*n_values, hero_n):
                trials = trials_by_n.get(N, hero_trials)
                dps = ccf.recommended_dps(N)
                alpha_samples = rng.uniform(-bound, bound, size=(trials, N))

                if pool is None:
                    errs = _mp_roundtrip_error_chunk(alpha_samples, dps)
                else:
                    n_chunks = min(trials, jobs * 4)
                    chunks = np.array_split(alpha_samples, n_chunks)
                    parts = pool.map(_mp_roundtrip_error_chunk, chunks, repeat(dps))
                    errs = np.concatenate(list(parts))

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

    if args.jobs is not None:
        jobs = args.jobs
    elif quick:
        jobs = 1
    else:
        jobs = DEFAULT_JOBS

    t0 = time.time()
    failrate = failure_rate_scan(failrate_n_values, failrate_trials, rng)
    t1 = time.time()
    print(f"Panel A (failure rate) done in {t1 - t0:.1f}s")

    precision = precision_scan(
        precision_n_values, precision_trials, hero_n, hero_trials, rng, jobs=jobs
    )
    t2 = time.time()
    print(
        f"Panel B (mp precision, incl. N={hero_n} hero, jobs={jobs}) "
        f"done in {t2 - t1:.1f}s"
    )

    make_figure(
        failrate, failrate_n_values,
        precision, precision_n_values, hero_n,
        FIGDIR / "fig_3_roundtrip",
    )
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
