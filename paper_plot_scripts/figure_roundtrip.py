"""
Figure 2 (\\label{fig:roundtrip} in CCF.tex) -- two self-contained panels
on the numerical behaviour of the Levinson-Durbin recursion.

Rather than mixing float64 and arbitrary-precision results into a single
"roundtrip error" curve (which would conflate genuine measurements with an
artificial failure sentinel), each panel makes exactly one factual claim,
backed by one arithmetic model:

  Panel A (float64 only):
      Fraction of trials on which the recovered sequence leaves the
      admissible interval (-1, 1), i.e. schurcorr.pacf() raises
      SingularToeplitzError or ValueError, vs. N, for alpha drawn
      independently and uniformly from (-b, b)^N, b in {0.9, 0.95}.
      Shows precisely where float64 stops being usable for this stress
      test -- a sharp transition around N~40-70, not a gradual one.

  Panel B (arbitrary precision only, mpmath):
      log10(max_n |alpha_n - alpha_n'|) vs. N, median and worst-trial
      band, same alpha bounds. No float64 curve here -- this panel's
      only claim is "arbitrary precision does not break", up to and
      including a N=1024 "hero" point with a handful of trials.

Panel C (O(N^2) timing) was dropped after investigation. A log-log scan
of `pacf` up to N=8192 (small alpha, purely to probe the complexity
question) gave a *local* slope climbing only from ~1.0 to ~1.4 -- nowhere
near the theoretical O(N^2) reference of slope 2 even at N=8192. Root
cause: the recursion is a Python-level loop calling NumPy once per step;
at these N, the per-call NumPy/Python dispatch overhead (microseconds)
exceeds the actual O(n) floating-point work per step (nanoseconds), so an
O(N) overhead term dominates the O(N^2) arithmetic term across the entire
practically testable range. This is a genuine property of this
implementation, not an artifact of the alpha range chosen for A/B. Rather
than show a plot that would need a lengthy methodological caveat to not
be misread as evidence against O(N^2) complexity, this panel is left
out; the observation itself belongs in the paper text, not in a figure
whose whole point is to be self-explanatory.

Framing note for the text: panel A's stress test draws alpha
independently at *every* lag from a wide, fixed interval -- exactly the
regime in which the Jacobian det(dr/dalpha) = prod(sigma_n^2) vanishes
fastest (generically, for "most" such draws). Realistic correlation
functions have partial autocorrelations that decay with lag, not i.i.d.
draws across all lags at once, and are expected to stay far more benign
in float64 at the same N. Panel A is intentionally an adversarial stress
test, not a claim about typical astrophysical use.

Failure definition (panel A): a trial counts as failed if and only if
pacf() raises SingularToeplitzError (the recursion reaches
sigma_n^2 = 0) or ValueError (the computed alpha_n overshoots the
admissible interval outright, |alpha_n| > 1 by more than a numerical
tolerance). Both checks fire immediately after the offending alpha_n is
computed, for every n including the last one, so -- unlike the earlier
ad-hoc `pacf_toeplitz.levinson_durbin` this script used to call, whose
single check ran only at the *top* of the next loop iteration and so
missed a small fraction of failures landing on the very last index --
this count has no such blind spot. See schurcorr/levinson.py,
_LevinsonState.from_correlations, for the two checks.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

import schurcorr as sc
from schurcorr.plotting import aa_plot

# Configure a full-width, two-column A&A figure.
# The original 11 x 5.5 inch layout had a height-to-width ratio of 0.5.
aa_plot(
    column="double",
    height_ratio=0.25,
    fontsize=8.0,
    use_tex=True,
)

SEED = 20260714
rng = np.random.default_rng(SEED)

FIGDIR = Path(__file__).resolve().parent / "../figs"
FIGDIR.mkdir(exist_ok=True)

BOUNDS = (0.9, 0.95)
BOUND_COLORS = {0.9: "tab:blue", 0.95: "tab:red"}

# --- Panel A: failure rate, float64, dense N grid (cheap: failed
# trials terminate early) -----------------------------------------------
FAILRATE_N_VALUES = (
    8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 112, 128,
)
FAILRATE_TRIALS = 3_000

# --- Panel B: arbitrary-precision roundtrip, physically-motivated N
# plus one "hero" point at N=1024 -----------------------------------
PRECISION_N_VALUES = (16, 64, 256)
PRECISION_TRIALS = {16: 5_000, 64: 2_500, 256: 250}
HERO_N = 1024
HERO_TRIALS = 15




def failure_rate_scan() -> dict[tuple[float, int], float]:
    """Panel A data: fraction of admissibility failures (SingularToeplitzError
    or ValueError from schurcorr.pacf) over FAILRATE_TRIALS draws, for each
    (bound, N)."""
    results = {}
    for bound in BOUNDS:
        for N in FAILRATE_N_VALUES:
            fails = 0
            for _ in range(FAILRATE_TRIALS):
                alpha = rng.uniform(-bound, bound, size=N)
                try:
                    r = sc.from_pacf(alpha)
                    sc.pacf(r)
                except (sc.SingularToeplitzError, ValueError):
                    fails += 1
            results[(bound, N)] = fails / FAILRATE_TRIALS
    return results


def mp_roundtrip_error(alpha: np.ndarray, dps: int) -> float:
    """max_n |alpha_n - alpha_n'| at arbitrary precision, compared in
    the mp domain (not after downcasting to float64 -- the whole point
    of the mpmath backend is to carry the comparison at working
    precision)."""
    r_mp = sc.from_pacf(alpha, backend="mpmath", dps=dps)
    rec_mp = sc.pacf(r_mp, backend="mpmath", dps=dps)
    err = max(abs(mp.mpf(a) - b) for a, b in zip(alpha, rec_mp))
    return float(err)


def precision_scan() -> dict[tuple[float, int], dict[str, float]]:
    """Panel B data: median / worst arbitrary-precision roundtrip error
    for each (bound, N), including the N=1024 hero point."""
    results = {}
    for bound in BOUNDS:
        for N in (*PRECISION_N_VALUES, HERO_N):
            trials = PRECISION_TRIALS.get(N, HERO_TRIALS)
            dps = sc.recommended_dps(N)
            errs = np.empty(trials)
            for i in range(trials):
                alpha = rng.uniform(-bound, bound, size=N)
                errs[i] = mp_roundtrip_error(alpha, dps)
            results[(bound, N)] = dict(
                median=float(np.median(errs)), worst=float(np.max(errs))
            )
    return results


def make_figure(failrate: dict, precision: dict, path_stub: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2)

    # Styling common to both axes (match fig_1 vibe)
    for ax in (ax_a, ax_b):
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xlabel(ax.get_xlabel(), labelpad=6)
        ax.set_ylabel(ax.get_ylabel(), labelpad=8)

    # Panel A
    for bound in BOUNDS:
        c = BOUND_COLORS[bound]
        y = [100 * failrate[(bound, N)] for N in FAILRATE_N_VALUES]
        ax_a.plot(
            FAILRATE_N_VALUES, y,
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
    precision_Ns = (*PRECISION_N_VALUES, HERO_N)
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
    ax_b.axvline(HERO_N, color="gray", linestyle=":", linewidth=1.3, alpha=0.9)
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
    t0 = time.time()
    failrate = failure_rate_scan()
    t1 = time.time()
    print(f"Panel A (failure rate) done in {t1 - t0:.1f}s")

    precision = precision_scan()
    t2 = time.time()
    print(f"Panel B (mp precision, incl. N={HERO_N} hero) done in {t2 - t1:.1f}s")

    make_figure(failrate, precision, FIGDIR / "fig_3_roundtrip")
    print(f"Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
