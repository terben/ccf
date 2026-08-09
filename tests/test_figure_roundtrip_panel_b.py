"""Regression tests for figure_roundtrip.py's parallel Panel B.

paper_plot_scripts/figure_roundtrip.py runs as a standalone script (its
own directory goes on sys.path for `from style import aa_plot`, matching
the other figure scripts), so it is imported here the same way rather
than as a package submodule.
"""

import sys
from pathlib import Path

import numpy as np

PAPER_PLOT_SCRIPTS = Path(__file__).resolve().parent.parent / "paper_plot_scripts"
if str(PAPER_PLOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOT_SCRIPTS))

import figure_roundtrip as fr  # noqa: E402


def test_batched_rng_draws_match_per_trial_draws():
    # precision_scan draws rng.uniform(..., size=(trials, N)) once instead
    # of `trials` calls of size (N,); confirm this yields the exact same
    # underlying draw sequence for numpy's default Generator, so the
    # Monte Carlo sample itself is unchanged by chunking for workers.
    N, trials, bound = 5, 23, 0.9

    rng = np.random.default_rng(7)
    per_trial = np.stack([rng.uniform(-bound, bound, size=N) for _ in range(trials)])

    rng = np.random.default_rng(7)
    batched = rng.uniform(-bound, bound, size=(trials, N))

    np.testing.assert_array_equal(per_trial, batched)


def test_worker_chunk_matches_per_trial_reference():
    N, trials, dps = 6, 5, 30
    rng = np.random.default_rng(3)
    alpha_chunk = rng.uniform(-0.9, 0.9, size=(trials, N))

    chunked = fr._mp_roundtrip_error_chunk(alpha_chunk, dps)
    reference = np.array([fr.mp_roundtrip_error(a, dps) for a in alpha_chunk])

    np.testing.assert_array_equal(chunked, reference)


def test_precision_scan_serial_and_parallel_are_numerically_identical():
    n_values = (8,)
    trials_by_n = {8: 6, 16: 4}
    hero_n, hero_trials = 16, 4

    serial = fr.precision_scan(
        n_values, trials_by_n, hero_n, hero_trials,
        np.random.default_rng(42), jobs=1,
    )
    parallel = fr.precision_scan(
        n_values, trials_by_n, hero_n, hero_trials,
        np.random.default_rng(42), jobs=2,
    )

    assert serial == parallel
