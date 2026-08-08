"""Regression tests for figure_roundtrip.py's batched Panel A.

paper_plot_scripts/figure_roundtrip.py runs as a standalone script (its
own directory goes on sys.path for `from style import aa_plot`, matching
the other figure scripts), so it is imported here the same way rather
than as a package submodule.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import schurcorr as sc

PAPER_PLOT_SCRIPTS = Path(__file__).resolve().parent.parent / "paper_plot_scripts"
if str(PAPER_PLOT_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOT_SCRIPTS))

import figure_roundtrip as fr  # noqa: E402


def _scalar_failure_mask(r: np.ndarray) -> np.ndarray:
    """Per-row failure mask via sc.pacf's own per-row exceptions."""
    failed = np.zeros(r.shape[0], dtype=bool)
    for i, row in enumerate(r):
        try:
            sc.pacf(row)
        except (sc.SingularToeplitzError, ValueError):
            failed[i] = True
    return failed


@pytest.mark.parametrize("bound,N", [(0.9, 32), (0.9, 64), (0.95, 64), (0.95, 96)])
def test_batch_diagnostic_matches_scalar_pacf(bound, N):
    rng = np.random.default_rng(0)
    alpha = rng.uniform(-bound, bound, size=(200, N))
    r = sc.from_pacf(alpha)

    batch_mask = fr._roundtrip_failure_mask(r)
    scalar_mask = _scalar_failure_mask(r)

    np.testing.assert_array_equal(batch_mask, scalar_mask)


@pytest.mark.parametrize("chunk_size", [1, 3, 11, 200])
def test_chunk_size_does_not_change_failure_count(chunk_size):
    rng = np.random.default_rng(1)
    alpha = rng.uniform(-0.9, 0.9, size=(200, 64))
    r = sc.from_pacf(alpha)

    full_mask = fr._roundtrip_failure_mask(r)

    chunked_fails = 0
    for start in range(0, r.shape[0], chunk_size):
        chunk = r[start : start + chunk_size]
        chunked_fails += int(np.count_nonzero(fr._roundtrip_failure_mask(chunk)))

    assert chunked_fails == int(np.count_nonzero(full_mask))


def test_failure_rate_scan_matches_per_trial_reference_loop():
    n_values = (32, 64)
    trials = 150

    def reference_scan(rng):
        results = {}
        for bound in fr.BOUNDS:
            for N in n_values:
                fails = 0
                for _ in range(trials):
                    alpha = rng.uniform(-bound, bound, size=N)
                    try:
                        r = sc.from_pacf(alpha)
                        sc.pacf(r)
                    except (sc.SingularToeplitzError, ValueError):
                        fails += 1
                results[(bound, N)] = fails / trials
        return results

    reference = reference_scan(np.random.default_rng(42))
    batched = fr.failure_rate_scan(n_values, trials, np.random.default_rng(42))

    assert reference == batched


def test_chunked_rng_draws_match_per_trial_draws():
    # The Panel A rewrite groups trial draws into rng.uniform(...,
    # size=(m, N)) calls instead of m calls of size (N,); confirm this
    # yields the exact same underlying draw sequence for numpy's default
    # Generator, so the Monte Carlo sample itself is unchanged.
    N, trials, chunk = 5, 23, 4

    rng = np.random.default_rng(7)
    per_trial = np.stack(
        [rng.uniform(-0.9, 0.9, size=N) for _ in range(trials)]
    )

    rng = np.random.default_rng(7)
    chunks = []
    for start in range(0, trials, chunk):
        m = min(chunk, trials - start)
        chunks.append(rng.uniform(-0.9, 0.9, size=(m, N)))
    chunked = np.concatenate(chunks, axis=0)

    np.testing.assert_array_equal(per_trial, chunked)
