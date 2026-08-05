"""
Characterization tests for CLAUDE.md Phase 1.

These tests add two things that ``tests/test_schurcorr.py`` does not yet
cover (see ``docs/boundary_semantics.md``, Phase 1b):

- independent symbolic ground truth *at* the degenerate boundary, not just
  at interior points (section A below), and
- regression coverage at dimensions the published figures actually use
  (section B below).

Deliberately not added here: further tests of the ``at_boundary=``/
``backend=`` mode-string combinatorics on ``pacf``/``from_pacf``.
``docs/boundary_semantics.md`` recommends replacing both with dedicated
functions, so ``tests/test_schurcorr.py``'s existing ~23 tests of that
surface characterize an API scheduled for replacement, not a permanent
contract to grow.
"""

import numpy as np
import pytest

import schurcorr as sc
from schurcorr import symbolic

# --- A. symbolic ground truth at the boundary -------------------------------
#
# For a fixed interior prefix r_1, ..., r_{order-1} (an exponential sequence
# rho, rho^2, ..., rho^{order-1}, which is admissible at every order since it
# is the autocorrelation of a stationary AR(1) process), the admissible
# interval for r_order is [r_lower, r_upper] per
# schurcorr.symbolic.admissible_bounds_symbolic. Setting r_order to exactly
# that boundary forces alpha_order = +-1, independently of the numerical
# recursion under test: the boundary r-value and the expected alpha-prefix
# both come from the symbolic module, not from schurcorr.levinson.


def _exponential_boundary_r(order: int, rho: float, sign: int) -> np.ndarray:
    """
    Build r_1, ..., r_order such that r_1, ..., r_{order-1} follow the
    admissible exponential sequence rho**k and r_order sits on the symbolic
    admissible boundary (alpha_order = sign), nudged a further ``sign *
    1e-13`` past it.

    The nudge is necessary because evaluating the symbolic boundary
    expression in float64 lands on either side of the true boundary by a
    few ULP, unpredictably: without it, ``abs(alpha_order)`` sometimes comes
    out as e.g. ``0.999999999999998`` and the recursion's
    ``abs(alpha_n) >= 1.0`` boundary check never fires. ``1e-13`` is chosen
    to comfortably clear that float64 rounding noise while staying far
    inside the recursion's own admissibility tolerance
    (``schurcorr.levinson._TOL = 1e-12``), so the nudge reliably lands in
    the "boundary, clamped to +-1" branch rather than the "inadmissible,
    raises" branch.
    """
    prefix = [rho**k for k in range(1, order)]

    r_lower_sym, r_upper_sym = symbolic.admissible_bounds_symbolic(order)
    boundary_expr = r_upper_sym if sign > 0 else r_lower_sym

    if order > 1:
        r_syms = symbolic.correlation_symbols(order - 1)
        r_last = float(boundary_expr.subs(dict(zip(r_syms, prefix))))
    else:
        r_last = float(boundary_expr)

    r_last += sign * 1e-13

    return np.array(prefix + [r_last])


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("sign", [1, -1])
def test_boundary_alpha_prefix_matches_symbolic(order, sign):
    r = _exponential_boundary_r(order, rho=0.3, sign=sign)

    alpha = sc.pacf_prefix(r).alpha

    assert alpha.size == order
    assert alpha[-1] == pytest.approx(sign, abs=1e-8)

    if order > 1:
        alpha_syms = symbolic.pacf_sequence_symbolic(order - 1)
        r_syms = symbolic.correlation_symbols(order - 1)
        subs = dict(zip(r_syms, r[: order - 1]))
        expected_prefix = [float(a.subs(subs)) for a in alpha_syms]
        np.testing.assert_allclose(alpha[:-1], expected_prefix, atol=1e-8)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_boundary_admissible_bounds_match_symbolic_at_every_order(order):
    r = _exponential_boundary_r(order, rho=0.3, sign=1)

    r_lower, r_upper = sc.admissible_bounds(r)

    for n in range(1, order + 1):
        lo_sym, hi_sym = symbolic.admissible_bounds_symbolic(n)

        if n > 1:
            r_syms = symbolic.correlation_symbols(n - 1)
            subs = dict(zip(r_syms, r[: n - 1]))
            lo_expected = float(lo_sym.subs(subs))
            hi_expected = float(hi_sym.subs(subs))
        else:
            lo_expected = float(lo_sym)
            hi_expected = float(hi_sym)

        assert r_lower[n - 1] == pytest.approx(lo_expected, abs=1e-8)
        assert r_upper[n - 1] == pytest.approx(hi_expected, abs=1e-8)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_boundary_sequence_is_admissible(order):
    r = _exponential_boundary_r(order, rho=0.3, sign=1)

    assert sc.check_admissibility(r)


# --- B. figure-relevant dimensions ------------------------------------------
#
# Regression coverage at dimensions the published figures actually use.
#
# paper_plot_scripts/figure_roundtrip.py's docstring mentions a float64
# complexity scan up to N=8192, but that panel ("Panel C") was explicitly
# dropped from the script after investigation (see the module docstring,
# "Panel C (O(N^2) timing) was dropped after investigation") -- N=8192 is
# not a dimension the current figure exercises, so it is not tested here.
#
# More importantly, figure_roundtrip.py's own Panel A *measures* the float64
# failure rate at BOUNDS = (0.9, 0.95) and treats a nonzero failure rate as
# the expected, plotted result -- reproduced independently below: even a far
# milder bound=0.5 fails about half the time by N=256, because the
# innovation variance sigma_n^2 = prod(1 - alpha_i^2) shrinks geometrically
# in N regardless of how small a fixed alpha bound is, and dividing by a
# tiny sigma_n^2 amplifies float64 roundoff. So "pacf/from_pacf roundtrips
# correctly at large N in backend='float64'" is not a claim this package or
# paper makes -- asserting it here would characterize an assumption, not
# paper-required behavior (CLAUDE.md Phase 1: "do not encode accidental
# complexity as a permanent requirement"). The paper's actual claim at large
# N (figure_roundtrip.py Panel B, its N=1024 "hero" point) is specifically
# that backend='mpmath' with recommended_dps recovers a reliable roundtrip
# where float64 would not -- that is what is tested below.


@pytest.mark.parametrize("bound", [0.9, 0.95])
def test_roundtrip_at_figure_roundtrip_hero_dimension_mpmath(bound):
    n_lags = 1024
    rng = np.random.default_rng(1)
    alpha = rng.uniform(-bound, bound, size=n_lags)

    dps = sc.recommended_dps(n_lags)
    r_mp = sc.from_pacf_mp(alpha, dps=dps)
    alpha_back_mp = sc.pacf_mp(r_mp, dps=dps)

    alpha_back = np.array([float(a) for a in alpha_back_mp])
    np.testing.assert_allclose(alpha_back, alpha, rtol=1e-9, atol=1e-9)


def test_roundtrip_at_gaussianization_figure_batch_dimensions():
    n_lags = 15
    n_samples = 200

    rng = np.random.default_rng(2)
    alpha = rng.uniform(-0.5, 0.5, size=(n_samples, n_lags))

    r = sc.from_pacf(alpha)
    alpha_back = sc.pacf(r)

    np.testing.assert_allclose(alpha_back, alpha, rtol=1e-9, atol=1e-9)
