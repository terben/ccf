import mpmath as mp
import numpy as np
import scipy.linalg as sla
import sympy as sp
import pytest
import ccf

from ccf.symbolic import (
    admissible_bounds_symbolic,
    pacf_symbolic,
    toeplitz_matrix,
    verify_x_equals_alpha,
)


def test_pacf_roundtrip():
    alpha = np.array([0.2, -0.3, 0.4])

    r = ccf.from_pacf(alpha)
    alpha_back = ccf.pacf(r)

    np.testing.assert_allclose(alpha_back, alpha, rtol=1e-12, atol=1e-12)


def test_pacf_default_raises_at_boundary():
    r = np.array([1.0, 0.5, 0.3])

    with pytest.raises(ccf.SingularToeplitzError):
        ccf.pacf(r)


def test_pacf_prefix_returns_truncated_alpha_at_boundary():
    r = np.array([1.0, 0.5, 0.3])

    result = ccf.pacf_prefix(r)

    assert result.reached_boundary
    np.testing.assert_allclose(result.alpha, [1.0])


def test_pacf_and_pacf_prefix_agree_at_interior_points():
    alpha = np.array([0.2, -0.3, 0.4])
    r = ccf.from_pacf(alpha)

    result = ccf.pacf_prefix(r)

    assert not result.reached_boundary
    np.testing.assert_allclose(ccf.pacf(r), alpha, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.alpha, alpha, rtol=1e-12, atol=1e-12)


# --- pacf_status ------------------------------------------------------------


def test_pacf_status_interior_1d():
    alpha = np.array([0.2, -0.3, 0.4])
    r = ccf.from_pacf(alpha)

    status = ccf.pacf_status(r)

    assert status.interior is True
    assert status.boundary is False
    assert status.invalid is False
    assert status.order == len(alpha)


def test_pacf_status_boundary_1d():
    r = np.array([1.0, 0.5, 0.3])

    status = ccf.pacf_status(r)

    assert status.boundary is True
    assert status.interior is False
    assert status.invalid is False
    assert status.order == ccf.pacf_prefix(r).order


def test_pacf_status_invalid_1d():
    r = np.array([0.9, 0.95, 0.99])

    status = ccf.pacf_status(r)

    assert status.invalid is True
    assert status.interior is False
    assert status.boundary is False
    assert status.order == 3


def test_pacf_status_does_not_raise_where_pacf_raises():
    r_mixed = np.array([
        [0.2, 0.1, 0.05],
        [1.0, 0.5, 0.3],
        [0.9, 0.95, 0.99],
    ])

    with pytest.raises(ccf.SingularToeplitzError):
        ccf.pacf(r_mixed)

    status = ccf.pacf_status(r_mixed)
    np.testing.assert_array_equal(status.interior, [True, False, False])
    np.testing.assert_array_equal(status.boundary, [False, True, False])
    np.testing.assert_array_equal(status.invalid, [False, False, True])


def test_pacf_status_batch_mixed_classification():
    interior_row = ccf.from_pacf(np.array([0.2, -0.3, 0.4]))
    boundary_row = np.array([1.0, 0.5, 0.3])
    invalid_row = np.array([0.9, 0.95, 0.99])
    r_batch = np.stack([interior_row, boundary_row, invalid_row])

    status = ccf.pacf_status(r_batch)

    np.testing.assert_array_equal(status.interior, [True, False, False])
    np.testing.assert_array_equal(status.boundary, [False, True, False])
    np.testing.assert_array_equal(status.invalid, [False, False, True])
    np.testing.assert_array_equal(status.order, [3, 1, 3])


def test_pacf_status_categories_are_mutually_exclusive_and_exhaustive():
    rng = np.random.default_rng(0)
    alpha = rng.uniform(-0.95, 0.95, size=(50, 20))
    r = ccf.from_pacf(alpha)

    status = ccf.pacf_status(r)

    membership_count = (
        status.interior.astype(int)
        + status.boundary.astype(int)
        + status.invalid.astype(int)
    )
    np.testing.assert_array_equal(membership_count, np.ones_like(membership_count))


def test_pacf_status_matches_pacf_per_row():
    interior_row = ccf.from_pacf(np.array([0.2, -0.3, 0.4]))
    boundary_row = np.array([1.0, 0.5, 0.3])
    invalid_row = np.array([0.9, 0.95, 0.99])
    r_batch = np.stack([interior_row, boundary_row, invalid_row])

    status = ccf.pacf_status(r_batch)

    for i, row in enumerate(r_batch):
        if status.interior[i]:
            ccf.pacf(row)
        elif status.boundary[i]:
            with pytest.raises(ccf.SingularToeplitzError):
                ccf.pacf(row)
        else:
            assert status.invalid[i]
            with pytest.raises(ValueError):
                ccf.pacf(row)


def test_pacf_status_order_matches_pacf_prefix_at_boundary():
    r_mixed = np.array([
        [0.2, 0.1, 0.05, 0.02],
        [1.0, 1.0, 1.0, 1.0],
        [0.3, -0.2, 0.1, 0.05],
    ])

    status = ccf.pacf_status(r_mixed)

    assert status.boundary[1]
    assert status.order[1] == ccf.pacf_prefix(r_mixed[1]).order


def test_fisher_roundtrip():
    alpha = np.array([0.2, -0.3, 0.4])

    y = ccf.fisher(alpha)
    alpha_back = ccf.inverse_fisher(y)

    np.testing.assert_allclose(alpha_back, alpha, rtol=1e-12, atol=1e-12)


# --- extend_at_boundary ----------------------------------------------------


def _order3_boundary_sequence():
    """
    Construct r_1, r_2, r_3 with A_1, A_2, A_3 positive definite and
    A_4 singular, i.e. the boundary is first reached at m = 4
    (alpha_3 = +1).
    """
    alpha12 = np.array([0.3, -0.2])
    r12 = ccf.from_pacf(alpha12)

    _, r_upper = ccf.admissible_bounds(np.append(r12, r12[-1]))

    r3 = r_upper[2]
    return np.append(r12, r3)


def test_extend_at_boundary_order1():
    r_ext = ccf.extend_at_boundary(np.array([1.0]), n_extra=3)
    np.testing.assert_allclose(r_ext, [1.0, 1.0, 1.0, 1.0])


def test_extend_at_boundary_null_vector_matches_scipy():
    r = _order3_boundary_sequence()

    r_ext = ccf.extend_at_boundary(r, n_extra=3)

    # A_4, built from r_1, r_2, r_3, must be singular, and the
    # phi-derived null vector used internally must be proportional
    # (up to sign) to the null space computed directly by scipy.
    a4 = sla.toeplitz([1.0, r[0], r[1], r[2]])
    eigenvalues = np.linalg.eigvalsh(a4)
    assert np.min(np.abs(eigenvalues)) < 1e-8

    null_space = sla.null_space(a4, rcond=1e-8)
    assert null_space.shape == (4, 1)
    w = null_space[:, 0]

    alpha_ext = ccf.pacf_prefix(r).alpha

    phi = -w[1:] / w[0]
    predicted_r4 = float(np.dot(phi, r[::-1]))
    np.testing.assert_allclose(predicted_r4, r_ext[3], atol=1e-8)
    assert alpha_ext[2] == 1.0


def test_extend_at_boundary_recurrence_holds_for_generated_values():
    r = _order3_boundary_sequence()

    r_ext = ccf.extend_at_boundary(r, n_extra=4)

    # every extended matrix A_(4+s) must remain singular
    for k in range(4, len(r_ext) + 1):
        a_k = sla.toeplitz(np.concatenate(([1.0], r_ext[: k - 1])))
        eigenvalues = np.linalg.eigvalsh(a_k)
        assert np.min(np.abs(eigenvalues)) < 1e-6


def test_extend_at_boundary_validates_consistent_prefix():
    r = _order3_boundary_sequence()

    r_ext = ccf.extend_at_boundary(r, n_extra=2)
    r_ext_again = ccf.extend_at_boundary(r_ext, n_extra=2)

    np.testing.assert_allclose(r_ext_again, r_ext)


def test_extend_at_boundary_rejects_inconsistent_prefix():
    r = _order3_boundary_sequence()
    r_bad = np.append(r, 999.0)

    with pytest.raises(ValueError, match="inconsistent"):
        ccf.extend_at_boundary(r_bad, n_extra=1)


def test_extend_at_boundary_n_extra_too_small_for_existing_excess():
    r = _order3_boundary_sequence()
    r_with_excess = np.append(r, r[-1])

    with pytest.raises(ValueError, match="n_extra"):
        ccf.extend_at_boundary(r_with_excess, n_extra=0)


def test_extend_at_boundary_negative_n_extra_raises():
    with pytest.raises(ValueError):
        ccf.extend_at_boundary(np.array([1.0]), n_extra=-1)


def test_extend_at_boundary_requires_boundary_reached():
    alpha = np.array([0.2, -0.3, 0.4])
    r = ccf.from_pacf(alpha)

    with pytest.raises(ValueError, match="boundary"):
        ccf.extend_at_boundary(r, n_extra=2)


# --- mpmath backend ---------------------------------------------------------


def test_from_pacf_mpmath_matches_float64():
    alpha = [0.2, -0.3, 0.4]

    r_mp = ccf.from_pacf_mp(alpha)
    r_f64 = ccf.from_pacf(np.array(alpha))

    assert all(isinstance(x, mp.mpf) for x in r_mp)
    np.testing.assert_allclose(
        [float(x) for x in r_mp], r_f64, rtol=1e-12, atol=1e-12
    )


def test_pacf_mpmath_matches_float64():
    alpha = [0.2, -0.3, 0.4]
    r = ccf.from_pacf_mp(alpha)

    alpha_back_mp = ccf.pacf_mp(r)
    alpha_back_f64 = ccf.pacf(np.array([float(x) for x in r]))

    assert all(isinstance(x, mp.mpf) for x in alpha_back_mp)
    np.testing.assert_allclose(
        [float(x) for x in alpha_back_mp],
        alpha_back_f64,
        rtol=1e-12,
        atol=1e-12,
    )


def test_pacf_mpmath_roundtrip_high_order():
    rng = np.random.default_rng(7)
    alpha = rng.uniform(-0.9, 0.9, size=60).tolist()
    dps = ccf.recommended_dps(len(alpha), safety_margin=30)

    r = ccf.from_pacf_mp(alpha, dps=dps)
    alpha_back = ccf.pacf_mp(r, dps=dps)

    for a, a_back in zip(alpha, alpha_back):
        assert abs(mp.mpf(a) - a_back) < mp.mpf("1e-12")


def test_recommended_dps_matches_formula():
    assert ccf.recommended_dps(100) == 69
    assert ccf.recommended_dps(100, target_exponent=-13, safety_margin=11) == 69


def test_pacf_mpmath_default_raises_at_boundary():
    r = [1.0, 0.5, 0.3]

    with pytest.raises(ccf.SingularToeplitzError):
        ccf.pacf_mp(r)


def test_pacf_mpmath_warn_returns_truncated():
    r = [1.0, 0.5, 0.3]

    with pytest.warns(RuntimeWarning):
        alpha = ccf.pacf_mp(r, at_boundary="warn")

    assert len(alpha) == 1
    assert alpha[0] == mp.mpf(1)


def test_pacf_mpmath_extend_not_supported():
    r = [1.0, 0.5, 0.3]

    with pytest.raises(ValueError, match="extend"):
        ccf.pacf_mp(r, at_boundary="extend")


def test_pacf_mpmath_rejects_2d_input():
    r = np.array([[0.2, 0.1], [0.3, -0.1]])

    with pytest.raises(ValueError):
        ccf.pacf_mp(r)


def test_from_pacf_mpmath_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        ccf.from_pacf_mp([0.2, 1.5])


def test_pacf_mpmath_custom_dps():
    alpha_in = [0.2, -0.3, 0.4]
    r = ccf.from_pacf_mp(alpha_in, dps=50)
    alpha = ccf.pacf_mp(r, dps=50)

    assert mp.mp.dps != 50  # workdps must not leak into the global context
    np.testing.assert_allclose(
        [float(x) for x in alpha], alpha_in, rtol=1e-12, atol=1e-12
    )


# --- admissible_volume / log_admissible_volume ------------------------------


def _volume_symbolic(N):
    prod = sp.Integer(1)
    for jj in range(1, N):
        prod *= sp.sqrt(sp.pi) * sp.factorial(jj) / sp.gamma(jj + sp.Rational(3, 2))
    return sp.nsimplify(2 * prod)


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6])
def test_admissible_volume_matches_closed_form(N):
    expected = float(_volume_symbolic(N))
    np.testing.assert_allclose(ccf.admissible_volume(N), expected, rtol=1e-12)


def test_admissible_volume_order1_is_hypercube_width():
    # N = 1: r_1 = alpha_1 in (-1, 1) directly, volume = 2.
    assert ccf.admissible_volume(1) == 2.0


@pytest.mark.parametrize("N", [1, 2, 5, 10, 50])
def test_log_admissible_volume_matches_admissible_volume(N):
    np.testing.assert_allclose(
        ccf.log_admissible_volume(N),
        np.log(ccf.admissible_volume(N)),
        rtol=1e-10,
    )


def test_admissible_volume_underflows_where_log_does_not():
    assert ccf.admissible_volume(2000) == 0.0
    assert np.isfinite(ccf.log_admissible_volume(2000))
    assert ccf.log_admissible_volume(2000) < 0.0


@pytest.mark.parametrize("N", [0, -1, 2.5])
def test_admissible_volume_rejects_invalid_N(N):
    with pytest.raises(ValueError):
        ccf.admissible_volume(N)

    with pytest.raises(ValueError):
        ccf.log_admissible_volume(N)


@pytest.mark.parametrize("N", [2, 3, 4])
def test_sh_normalized_volume_matches_monte_carlo_admissible_fraction(N):
    # Independent cross-check of V_N^SH = V_N / 2^N (Eq. volume_SH):
    # it is the probability that a correlation sequence drawn
    # uniformly from [-1, 1]^N is admissible.
    rng = np.random.default_rng(20260714)
    trials = 20_000
    samples = rng.uniform(-1.0, 1.0, size=(trials, N))

    admissible_count = sum(
        1 for row in samples if ccf.check_admissibility(row)
    )
    empirical_fraction = admissible_count / trials

    v_sh = ccf.admissible_volume(N) / 2.0**N

    assert abs(empirical_fraction - v_sh) < 0.02


def test_sh_normalized_volume_is_a_probability():
    for N in [1, 2, 5, 10, 30]:
        v_sh = ccf.admissible_volume(N) / 2.0**N
        assert 0.0 < v_sh <= 1.0


# --- Batched (2-D) interface ---------------------------------------------


def _random_batch(rng, n_samples=5, n=6):
    return rng.uniform(-0.9, 0.9, size=(n_samples, n))


def test_batched_from_pacf_matches_looped_1d():
    # The batched path is vectorized across rows (a different
    # summation order than the scalar per-row path's np.dot), so this
    # is a tight-tolerance comparison, not exact equality -- see the
    # Notes in ccf.levinson._pacf_2d_fast.
    rng = np.random.default_rng(0)
    alpha = _random_batch(rng)

    r_batch = ccf.from_pacf(alpha)
    r_loop = np.stack([ccf.from_pacf(row) for row in alpha])

    assert r_batch.shape == alpha.shape
    np.testing.assert_allclose(r_batch, r_loop, rtol=1e-14, atol=1e-14)


def test_batched_pacf_matches_looped_1d():
    # Same summation-order caveat as test_batched_from_pacf_matches_looped_1d.
    rng = np.random.default_rng(1)
    alpha = _random_batch(rng)
    r = ccf.from_pacf(alpha)

    alpha_batch = ccf.pacf(r)
    alpha_loop = np.stack([ccf.pacf(row) for row in r])

    assert alpha_batch.shape == r.shape
    np.testing.assert_allclose(alpha_batch, alpha_loop, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(alpha_batch, alpha, rtol=1e-12, atol=1e-12)


def test_batched_fisher_and_inverse_fisher_match_looped_1d():
    rng = np.random.default_rng(2)
    alpha = _random_batch(rng)

    y_batch = ccf.fisher(alpha)
    y_loop = np.stack([ccf.fisher(row) for row in alpha])
    np.testing.assert_array_equal(y_batch, y_loop)

    alpha_back_batch = ccf.inverse_fisher(y_batch)
    alpha_back_loop = np.stack([ccf.inverse_fisher(row) for row in y_batch])
    np.testing.assert_array_equal(alpha_back_batch, alpha_back_loop)


def test_batched_innovation_variances_matches_looped_1d():
    rng = np.random.default_rng(3)
    alpha = _random_batch(rng)

    sigma2_batch = ccf.innovation_variances(alpha)
    sigma2_loop = np.stack([ccf.innovation_variances(row) for row in alpha])

    assert sigma2_batch.shape == (alpha.shape[0], alpha.shape[1] + 1)
    np.testing.assert_array_equal(sigma2_batch, sigma2_loop)


def test_1d_output_unchanged_by_batched_interface():
    alpha = np.array([0.2, -0.3, 0.4])
    r = ccf.from_pacf(alpha)

    assert r.ndim == 1
    assert ccf.pacf(r).ndim == 1
    assert ccf.fisher(alpha).ndim == 1
    assert ccf.inverse_fisher(ccf.fisher(alpha)).ndim == 1
    assert ccf.innovation_variances(alpha).ndim == 1

    np.testing.assert_allclose(ccf.pacf(r), alpha, rtol=1e-12, atol=1e-12)


def test_batched_pacf_raise_reports_offending_sample():
    r = np.array([
        [0.2, 0.1, 0.05],
        [1.0, 0.5, 0.3],
    ])

    with pytest.raises(ccf.SingularToeplitzError, match="sample 1"):
        ccf.pacf(r)


def test_batched_from_pacf_reports_offending_sample():
    alpha = np.array([
        [0.1, 0.2],
        [0.1, 1.5],
    ])

    with pytest.raises(ValueError, match="sample 1"):
        ccf.from_pacf(alpha)


def test_pacf_invalid_ndim_raises_value_error():
    with pytest.raises(ValueError):
        ccf.pacf(np.zeros((2, 2, 2)))


# --- Batch vectorization: shared kernel behavior --------------------------


def test_pacf_batch_size_one_matches_1d():
    rng = np.random.default_rng(11)
    alpha = rng.uniform(-0.7, 0.7, size=10)
    r = ccf.from_pacf(alpha)

    assert r.shape == (10,)
    np.testing.assert_allclose(ccf.pacf(r[None, :])[0], ccf.pacf(r))


def test_from_pacf_batch_size_one_matches_1d():
    rng = np.random.default_rng(11)
    alpha = rng.uniform(-0.7, 0.7, size=10)

    np.testing.assert_allclose(
        ccf.from_pacf(alpha[None, :])[0], ccf.from_pacf(alpha)
    )


def test_batched_pacf_and_from_pacf_large_scale_match_looped_1d():
    # A larger, well-conditioned batch, comparing the vectorized batch
    # kernel against the per-row scalar loop. Deliberately away from the
    # numerically fragile bound~1 regime (see figure_roundtrip.py /
    # Panel A): near that regime, the recursion itself amplifies even
    # ULP-level perturbations, so batch-vs-loop agreement legitimately
    # degrades regardless of implementation -- not exercised here.
    rng = np.random.default_rng(12)
    alpha = rng.uniform(-0.5, 0.5, size=(2000, 15))

    r_batch = ccf.from_pacf(alpha)
    r_loop = np.stack([ccf.from_pacf(row) for row in alpha])
    np.testing.assert_allclose(r_batch, r_loop, rtol=1e-11, atol=1e-11)

    alpha_batch = ccf.pacf(r_batch)
    alpha_loop = np.stack([ccf.pacf(row) for row in r_batch])
    np.testing.assert_allclose(alpha_batch, alpha_loop, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(alpha_batch, alpha, rtol=1e-9, atol=1e-9)


def test_batched_pacf_boundary_row_matches_per_row_semantics():
    # A batch where one row hits the boundary: pacf() raises with the
    # offending sample index, and pacf_prefix()/extend_at_boundary()
    # applied to that row directly still recover the boundary-inclusive
    # prefix / forced continuation (pacf_prefix is 1-D only, so there is
    # no batched equivalent to compare here).
    r_mixed = np.array([
        [0.2, 0.1, 0.05, 0.02],
        [1.0, 1.0, 1.0, 1.0],
        [0.3, -0.2, 0.1, 0.05],
    ])

    with pytest.raises(ccf.SingularToeplitzError, match="sample 1"):
        ccf.pacf(r_mixed)

    result = ccf.pacf_prefix(r_mixed[1])
    assert result.reached_boundary
    np.testing.assert_allclose(result.alpha, [1.0])

    r_ext = ccf.extend_at_boundary(r_mixed[1], n_extra=4)
    assert np.all(r_ext == 1.0)


def test_batched_pacf_mixed_invalid_row_reports_sample_index():
    r_mixed = np.array([
        [0.2, 0.1, 0.05],
        [0.9, 0.95, 0.99],
        [0.3, -0.2, 0.1],
    ])

    with pytest.raises(ValueError, match="sample 1"):
        ccf.pacf(r_mixed)


def test_batched_pacf_reports_smallest_index_among_multiple_bad_rows():
    # A boundary row before an invalid row: the smallest row index wins,
    # with that row's own error type, regardless of which order each
    # row's problem occurs at internally.
    r_mixed = np.array([
        [0.2, 0.1, 0.05],
        [1.0, 0.5, 0.3],
        [0.9, 0.95, 0.99],
    ])

    with pytest.raises(ccf.SingularToeplitzError, match="sample 1"):
        ccf.pacf(r_mixed)

    # And the reverse ordering: invalid row before boundary row.
    r_mixed_reordered = np.array([
        [0.2, 0.1, 0.05],
        [0.9, 0.95, 0.99],
        [1.0, 0.5, 0.3],
    ])

    with pytest.raises(ValueError, match="sample 1"):
        ccf.pacf(r_mixed_reordered)


def test_from_pacf_batch_never_reaches_boundary_near_alpha_extremes():
    # abs(alpha_n) < 1 strictly guarantees sigma2 stays positive, so the
    # batch kernel for from_pacf has no boundary case, even for alpha
    # very close to +/-1.
    alpha = np.array([
        [0.999999, -0.5, 0.1],
        [-0.999999, 0.999999, -0.1],
    ])

    r_batch = ccf.from_pacf(alpha)
    r_loop = np.stack([ccf.from_pacf(row) for row in alpha])
    np.testing.assert_allclose(r_batch, r_loop, rtol=1e-9, atol=1e-9)


def test_innovation_variances():
    alpha = np.array([0.2, -0.3, 0.4])

    sigma2 = ccf.innovation_variances(alpha)

    expected = np.array([
        1.0,
        1.0 * (1.0 - 0.2**2),
        1.0 * (1.0 - 0.2**2) * (1.0 - (-0.3)**2),
        1.0 * (1.0 - 0.2**2) * (1.0 - (-0.3)**2) * (1.0 - 0.4**2),
    ])

    np.testing.assert_allclose(sigma2, expected, rtol=1e-12, atol=1e-12)


def test_admissible_bounds_contain_input():
    alpha = np.array([0.2, -0.3, 0.4])
    r = ccf.from_pacf(alpha)

    r_lower, r_upper = ccf.admissible_bounds(r)

    assert r_lower.shape == (r.size + 1,)
    assert r_upper.shape == (r.size + 1,)
    assert np.all(r >= r_lower[:-1])
    assert np.all(r <= r_upper[:-1])


def test_admissible_bounds_first_interval_is_exactly_pm1():
    for r in (np.array([0.3]), np.array([0.1, 0.2, 0.3])):
        r_lower, r_upper = ccf.admissible_bounds(r)

        assert r_lower[0] == -1.0
        assert r_upper[0] == 1.0


def test_admissible_bounds_empty_input():
    r_lower, r_upper = ccf.admissible_bounds(np.array([]))

    np.testing.assert_allclose(r_lower, [-1.0])
    np.testing.assert_allclose(r_upper, [1.0])


def test_admissible_bounds_next_interval_has_positive_width_interior():
    r = np.array([0.1, 0.2, 0.3])

    r_lower, r_upper = ccf.admissible_bounds(r)

    assert r_lower.shape == (4,)
    assert r_upper.shape == (4,)
    assert r_lower[-1] < r_upper[-1]


def test_admissible_bounds_next_interval_midpoint_gives_alpha_next_zero():
    r = np.array([0.1, 0.2, 0.3, 0.9])

    r_lower, r_upper = ccf.admissible_bounds(r)
    mid = 0.5 * (r_lower[-1] + r_upper[-1])

    alpha_mid = ccf.pacf(np.append(r, mid))

    assert alpha_mid[-1] == pytest.approx(0.0, abs=1e-8)


def test_admissible_bounds_next_interval_affine_alpha_next():
    r = np.array([0.1, 0.2, 0.3, 0.9])
    a = 0.37

    r_lower, r_upper = ccf.admissible_bounds(r)
    mid = 0.5 * (r_lower[-1] + r_upper[-1])
    half_width = 0.5 * (r_upper[-1] - r_lower[-1])
    next_r = mid + a * half_width

    alpha_ext = ccf.pacf(np.append(r, next_r))

    assert alpha_ext[-1] == pytest.approx(a, abs=1e-8)


def test_admissible_bounds_next_interval_user_case():
    r = np.array([0.1, 0.2, 0.3, 0.9])

    r_lower, r_upper = ccf.admissible_bounds(r)

    assert r_lower[-1] < r_upper[-1]

    r_next_ok = 0.5 * (r_lower[-1] + r_upper[-1])
    ccf.pacf(np.append(r, r_next_ok))

    r_next_bad = r_upper[-1] + 0.5
    with pytest.raises(ValueError):
        ccf.pacf(np.append(r, r_next_bad))


def test_admissible_bounds_next_interval_collapses_at_boundary():
    r = np.array([1.0])

    r_lower, r_upper = ccf.admissible_bounds(r)
    extended = ccf.extend_at_boundary(r, n_extra=1)

    assert r_lower.shape == (2,)
    assert r_upper.shape == (2,)
    assert r_lower[-1] == r_upper[-1]
    assert r_lower[-1] == pytest.approx(extended[-1])
    assert r_upper[-1] == pytest.approx(extended[-1])


def test_admissible_bounds_next_interval_collapses_with_supplied_continuation():
    r = ccf.extend_at_boundary(np.array([1.0]), n_extra=3)

    r_lower, r_upper = ccf.admissible_bounds(r)

    assert r_lower.shape == (r.size + 1,)
    assert r_upper.shape == (r.size + 1,)
    assert r_lower[-1] == r_upper[-1]

    extended = ccf.extend_at_boundary(r, n_extra=r.size + 1)
    assert r_lower[-1] == pytest.approx(extended[-1])


def test_sh_coordinates_equal_pacf():
    alpha = np.array([0.2, -0.3, 0.4])
    r = ccf.from_pacf(alpha)

    x = ccf.sh_coordinates(r)

    np.testing.assert_allclose(x, alpha, rtol=1e-12, atol=1e-12)


def test_check_admissibility_true():
    alpha = np.array([0.2, -0.3, 0.4])
    r = ccf.from_pacf(alpha)

    assert ccf.check_admissibility(r)


def test_check_admissibility_false():
    r = np.array([0.9, -0.9])

    assert not ccf.check_admissibility(r)


def test_admissible_bounds_boundary_consistent_tail_collapses_to_point():
    r_lower, r_upper = ccf.admissible_bounds(np.array([1.0, 1.0]))

    # The third entry is the admissible interval for the next, unsupplied
    # coefficient r_3, which also collapses to the forced continuation.
    np.testing.assert_allclose(r_lower, [-1.0, 1.0, 1.0])
    np.testing.assert_allclose(r_upper, [1.0, 1.0, 1.0])


def test_admissible_bounds_boundary_inconsistent_tail_raises():
    with pytest.raises(ValueError, match="inconsistent"):
        ccf.admissible_bounds(np.array([1.0, 0.5]))


def test_check_admissibility_true_at_boundary_order1():
    # r_1 = 1 forces r_2 = 1 too; this is a legitimate degenerate
    # (positive semidefinite but singular) correlation sequence, not
    # an inadmissible one.
    assert ccf.check_admissibility(np.array([1.0, 1.0]))


def test_check_admissibility_false_at_boundary_inconsistent_tail():
    assert not ccf.check_admissibility(np.array([1.0, 0.5]))


def test_check_admissibility_true_at_boundary_higher_order():
    r = _order3_boundary_sequence()
    r_ext = ccf.extend_at_boundary(r, n_extra=3)

    assert ccf.check_admissibility(r_ext)


def test_invalid_alpha_from_pacf():
    alpha = np.array([0.2, 1.0])

    with pytest.raises(ValueError):
        ccf.from_pacf(alpha)


def test_invalid_alpha_fisher():
    alpha = np.array([0.2, 1.0])

    with pytest.raises(ValueError):
        ccf.fisher(alpha)


def test_log_jacobian_matches_jacobian():
    alpha = np.array([0.2, -0.3, 0.4])

    log_j = ccf.log_jacobian(alpha)
    j = ccf.jacobian(alpha)

    np.testing.assert_allclose(np.log(j), log_j, rtol=1e-12, atol=1e-12)


def test_symbolic_toeplitz_matrix():
    r1, r2 = sp.symbols("r1 r2", real=True)

    expected = sp.Matrix(
        [
            [1, r1, r2],
            [r1, 1, r1],
            [r2, r1, 1],
        ]
    )

    assert toeplitz_matrix(3) == expected


def test_symbolic_pacf_second_order():
    r1, r2 = sp.symbols("r1 r2", real=True)

    expected = (r2 - r1**2) / (1 - r1**2)

    assert sp.simplify(pacf_symbolic(2) - expected) == 0


def test_symbolic_second_order_bounds():
    r1 = sp.symbols("r1", real=True)

    r_lower, r_upper = admissible_bounds_symbolic(2)

    assert sp.simplify(r_lower - (2 * r1**2 - 1)) == 0
    assert sp.simplify(r_upper - 1) == 0


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_symbolic_sh_coordinate_equals_pacf(order):
    assert verify_x_equals_alpha(order) == 0


# --- non-finite inputs -------------------------------------------------------


_NON_FINITE = [np.nan, np.inf, -np.inf]


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_pacf_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.pacf(np.array([0.2, bad]))


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_pacf_status_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.pacf_status(np.array([bad]))


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_pacf_prefix_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.pacf_prefix(np.array([0.2, bad]))


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_from_pacf_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.from_pacf(np.array([0.2, bad]))


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_admissible_bounds_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.admissible_bounds(np.array([0.2, bad]))


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_extend_at_boundary_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.extend_at_boundary(np.array([1.0, bad]), n_extra=1)


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_check_admissibility_treats_non_finite_as_inadmissible(bad):
    # check_admissibility() catches ValueError internally and reports
    # False rather than propagating it; raise_error=True still surfaces a
    # ValueError, just its own generic message, not the finite-value one.
    assert not ccf.check_admissibility(np.array([0.2, bad]))

    with pytest.raises(ValueError):
        ccf.check_admissibility(np.array([0.2, bad]), raise_error=True)


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_fisher_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.fisher(np.array([0.2, bad]))


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_inverse_fisher_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.inverse_fisher(np.array([0.2, bad]))


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_innovation_variances_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.innovation_variances(np.array([0.2, bad]))


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_jacobian_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.jacobian(np.array([0.2, bad]))


@pytest.mark.parametrize("bad", _NON_FINITE)
def test_log_jacobian_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.log_jacobian(np.array([0.2, bad]))


def test_pacf_status_batch_one_bad_row_fails_immediately_not_interior():
    r = np.array([
        [0.2, 0.1],
        [0.2, np.nan],
    ])

    with pytest.raises(ValueError, match="finite"):
        ccf.pacf_status(r)


def test_pacf_batch_one_bad_row_fails_immediately():
    r = np.array([
        [0.2, 0.1],
        [0.2, np.nan],
    ])

    with pytest.raises(ValueError, match="finite"):
        ccf.pacf(r)


def test_admissible_bounds_empty_input_still_allowed():
    # The finite check must accept empty arrays (np.all(np.isfinite([]))
    # is vacuously True); this preserves the existing empty-input contract.
    r_lower, r_upper = ccf.admissible_bounds(np.array([]))

    np.testing.assert_allclose(r_lower, [-1.0])
    np.testing.assert_allclose(r_upper, [1.0])


# --- whole-sequence boundary semantics ---------------------------------------


def test_boundary_tail_inconsistent_freezes_status_prefix_and_admissibility():
    r = np.array([1.0, 0.5])

    status = ccf.pacf_status(r)
    assert status.boundary
    assert status.order == 1
    assert not ccf.check_admissibility(r)

    prefix = ccf.pacf_prefix(r)
    assert prefix.reached_boundary
    assert prefix.order == 1

    with pytest.raises(ValueError, match="inconsistent"):
        ccf.admissible_bounds(r)


def test_boundary_tail_consistent_status_reports_first_boundary_only():
    r = np.array([1.0, 1.0, 1.0])

    status = ccf.pacf_status(r)
    assert status.boundary
    assert status.order == 1
    assert ccf.check_admissibility(r)

    prefix = ccf.pacf_prefix(r)
    assert prefix.order == 1


# --- extend_at_boundary n_extra semantics -------------------------------------


def test_extend_at_boundary_n_extra_counts_existing_tail():
    r_boundary = np.array([1.0])
    r_with_two_forced_values = ccf.extend_at_boundary(r_boundary, n_extra=2)
    assert r_with_two_forced_values.size == 3  # 1 (boundary order) + 2

    # n_extra already matches the number of entries past the boundary that
    # r_with_two_forced_values supplies, so nothing new is appended.
    same = ccf.extend_at_boundary(r_with_two_forced_values, n_extra=2)
    np.testing.assert_allclose(same, r_with_two_forced_values)

    # n_extra=4 counts the 2 already-supplied entries, so 2 more are added.
    extended = ccf.extend_at_boundary(r_with_two_forced_values, n_extra=4)
    assert extended.size == 5  # 1 (boundary order) + 4
    np.testing.assert_allclose(extended[:3], r_with_two_forced_values)
