import mpmath as mp
import numpy as np
import scipy.linalg as sla
import sympy as sp
import pytest
import schurcorr as sc

from schurcorr import symbolic


def test_pacf_roundtrip():
    alpha = np.array([0.2, -0.3, 0.4])

    r = sc.from_pacf(alpha)
    alpha_back = sc.pacf(r)

    np.testing.assert_allclose(alpha_back, alpha, rtol=1e-12, atol=1e-12)


def test_pacf_default_raises_at_boundary():
    r = np.array([1.0, 0.5, 0.3])

    with pytest.raises(sc.SingularToeplitzError):
        sc.pacf(r)


def test_pacf_warn_at_boundary_returns_truncated():
    r = np.array([1.0, 0.5, 0.3])

    with pytest.warns(RuntimeWarning):
        alpha = sc.pacf(r, at_boundary="warn")

    np.testing.assert_allclose(alpha, [1.0])


def test_pacf_extend_at_boundary_simple_order1():
    # r_1 = 1 forces every subsequent correlation to also equal 1.
    r = np.array([1.0, 1.0, 1.0])

    alpha = sc.pacf(r, at_boundary="extend")

    np.testing.assert_allclose(alpha, [1.0, 1.0, 1.0])


def test_pacf_extend_at_boundary_rejects_inconsistent_tail():
    r = np.array([1.0, 0.5, 0.3])

    with pytest.raises(ValueError, match="inconsistent"):
        sc.pacf(r, at_boundary="extend")


def test_pacf_extend_not_at_boundary_behaves_normally():
    alpha = np.array([0.2, -0.3, 0.4])
    r = sc.from_pacf(alpha)

    np.testing.assert_allclose(
        sc.pacf(r, at_boundary="extend"), alpha, rtol=1e-12, atol=1e-12
    )


def test_pacf_invalid_at_boundary_raises_value_error():
    alpha = np.array([0.2, -0.3, 0.4])
    r = sc.from_pacf(alpha)

    with pytest.raises(ValueError):
        sc.pacf(r, at_boundary="bogus")


def test_pacf_not_at_boundary_unaffected_by_at_boundary():
    alpha = np.array([0.2, -0.3, 0.4])
    r = sc.from_pacf(alpha)

    np.testing.assert_allclose(sc.pacf(r), alpha, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        sc.pacf(r, at_boundary="warn"), alpha, rtol=1e-12, atol=1e-12
    )


def test_fisher_roundtrip():
    alpha = np.array([0.2, -0.3, 0.4])

    y = sc.fisher(alpha)
    alpha_back = sc.inverse_fisher(y)

    np.testing.assert_allclose(alpha_back, alpha, rtol=1e-12, atol=1e-12)


# --- extend_at_boundary ----------------------------------------------------


def _order3_boundary_sequence():
    """
    Construct r_1, r_2, r_3 with A_1, A_2, A_3 positive definite and
    A_4 singular, i.e. the boundary is first reached at m = 4
    (alpha_3 = +1).
    """
    alpha12 = np.array([0.3, -0.2])
    r12 = sc.from_pacf(alpha12)

    _, r_upper = sc.admissible_bounds(np.append(r12, r12[-1]))

    r3 = r_upper[2]
    return np.append(r12, r3)


def test_extend_at_boundary_order1():
    r_ext = sc.extend_at_boundary(np.array([1.0]), n_extra=3)
    np.testing.assert_allclose(r_ext, [1.0, 1.0, 1.0, 1.0])


def test_extend_at_boundary_null_vector_matches_scipy():
    r = _order3_boundary_sequence()

    with pytest.warns(RuntimeWarning):
        r_ext = sc.extend_at_boundary(r, n_extra=3)

    # A_4, built from r_1, r_2, r_3, must be singular, and the
    # phi-derived null vector used internally must be proportional
    # (up to sign) to the null space computed directly by scipy.
    a4 = sla.toeplitz([1.0, r[0], r[1], r[2]])
    eigenvalues = np.linalg.eigvalsh(a4)
    assert np.min(np.abs(eigenvalues)) < 1e-8

    null_space = sla.null_space(a4, rcond=1e-8)
    assert null_space.shape == (4, 1)
    w = null_space[:, 0]

    with pytest.warns(RuntimeWarning):
        alpha_ext = sc.pacf(r, at_boundary="extend")

    phi = -w[1:] / w[0]
    predicted_r4 = float(np.dot(phi, r[::-1]))
    np.testing.assert_allclose(predicted_r4, r_ext[3], atol=1e-8)
    assert alpha_ext[2] == 1.0


def test_extend_at_boundary_recurrence_holds_for_generated_values():
    r = _order3_boundary_sequence()

    with pytest.warns(RuntimeWarning):
        r_ext = sc.extend_at_boundary(r, n_extra=4)

    # every extended matrix A_(4+s) must remain singular
    for k in range(4, len(r_ext) + 1):
        a_k = sla.toeplitz(np.concatenate(([1.0], r_ext[: k - 1])))
        eigenvalues = np.linalg.eigvalsh(a_k)
        assert np.min(np.abs(eigenvalues)) < 1e-6


def test_extend_at_boundary_validates_consistent_prefix():
    r = _order3_boundary_sequence()

    with pytest.warns(RuntimeWarning):
        r_ext = sc.extend_at_boundary(r, n_extra=2)

    with pytest.warns(RuntimeWarning):
        r_ext_again = sc.extend_at_boundary(r_ext, n_extra=2)

    np.testing.assert_allclose(r_ext_again, r_ext)


def test_extend_at_boundary_rejects_inconsistent_prefix():
    r = _order3_boundary_sequence()
    r_bad = np.append(r, 999.0)

    with pytest.raises(ValueError, match="inconsistent"):
        sc.extend_at_boundary(r_bad, n_extra=1)


def test_extend_at_boundary_n_extra_too_small_for_existing_excess():
    r = _order3_boundary_sequence()
    r_with_excess = np.append(r, r[-1])

    with pytest.raises(ValueError, match="n_extra"):
        sc.extend_at_boundary(r_with_excess, n_extra=0)


def test_extend_at_boundary_negative_n_extra_raises():
    with pytest.raises(ValueError):
        sc.extend_at_boundary(np.array([1.0]), n_extra=-1)


def test_extend_at_boundary_requires_boundary_reached():
    alpha = np.array([0.2, -0.3, 0.4])
    r = sc.from_pacf(alpha)

    with pytest.raises(ValueError, match="boundary"):
        sc.extend_at_boundary(r, n_extra=2)


def test_pacf_extend_batched():
    r_no_boundary = sc.from_pacf(np.array([0.2, -0.3, 0.4]))
    r_boundary = np.array([1.0, 1.0, 1.0])

    with pytest.warns(RuntimeWarning):
        alpha_batch = sc.pacf(
            np.stack([r_no_boundary, r_boundary]), at_boundary="extend"
        )

    np.testing.assert_allclose(alpha_batch[0], [0.2, -0.3, 0.4])
    np.testing.assert_allclose(alpha_batch[1], [1.0, 1.0, 1.0])


# --- mpmath backend ---------------------------------------------------------


def test_from_pacf_mpmath_matches_float64():
    alpha = [0.2, -0.3, 0.4]

    r_mp = sc.from_pacf(alpha, backend="mpmath")
    r_f64 = sc.from_pacf(np.array(alpha))

    assert all(isinstance(x, mp.mpf) for x in r_mp)
    np.testing.assert_allclose(
        [float(x) for x in r_mp], r_f64, rtol=1e-12, atol=1e-12
    )


def test_pacf_mpmath_matches_float64():
    alpha = [0.2, -0.3, 0.4]
    r = sc.from_pacf(alpha, backend="mpmath")

    alpha_back_mp = sc.pacf(r, backend="mpmath")
    alpha_back_f64 = sc.pacf(np.array([float(x) for x in r]))

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
    dps = sc.recommended_dps(len(alpha), safety_margin=30)

    r = sc.from_pacf(alpha, backend="mpmath", dps=dps)
    alpha_back = sc.pacf(r, backend="mpmath", dps=dps)

    for a, a_back in zip(alpha, alpha_back):
        assert abs(mp.mpf(a) - a_back) < mp.mpf("1e-12")


def test_recommended_dps_matches_formula():
    assert sc.recommended_dps(100) == 69
    assert sc.recommended_dps(100, target_exponent=-13, safety_margin=11) == 69


def test_pacf_mpmath_default_raises_at_boundary():
    r = [1.0, 0.5, 0.3]

    with pytest.raises(sc.SingularToeplitzError):
        sc.pacf(r, backend="mpmath")


def test_pacf_mpmath_warn_returns_truncated():
    r = [1.0, 0.5, 0.3]

    with pytest.warns(RuntimeWarning):
        alpha = sc.pacf(r, backend="mpmath", at_boundary="warn")

    assert len(alpha) == 1
    assert alpha[0] == mp.mpf(1)


def test_pacf_mpmath_extend_not_supported():
    r = [1.0, 0.5, 0.3]

    with pytest.raises(ValueError, match="extend"):
        sc.pacf(r, backend="mpmath", at_boundary="extend")


def test_pacf_mpmath_rejects_2d_input():
    r = np.array([[0.2, 0.1], [0.3, -0.1]])

    with pytest.raises(ValueError):
        sc.pacf(r, backend="mpmath")


def test_from_pacf_mpmath_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        sc.from_pacf([0.2, 1.5], backend="mpmath")


def test_pacf_invalid_backend_raises_value_error():
    with pytest.raises(ValueError):
        sc.pacf(np.array([0.2, -0.3]), backend="bogus")


def test_dps_rejected_for_float64_backend():
    with pytest.raises(ValueError):
        sc.pacf(np.array([0.2, -0.3]), dps=50)

    with pytest.raises(ValueError):
        sc.from_pacf(np.array([0.2, -0.3]), dps=50)


def test_pacf_mpmath_custom_dps():
    alpha_in = [0.2, -0.3, 0.4]
    r = sc.from_pacf(alpha_in, backend="mpmath", dps=50)
    alpha = sc.pacf(r, backend="mpmath", dps=50)

    assert mp.mp.dps != 50  # workdps must not leak into the global context
    np.testing.assert_allclose(
        [float(x) for x in alpha], alpha_in, rtol=1e-12, atol=1e-12
    )


# --- admissible_volume / log_admissible_volume ------------------------------


def _volume_symbolic(N):
    j = sp.symbols("j", positive=True, integer=True)
    prod = sp.Integer(1)
    for jj in range(1, N):
        prod *= sp.sqrt(sp.pi) * sp.factorial(jj) / sp.gamma(jj + sp.Rational(3, 2))
    return sp.nsimplify(2 * prod)


@pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6])
def test_admissible_volume_matches_closed_form(N):
    expected = float(_volume_symbolic(N))
    np.testing.assert_allclose(sc.admissible_volume(N), expected, rtol=1e-12)


def test_admissible_volume_order1_is_hypercube_width():
    # N = 1: r_1 = alpha_1 in (-1, 1) directly, volume = 2.
    assert sc.admissible_volume(1) == 2.0


@pytest.mark.parametrize("N", [1, 2, 5, 10, 50])
def test_log_admissible_volume_matches_admissible_volume(N):
    np.testing.assert_allclose(
        sc.log_admissible_volume(N),
        np.log(sc.admissible_volume(N)),
        rtol=1e-10,
    )


def test_admissible_volume_underflows_where_log_does_not():
    assert sc.admissible_volume(2000) == 0.0
    assert np.isfinite(sc.log_admissible_volume(2000))
    assert sc.log_admissible_volume(2000) < 0.0


@pytest.mark.parametrize("N", [0, -1, 2.5])
def test_admissible_volume_rejects_invalid_N(N):
    with pytest.raises(ValueError):
        sc.admissible_volume(N)

    with pytest.raises(ValueError):
        sc.log_admissible_volume(N)


@pytest.mark.parametrize("N", [2, 3, 4])
def test_sh_normalized_volume_matches_monte_carlo_admissible_fraction(N):
    # Independent cross-check of V_N^SH = V_N / 2^N (Eq. volume_SH):
    # it is the probability that a correlation sequence drawn
    # uniformly from [-1, 1]^N is admissible.
    rng = np.random.default_rng(20260714)
    trials = 20_000
    samples = rng.uniform(-1.0, 1.0, size=(trials, N))

    admissible_count = sum(
        1 for row in samples if sc.check_admissibility(row)
    )
    empirical_fraction = admissible_count / trials

    v_sh = sc.admissible_volume(N) / 2.0**N

    assert abs(empirical_fraction - v_sh) < 0.02


def test_sh_normalized_volume_is_a_probability():
    for N in [1, 2, 5, 10, 30]:
        v_sh = sc.admissible_volume(N) / 2.0**N
        assert 0.0 < v_sh <= 1.0


# --- Batched (2-D) interface ---------------------------------------------


def _random_batch(rng, n_samples=5, n=6):
    return rng.uniform(-0.9, 0.9, size=(n_samples, n))


def test_batched_from_pacf_matches_looped_1d():
    rng = np.random.default_rng(0)
    alpha = _random_batch(rng)

    r_batch = sc.from_pacf(alpha)
    r_loop = np.stack([sc.from_pacf(row) for row in alpha])

    assert r_batch.shape == alpha.shape
    np.testing.assert_array_equal(r_batch, r_loop)


def test_batched_pacf_matches_looped_1d():
    rng = np.random.default_rng(1)
    alpha = _random_batch(rng)
    r = sc.from_pacf(alpha)

    alpha_batch = sc.pacf(r)
    alpha_loop = np.stack([sc.pacf(row) for row in r])

    assert alpha_batch.shape == r.shape
    np.testing.assert_array_equal(alpha_batch, alpha_loop)
    np.testing.assert_allclose(alpha_batch, alpha, rtol=1e-12, atol=1e-12)


def test_batched_fisher_and_inverse_fisher_match_looped_1d():
    rng = np.random.default_rng(2)
    alpha = _random_batch(rng)

    y_batch = sc.fisher(alpha)
    y_loop = np.stack([sc.fisher(row) for row in alpha])
    np.testing.assert_array_equal(y_batch, y_loop)

    alpha_back_batch = sc.inverse_fisher(y_batch)
    alpha_back_loop = np.stack([sc.inverse_fisher(row) for row in y_batch])
    np.testing.assert_array_equal(alpha_back_batch, alpha_back_loop)


def test_batched_innovation_variances_matches_looped_1d():
    rng = np.random.default_rng(3)
    alpha = _random_batch(rng)

    sigma2_batch = sc.innovation_variances(alpha)
    sigma2_loop = np.stack([sc.innovation_variances(row) for row in alpha])

    assert sigma2_batch.shape == (alpha.shape[0], alpha.shape[1] + 1)
    np.testing.assert_array_equal(sigma2_batch, sigma2_loop)


def test_1d_output_unchanged_by_batched_interface():
    alpha = np.array([0.2, -0.3, 0.4])
    r = sc.from_pacf(alpha)

    assert r.ndim == 1
    assert sc.pacf(r).ndim == 1
    assert sc.fisher(alpha).ndim == 1
    assert sc.inverse_fisher(sc.fisher(alpha)).ndim == 1
    assert sc.innovation_variances(alpha).ndim == 1

    np.testing.assert_allclose(sc.pacf(r), alpha, rtol=1e-12, atol=1e-12)


def test_batched_pacf_raise_reports_offending_sample():
    r = np.array([
        [0.2, 0.1, 0.05],
        [1.0, 0.5, 0.3],
    ])

    with pytest.raises(sc.SingularToeplitzError, match="sample 1"):
        sc.pacf(r)


def test_batched_pacf_warn_pads_truncated_row_with_nan():
    r = np.array([
        [0.2, 0.1, 0.05],
        [1.0, 0.5, 0.3],
    ])

    with pytest.warns(RuntimeWarning):
        alpha = sc.pacf(r, at_boundary="warn")

    assert alpha.shape == r.shape
    np.testing.assert_allclose(alpha[0], [0.2, 0.0625, 0.01960784], rtol=1e-6)
    assert alpha[1, 0] == 1.0
    assert np.all(np.isnan(alpha[1, 1:]))


def test_batched_from_pacf_reports_offending_sample():
    alpha = np.array([
        [0.1, 0.2],
        [0.1, 1.5],
    ])

    with pytest.raises(ValueError, match="sample 1"):
        sc.from_pacf(alpha)


def test_pacf_invalid_ndim_raises_value_error():
    with pytest.raises(ValueError):
        sc.pacf(np.zeros((2, 2, 2)))


def test_innovation_variances():
    alpha = np.array([0.2, -0.3, 0.4])

    sigma2 = sc.innovation_variances(alpha)

    expected = np.array([
        1.0,
        1.0 * (1.0 - 0.2**2),
        1.0 * (1.0 - 0.2**2) * (1.0 - (-0.3)**2),
        1.0 * (1.0 - 0.2**2) * (1.0 - (-0.3)**2) * (1.0 - 0.4**2),
    ])

    np.testing.assert_allclose(sigma2, expected, rtol=1e-12, atol=1e-12)


def test_admissible_bounds_contain_input():
    alpha = np.array([0.2, -0.3, 0.4])
    r = sc.from_pacf(alpha)

    r_lower, r_upper = sc.admissible_bounds(r)

    assert np.all(r >= r_lower)
    assert np.all(r <= r_upper)


def test_sh_coordinates_equal_pacf():
    alpha = np.array([0.2, -0.3, 0.4])
    r = sc.from_pacf(alpha)

    x = sc.sh_coordinates(r)

    np.testing.assert_allclose(x, alpha, rtol=1e-12, atol=1e-12)


def test_check_admissibility_true():
    alpha = np.array([0.2, -0.3, 0.4])
    r = sc.from_pacf(alpha)

    assert sc.check_admissibility(r)


def test_check_admissibility_false():
    r = np.array([0.9, -0.9])

    assert not sc.check_admissibility(r)


def test_invalid_alpha_from_pacf():
    alpha = np.array([0.2, 1.0])

    with pytest.raises(ValueError):
        sc.from_pacf(alpha)


def test_invalid_alpha_fisher():
    alpha = np.array([0.2, 1.0])

    with pytest.raises(ValueError):
        sc.fisher(alpha)


def test_log_jacobian_matches_jacobian():
    alpha = np.array([0.2, -0.3, 0.4])

    log_j = sc.log_jacobian(alpha)
    j = sc.jacobian(alpha)

    np.testing.assert_allclose(np.log(j), log_j, rtol=1e-12, atol=1e-12)


from schurcorr.symbolic import (
    admissible_bounds_symbolic,
    pacf_symbolic,
    toeplitz_matrix,
    verify_x_equals_alpha,
)

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
