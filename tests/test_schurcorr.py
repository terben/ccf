import numpy as np
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


def test_pacf_extend_at_boundary_not_implemented():
    r = np.array([1.0, 0.5, 0.3])

    with pytest.raises(NotImplementedError):
        sc.pacf(r, at_boundary="extend")


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
