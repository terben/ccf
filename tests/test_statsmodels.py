"""
Independent cross-checks against statsmodels.

These tests compare the numerical Levinson--Durbin implementation in
ccf with the corresponding routines provided by statsmodels.

The comparison uses exact autocorrelation sequences rather than estimated
sample autocorrelations. Therefore, statsmodels is called with
``isacov=True`` where appropriate.
"""

from __future__ import annotations

import numpy as np
import pytest

import ccf as sc


statsmodels = pytest.importorskip("statsmodels")

from statsmodels.tsa.stattools import (  # noqa: E402
    levinson_durbin,
    levinson_durbin_pacf,
)


@pytest.mark.parametrize(
    "alpha",
    [
        np.array([0.2]),
        np.array([0.2, -0.3]),
        np.array([0.2, -0.3, 0.4]),
        np.array([-0.7, 0.1, 0.5, -0.2]),
        np.array([0.8, -0.6, 0.3, 0.1, -0.4]),
    ],
)
def test_pacf_matches_statsmodels(alpha: np.ndarray) -> None:
    """
    Compare the forward recursion r -> alpha with statsmodels.
    """
    r = sc.from_pacf(alpha)

    # statsmodels expects the autocovariance sequence beginning at lag 0.
    autocovariance = np.concatenate(([1.0], r))

    _, _, pacf_statsmodels, _, _ = levinson_durbin(
        autocovariance,
        nlags=alpha.size,
        isacov=True,
    )

    alpha_ccf = sc.pacf(r)

    # statsmodels includes PACF(0) = 1.
    np.testing.assert_allclose(
        alpha_ccf,
        pacf_statsmodels[1:],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "alpha",
    [
        np.array([0.2]),
        np.array([0.2, -0.3]),
        np.array([0.2, -0.3, 0.4]),
        np.array([-0.7, 0.1, 0.5, -0.2]),
        np.array([0.8, -0.6, 0.3, 0.1, -0.4]),
    ],
)
def test_from_pacf_matches_statsmodels(alpha: np.ndarray) -> None:
    """
    Compare the inverse recursion alpha -> r with statsmodels.
    """
    # statsmodels expects PACF(0) = 1.
    pacf_with_zero_lag = np.concatenate(([1.0], alpha))

    _, autocorrelation_statsmodels = levinson_durbin_pacf(
        pacf_with_zero_lag,
        nlags=alpha.size,
    )

    r_ccf = sc.from_pacf(alpha)

    # statsmodels includes ACF(0) = 1.
    np.testing.assert_allclose(
        r_ccf,
        autocorrelation_statsmodels[1:],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_statsmodels_crosscheck_random_sequences() -> None:
    """
    Compare both recursion directions for reproducible random PACFs.
    """
    rng = np.random.default_rng(20260712)

    for order in range(1, 11):
        for _ in range(20):
            # Staying away from the boundary avoids testing unrelated
            # floating-point behavior near abs(alpha_n) = 1.
            alpha = rng.uniform(-0.8, 0.8, size=order)

            r_ccf = sc.from_pacf(alpha)

            pacf_with_zero_lag = np.concatenate(([1.0], alpha))
            _, r_statsmodels = levinson_durbin_pacf(
                pacf_with_zero_lag,
                nlags=order,
            )

            np.testing.assert_allclose(
                r_ccf,
                r_statsmodels[1:],
                rtol=1.0e-11,
                atol=1.0e-12,
            )

            autocovariance = np.concatenate(([1.0], r_ccf))
            _, _, alpha_statsmodels, _, _ = levinson_durbin(
                autocovariance,
                nlags=order,
                isacov=True,
            )

            np.testing.assert_allclose(
                sc.pacf(r_ccf),
                alpha_statsmodels[1:],
                rtol=1.0e-11,
                atol=1.0e-12,
            )
