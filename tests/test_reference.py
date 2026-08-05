"""Tests for the didactic reference implementation (schurcorr.reference)."""

import numpy as np
import pytest

import schurcorr as sc
from schurcorr.reference import from_pacf_reference, pacf_reference


@pytest.mark.parametrize("order", [1, 2, 3, 5, 10, 20])
def test_pacf_reference_matches_pacf(order):
    rng = np.random.default_rng(order)
    alpha = rng.uniform(-0.8, 0.8, size=order)
    r = sc.from_pacf(alpha)

    np.testing.assert_allclose(
        pacf_reference(r), sc.pacf(r), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("order", [1, 2, 3, 5, 10, 20])
def test_from_pacf_reference_matches_from_pacf(order):
    rng = np.random.default_rng(order + 100)
    alpha = rng.uniform(-0.8, 0.8, size=order)

    np.testing.assert_allclose(
        from_pacf_reference(alpha), sc.from_pacf(alpha), rtol=1e-12, atol=1e-12
    )


def test_reference_roundtrip():
    alpha = np.array([0.5, -0.3, 0.2, 0.4, -0.1])

    r = from_pacf_reference(alpha)
    alpha_back = pacf_reference(r)

    np.testing.assert_allclose(alpha_back, alpha, rtol=1e-12, atol=1e-12)


def test_pacf_reference_rejects_2d_input():
    with pytest.raises(ValueError):
        pacf_reference(np.zeros((2, 3)))


def test_from_pacf_reference_rejects_2d_input():
    with pytest.raises(ValueError):
        from_pacf_reference(np.zeros((2, 3)))


def test_from_pacf_reference_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        from_pacf_reference(np.array([0.2, 1.0]))


def test_pacf_reference_rejects_inadmissible_sequence():
    with pytest.raises(ValueError):
        pacf_reference(np.array([1.0, 0.5, 0.3]))


def test_reference_not_exported_at_top_level():
    assert not hasattr(sc, "pacf_reference")
    assert not hasattr(sc, "from_pacf_reference")
