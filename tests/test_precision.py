"""Regression tests for ccf.precision (arbitrary-precision path)."""

import mpmath as mp
import pytest

import ccf


def test_from_pacf_mp_preserves_mpf_precision_beyond_float64():
    # Order 1 isolates the input-conversion step: r_1 = alpha_1 exactly,
    # regardless of dps, so any precision lost here can only have
    # happened while converting the input, not during the recursion.
    with mp.workdps(50):
        alpha_hi = mp.mpf("0.123456789012345678901234567890123456789")

    r = ccf.from_pacf_mp([alpha_hi], dps=50)

    with mp.workdps(50):
        assert abs(r[0] - alpha_hi) < mp.mpf("1e-45")


def test_from_pacf_mp_admissibility_checked_in_mpf_domain_not_float64():
    # 1 - 1e-40 rounds to exactly 1.0 in float64, so a float64 round-trip
    # before validation would incorrectly reject it; checked as mpf at
    # dps=50, it is a valid, strictly admissible coefficient.
    with mp.workdps(50):
        alpha_hi = mp.mpf(1) - mp.mpf("1e-40")

    r = ccf.from_pacf_mp([alpha_hi], dps=50)

    with mp.workdps(50):
        assert abs(r[0] - alpha_hi) < mp.mpf("1e-45")


def test_from_pacf_mp_rejects_out_of_range_alpha():
    with pytest.raises(ValueError):
        ccf.from_pacf_mp([mp.mpf(1) + mp.mpf("1e-40")], dps=50)


def test_pacf_mp_and_from_pacf_mp_roundtrip_at_high_dps():
    with mp.workdps(60):
        alpha_hi = [
            mp.mpf(1) / mp.mpf(3),
            -mp.mpf(2) / mp.mpf(7),
            mp.mpf("0.123456789012345678901234567890"),
        ]

    r = ccf.from_pacf_mp(alpha_hi, dps=60)
    alpha_back = ccf.pacf_mp(r, dps=60)

    with mp.workdps(60):
        for a, b in zip(alpha_hi, alpha_back):
            assert abs(mp.mpf(a) - b) < mp.mpf("1e-40")


def test_recommended_dps_rejects_invalid_n_max():
    with pytest.raises(ValueError):
        ccf.recommended_dps(0)

    with pytest.raises(ValueError):
        ccf.recommended_dps(-5)

    with pytest.raises(ValueError):
        ccf.recommended_dps(2.5)


def test_recommended_dps_rejects_negative_safety_margin():
    with pytest.raises(ValueError):
        ccf.recommended_dps(10, safety_margin=-1)


@pytest.mark.parametrize("bad", [mp.nan, mp.inf, -mp.inf, float("nan"), float("inf")])
def test_pacf_mp_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.pacf_mp([0.2, bad])


@pytest.mark.parametrize("bad", [mp.nan, mp.inf, -mp.inf, float("nan"), float("inf")])
def test_from_pacf_mp_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite"):
        ccf.from_pacf_mp([0.2, bad])
