import numpy as np

from llrwa.hry.peaks import (far_lag_z, llr_squared, llr_unsquared_abs, local_maxima, pick_peak,
                             smooth_valid)


def test_smooth_valid_has_no_edge_padding():
    v = np.arange(20, dtype=float)             # rising to the edge
    s = smooth_valid(v, 9)
    assert len(s) == 12
    assert np.allclose(s, np.arange(4, 16))    # exact moving average, no zeros mixed in


def test_rising_curve_is_censored_not_reported_inside():
    lags = np.arange(-254, 255, dtype=float)   # [-250, 250] after a width-9 valid smooth
    v = lags.copy()                            # true peak beyond +250 ms
    peak, lv, _ = pick_peak(lags, v, width=9, max_lag_ms=250)
    assert peak.censored and peak.theta_ms == lv[-1] == 250


def test_interior_peak_not_censored():
    lags = np.arange(-254, 255, dtype=float)
    v = np.exp(-((lags - 40) / 30) ** 2)
    peak, _, _ = pick_peak(lags, v, width=9, max_lag_ms=250)
    assert peak.theta_ms == 40 and not peak.censored and not peak.ambiguous


def test_ambiguous_double_peak():
    lags = np.arange(-254, 255, dtype=float)
    v = np.exp(-((lags - 40) / 10) ** 2) + 0.9 * np.exp(-((lags + 120) / 10) ** 2)
    peak, _, _ = pick_peak(lags, v, width=9, max_lag_ms=250)
    assert peak.theta_ms == 40 and peak.ambiguous and abs(peak.second_ms + 120) <= 1


def test_local_maxima_plateau_counted_once():
    assert list(local_maxima(np.array([0, 1, 2, 2, 2, 1, 0, 3, 0.0]))) == [2, 7]


def test_llr_definitions():
    lags = np.arange(-10, 11, dtype=float)
    rho = np.where(lags > 0, 0.4, 0.2)
    assert np.isclose(llr_squared(lags, rho, np.array([1, 2, 3.0])), 4.0)
    assert np.isclose(llr_unsquared_abs(lags, rho, np.array([1, 2, 3.0])), 2.0)


def test_far_lag_z_detects_signal():
    rng = np.random.default_rng(0)
    lags = np.arange(-8000, 8001, 10, dtype=float)
    v = rng.normal(size=len(lags))
    assert far_lag_z(lags, v, peak_value=10.0) > 5
