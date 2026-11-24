import numpy as np

from llrwa.hry.engine import LagGrid, NS_PER_MS, contrast, increments
from llrwa.hry.peaks import pick_peak
from llrwa.simulate import baselines, prices, sampling, theory


def _pair(seed, theta, T_ms=1_200_000, lam_x=20.0, lam_y=5.0, grid_ms=None, rho=0.8):
    rng = np.random.default_rng(seed)
    tx = sampling.poisson(lam_x, T_ms, rng)
    ty = sampling.grid(grid_ms, T_ms, rng) if grid_ms else sampling.poisson(lam_y, T_ms, rng)
    x, y = prices.lead_lag_pair(tx, ty, theta, rho, 1e-4, rng)
    return (*prices.observe(tx, x), *prices.observe(ty, y))


def test_reproducible_with_seed():
    a, b = _pair(7, 40.0), _pair(7, 40.0)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def _mean_rho_curve(reps, **kw):
    grid = LagGrid.symmetric(300 * NS_PER_MS, 5 * NS_PER_MS)
    curves = []
    for r in range(reps):
        tx, x, ty, y = _pair(100 + r, **kw)
        X, Y = increments(tx, x, keep_zero=True), increments(ty, y, keep_zero=True)
        curves.append(contrast(X, Y, grid) / np.sqrt(X.sum_sq * Y.sum_sq))
    return grid.lags_ms, np.mean(curves, axis=0)


def test_theory_matches_monte_carlo_poisson():
    lags, m = _mean_rho_curve(8, theta=40.0)
    k = theory.k_poisson(lags - 40.0, 20.0, 5.0)
    assert np.max(np.abs(m - 0.8 * k)) < 0.03


def test_theory_matches_monte_carlo_grid():
    lags, m = _mean_rho_curve(8, theta=40.0, grid_ms=100.0)
    k = theory.k_poisson_grid(lags - 40.0, 20.0, 100.0)
    assert np.max(np.abs(m - 0.8 * k)) < 0.03


def test_k_empirical_agrees_with_closed_form():
    rng = np.random.default_rng(1)
    tx = sampling.poisson(20.0, 3_600_000, rng)
    ty = sampling.poisson(5.0, 3_600_000, rng)
    h = np.array([-200.0, -50.0, 0.0, 50.0, 200.0])
    assert np.allclose(theory.k_empirical(tx, ty, 0.0, h, 50000, rng), theory.k_poisson(h, 20.0, 5.0), atol=0.02)


def test_previous_tick_baseline_skews_mass_but_hry_stays_symmetric():
    # theta = 0; X Poisson 50/s (E[D_X] = 20 ms), Y on a 100 ms grid (E[D_Y] = 50 ms)
    tx, x, ty, y = _pair(5, 0.0, T_ms=3_600_000, lam_x=50.0, grid_ms=100.0, rho=0.9)
    lags, c = baselines.xcorr_grid(tx, x, ty, y, bar_ms=1.0, max_lag_ms=300)
    centroid = np.sum(lags * c) / np.sum(c)
    assert abs(centroid - theory.previous_tick_centroid_shift_ms(20.0, 50.0)) <= 8
    assert c[lags > 0].clip(0).sum() > 2.5 * c[lags < 0].clip(0).sum()     # artificial lead of the fast venue
    grid = LagGrid.symmetric(304 * NS_PER_MS, NS_PER_MS)
    v = contrast(increments(tx, x), increments(ty, y), grid)
    peak, lv, s = pick_peak(grid.lags_ms, v, width=9, max_lag_ms=300)
    assert abs(peak.theta_ms) <= 10
    assert 0.7 < s[lv > 0].clip(0).sum() / s[lv < 0].clip(0).sum() < 1.4    # HRY mass stays balanced
