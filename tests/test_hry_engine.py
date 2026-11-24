import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from llrwa.hry.engine import (Increments, LagGrid, contrast, contrast_bruteforce,
                              contrast_per_lag, increments, NS_PER_MS)
from llrwa.hry.peaks import pick_peak


def _random_series(rng, n, span, unit):
    t = np.unique(rng.integers(0, span, n)) * unit
    p = np.cumsum(rng.normal(size=len(t)))
    return t.astype(np.int64), p


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), nx=st.integers(0, 60), ny=st.integers(0, 60),
       unit=st.sampled_from([1, 7, 1000]), step=st.sampled_from([2, 6, 10, 1000]),
       half=st.booleans(), change_only=st.booleans())
def test_three_implementations_agree(seed, nx, ny, unit, step, half, change_only):
    rng = np.random.default_rng(seed)
    tx, px = _random_series(rng, nx, 400, unit)
    ty, py = _random_series(rng, ny, 400, unit)
    X = increments(tx, px, change_only=change_only)
    Y = increments(ty, py, change_only=change_only)
    grid = LagGrid.symmetric(max_lag_ns=60 * unit, step_ns=step, half_step=half)
    fast = contrast(X, Y, grid)
    ref = contrast_per_lag(X, Y, grid.lags)
    brute = contrast_bruteforce(X, Y, grid.lags)
    assert np.allclose(fast, brute, atol=1e-9)
    assert np.allclose(ref, brute, atol=1e-9)


def test_ties_across_series_and_long_intervals():
    # identical timestamps in both series, and one Y interval much longer than the lag range
    tx = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64) * NS_PER_MS
    ty = np.array([0, 10, 20, 500, 510], dtype=np.int64) * NS_PER_MS
    px = np.array([0.0, 1, 0, 2, 1, 3])
    py = np.array([0.0, -1, 1, 0, 2])
    X, Y = increments(tx, px), increments(ty, py)
    grid = LagGrid.symmetric(100 * NS_PER_MS, NS_PER_MS)
    assert np.allclose(contrast(X, Y, grid), contrast_bruteforce(X, Y, grid.lags))


def test_empty_and_single_observation():
    one = increments(np.array([5], dtype=np.int64), np.array([1.0]))
    assert len(one) == 0
    grid = LagGrid.symmetric(10, 1)
    assert np.all(contrast(one, one, grid) == 0)


def test_blocks_sum_to_full_and_chunking_invariant():
    rng = np.random.default_rng(3)
    tx, px = _random_series(rng, 3000, 200_000, NS_PER_MS)
    ty, py = _random_series(rng, 1500, 200_000, NS_PER_MS)
    X, Y = increments(tx, px), increments(ty, py)
    grid = LagGrid.symmetric(500 * NS_PER_MS, NS_PER_MS, half_step=True)
    full = contrast(X, Y, grid)
    block = (X.end // (20_000 * NS_PER_MS)).astype(np.int64)
    per_block = contrast(X, Y, grid, block_of_x=block)
    assert per_block.shape == (block.max() + 1, grid.n)
    assert np.allclose(per_block.sum(axis=0), full)
    assert np.allclose(contrast(X, Y, grid, max_pairs=997), full)


def test_segments_drop_cross_boundary_increments():
    t = np.array([0, 1, 2, 10, 11], dtype=np.int64)
    p = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    seg = np.array([0, 0, 0, 1, 1])
    inc = increments(t, p, segment=seg)
    assert list(zip(inc.start, inc.end)) == [(0, 1), (1, 2), (10, 11)]
    assert np.allclose(inc.dx, [1.0, 1.0, 1.0])


def test_change_only_merges_intervals():
    t = np.array([0, 10, 20, 30], dtype=np.int64)
    p = np.array([1.0, 1.0, 2.0, 2.0])
    all_upd = increments(t, p)
    chg = increments(t, p, change_only=True)
    assert list(zip(all_upd.start, all_upd.end)) == [(10, 20)]   # true left end kept
    assert list(zip(chg.start, chg.end)) == [(0, 20)]            # merged back to the last change


def test_unsorted_or_duplicate_times_rejected():
    with pytest.raises(ValueError):
        increments(np.array([0, 0, 1], dtype=np.int64), np.array([1.0, 2, 3]))


def _lead_lag_pair(rng, theta_ms, lam_x=20.0, lam_y=8.0, T_s=1800.0, rho=0.9):
    """X = W at Poisson times; Y = rho W(. - theta) + sqrt(1-rho^2) B at other Poisson times."""
    T = T_s * 1e3
    tx = np.sort(rng.uniform(0, T, rng.poisson(lam_x * T_s)))
    ty = np.sort(rng.uniform(0, T, rng.poisson(lam_y * T_s)))
    allt = np.concatenate([tx, ty - theta_ms])
    o = np.argsort(allt, kind="stable")
    ts = allt[o]
    W = np.empty_like(ts)
    W[o] = np.cumsum(rng.normal(size=len(ts)) * np.sqrt(np.diff(np.concatenate([[ts[0] - 1.0], ts]))))
    B = np.cumsum(rng.normal(size=len(ty)) * np.sqrt(np.diff(np.concatenate([[0.0], ty]))))
    x, y = W[:len(tx)], rho * W[len(tx):] + np.sqrt(1 - rho ** 2) * B
    to_ns = lambda a: np.round(a * NS_PER_MS).astype(np.int64)
    tx_ns, ix = np.unique(to_ns(tx), return_index=True)
    ty_ns, iy = np.unique(to_ns(ty), return_index=True)
    return tx_ns, x[ix], ty_ns, y[iy]


def test_sign_convention_x_leads_y_positive():
    rng = np.random.default_rng(42)
    tx, x, ty, y = _lead_lag_pair(rng, theta_ms=50.0)
    X, Y = increments(tx, x), increments(ty, y)
    grid = LagGrid.symmetric(304 * NS_PER_MS, NS_PER_MS)          # +-300 ms plus smoothing margin
    peak, _, _ = pick_peak(grid.lags_ms, contrast(X, Y, grid), width=9, max_lag_ms=300)
    assert abs(peak.theta_ms - 50) <= 2 and not peak.censored
    peak_swapped, _, _ = pick_peak(grid.lags_ms, contrast(Y, X, grid), width=9, max_lag_ms=300)
    assert abs(peak_swapped.theta_ms + 50) <= 2
