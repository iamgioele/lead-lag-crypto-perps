import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lead_lag")

from llrwa.hry.engine import LagGrid, NS_PER_MS, contrast, increments
from llrwa.hry.library import library_contrast, snap_floor, to_series


def _pair(seed, n_x=4000, n_y=2500, span_ms=600_000, margin_ms=5_000):
    rng = np.random.default_rng(seed)
    ty = np.unique(rng.integers(0, span_ms, n_y)) * NS_PER_MS
    tx = np.unique(rng.integers(margin_ms, span_ms - margin_ms, n_x)) * NS_PER_MS
    base = np.cumsum(rng.normal(size=span_ms + 1)) * 1e-4            # common 1 ms path
    x = base[tx // NS_PER_MS]
    y = 0.8 * base[np.maximum(ty // NS_PER_MS - 30, 0)] + 1e-4 * rng.normal(size=len(ty)).cumsum()
    return tx.astype(np.int64), x, ty.astype(np.int64), y


def test_library_matches_normalised_abs_contrast_away_from_edges():
    tx, x, ty, y = _pair(1)
    lags_ms, c_lib, _ = library_contrast(tx, x, ty, y, max_lag_ms=200)
    X, Y = increments(tx, x, keep_zero=True), increments(ty, y, keep_zero=True)
    grid = LagGrid.symmetric(200 * NS_PER_MS, NS_PER_MS)
    v = contrast(X, Y, grid)
    ours = np.abs(v) / ((np.sqrt(X.sum_sq) + 1e-8) * (np.sqrt(Y.sum_sq) + 1e-8))
    assert np.allclose(grid.lags_ms, lags_ms)
    assert np.allclose(ours, c_lib, rtol=1e-9, atol=1e-12)
    assert lags_ms[np.argmax(c_lib)] == grid.lags_ms[np.argmax(np.abs(v))]   # same estimate


def test_microsecond_index_would_break_library_but_adapter_sends_ns():
    from lead_lag import lag
    rng = np.random.default_rng(0)
    vals = np.cumsum(rng.normal(size=2000))
    idx_us = pd.date_range("2026-01-01", periods=2000, freq="1s").as_unit("us")
    x = pd.Series(vals, index=idx_us)
    y = x.shift(9).dropna()
    assert lag(x.copy(), y.copy(), max_lag=20) != 9.0          # silent failure in the raw library
    s = to_series(x.index.as_unit("ns").asi8, x.values)
    assert s.index.unit == "ns"
    assert lag(s.copy(), to_series(y.index.as_unit("ns").asi8, y.values), max_lag=20) == 9.0


def test_snap_floor_keeps_last_per_cell():
    t = np.array([1_000_100, 1_000_900, 2_000_000], dtype=np.int64)
    p = np.array([1.0, 2.0, 3.0])
    ts, ps = snap_floor(t, p, 1.0)
    assert list(ts) == [1_000_000, 2_000_000] and list(ps) == [2.0, 3.0]
