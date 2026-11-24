"""Baselines: synchronise on a fixed grid (previous tick), then cross-correlate returns.

corr(L) = corr(r_x[t], r_y[t + L]), so L > 0 means Y follows X. One FFT gives all lags.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sfft


def previous_tick(t_ns: np.ndarray, logp: np.ndarray, t0_ns: int, t1_ns: int, bar_ns: int) -> np.ndarray:
    """Log price on the grid t0 + k*bar: the last observation at or before each grid point."""
    grid = np.arange(t0_ns, t1_ns, bar_ns, dtype=np.int64)
    idx = np.searchsorted(t_ns, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    ok = idx >= 0
    out[ok] = np.asarray(logp)[idx[ok]]
    return out


def xcorr_grid(tx_ns, x, ty_ns, y, bar_ms: float, max_lag_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """(lags_ms, corr) of previous-tick bar returns, for lags in [-max_lag, max_lag]."""
    bar = int(round(bar_ms * 1e6))
    t0 = int(max(tx_ns[0], ty_ns[0]))
    t1 = int(min(tx_ns[-1], ty_ns[-1]))
    px = previous_tick(np.asarray(tx_ns), x, t0, t1, bar)
    py = previous_tick(np.asarray(ty_ns), y, t0, t1, bar)
    rx = np.nan_to_num(np.diff(px))
    ry = np.nan_to_num(np.diff(py))
    rx -= rx.mean()
    ry -= ry.mean()
    W = int(max_lag_ms // bar_ms)
    n = sfft.next_fast_len(len(rx) + W + 1)
    cc = sfft.irfft(np.conj(sfft.rfft(rx, n)) * sfft.rfft(ry, n), n)
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    pos, neg = cc[:W + 1] / d, cc[-W:][::-1] / d
    corr = np.concatenate([neg[::-1], pos])
    return np.arange(-W, W + 1) * bar_ms, corr
