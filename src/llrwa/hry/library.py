"""Adapter around the `lead-lag` package (v2.3), which serves as a parity benchmark.

Library conventions this adapter handles:
- Timestamps are taken as nanoseconds. A pandas index in any other unit silently breaks
  the estimate (pandas 3 often defaults to microseconds), so we always send ns.
- Values are used as given, so we pass log prices.
- `specific_lags` are integer grid steps. A positive lag means the first series leads.
- The input Series are mutated in place, so we pass copies.
- Contrast = |shifted HY sum| / ((||dX|| + 1e-8)(||dY|| + 1e-8)).
- LLR = sum_{k>0} C(k) / sum_{k<0} C(k), unsquared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import NS_PER_MS

_ALLOWED_PRECISION_MS = (1000.0, 100.0, 10.0, 1.0, 0.1)


def snap_floor(t_ns: np.ndarray, p: np.ndarray, precision_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Floor timestamps to the precision grid and keep the last value per cell.

    This mirrors the library's gridding. It is used only for parity runs: flooring
    shifts continuous timestamps by about -precision/2 (see DECISIONS.md D8).
    """
    step = int(round(precision_ms * NS_PER_MS))
    cell = (np.asarray(t_ns, dtype=np.int64) // step) * step
    last = np.ones(len(cell), dtype=bool)
    last[:-1] = cell[1:] != cell[:-1]
    return cell[last], np.asarray(p, dtype=float)[last]


def to_series(t_ns: np.ndarray, values: np.ndarray) -> pd.Series:
    idx = pd.DatetimeIndex(np.asarray(t_ns, dtype="datetime64[ns]")).as_unit("ns")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def library_contrast(tx_ns: np.ndarray, x: np.ndarray, ty_ns: np.ndarray, y: np.ndarray, *,
                     max_lag_ms: float, precision_ms: float = 1.0,
                     lags_steps: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Run `lead_lag.LeadLag` single-threaded. Returns (lags_ms, contrasts, llr).

    Inputs must already be on the precision grid, with unique timestamps
    (see `snap_floor`).
    """
    from lead_lag import LeadLag

    if precision_ms not in _ALLOWED_PRECISION_MS:
        raise ValueError(f"precision_ms must be one of {_ALLOWED_PRECISION_MS}")
    step_ns = int(round(precision_ms * NS_PER_MS))
    for t in (tx_ns, ty_ns):
        if np.any(np.asarray(t) % step_ns):
            raise ValueError("timestamps are not on the precision grid; call snap_floor first")
    kmax = int(round(max_lag_ms / precision_ms))
    lags = np.arange(-kmax, kmax + 1) if lags_steps is None else np.asarray(lags_steps, dtype=int)
    ll = LeadLag(to_series(tx_ns, x), to_series(ty_ns, y), max_lag=max_lag_ms / 1e3,
                 min_precision=precision_ms / 1e3, specific_lags=[int(k) for k in lags])
    ll.run_inference(num_threads=1)
    return np.asarray(ll.lag_range, dtype=float) * precision_ms, np.asarray(ll.contrasts, dtype=float), float(ll.llr)
