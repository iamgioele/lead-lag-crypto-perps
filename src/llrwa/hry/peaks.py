"""Peak picking, lead/lag ratios and a far-lag signal test on a contrast curve.

Curves are computed on an extended lag range [-L - h, L + h], where h is the smoothing
half-width. Smoothing then runs in 'valid' mode and returns values on [-L, L] only, with
no zero padding at the edges.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def smooth_valid(v: np.ndarray, width: int) -> np.ndarray:
    """Boxcar moving average in 'valid' mode. Returns len(v) - width + 1 values."""
    if width < 1 or width % 2 == 0:
        raise ValueError("width must be a positive odd integer")
    if width == 1:
        return np.asarray(v, dtype=float).copy()
    return np.convolve(np.asarray(v, dtype=float), np.ones(width) / width, mode="valid")


def local_maxima(v: np.ndarray) -> np.ndarray:
    """Indices of interior local maxima. A plateau counts once, at its first index."""
    v = np.asarray(v, dtype=float)
    if len(v) < 3:
        return np.array([], dtype=int)
    d = np.diff(v)
    idx = []
    i = 1
    while i < len(v) - 1:
        if d[i - 1] > 0:
            j = i
            while j < len(v) - 1 and d[j] == 0:
                j += 1
            if j < len(v) - 1 and d[j] < 0:
                idx.append(i)
            i = j + 1
        else:
            i += 1
    return np.array(idx, dtype=int)


@dataclass(frozen=True)
class Peak:
    theta_ms: float        # argmax of the smoothed signed curve
    value: float           # smoothed curve at the argmax
    censored: bool         # too close to the edge, or the curve keeps rising towards it
    ambiguous: bool        # a second local maximum reaches >= ratio * the main one
    second_ms: float | None
    second_value: float | None


def pick_peak(lags_ms: np.ndarray, v: np.ndarray, *, width: int = 9, max_lag_ms: float | None = None,
              edge_frac: float = 0.05, ambiguity_ratio: float = 0.8, min_sep_ms: float = 50.0,
              min_prominence: float = 0.1) -> tuple[Peak, np.ndarray, np.ndarray]:
    """Smooth, then pick the argmax of the **signed** curve.

    `lags_ms`/`v` must cover the extended range, i.e. (width-1)/2 extra points on each
    side of [-max_lag, max_lag]. Returns (peak, lags_valid, smoothed_valid).
    """
    lags_ms = np.asarray(lags_ms, dtype=float)
    h = (width - 1) // 2
    s = smooth_valid(v, width)
    lv = lags_ms[h:len(lags_ms) - h] if h else lags_ms.copy()
    if max_lag_ms is None:
        max_lag_ms = float(max(abs(lv[0]), abs(lv[-1])))
    k = int(np.argmax(s))
    theta = float(lv[k])
    step = float(np.median(np.diff(lv))) if len(lv) > 1 else 1.0
    guard = max(h * step, edge_frac * max_lag_ms)
    near_edge = (theta - lv[0] <= guard) or (lv[-1] - theta <= guard)
    rising_to_edge = bool(k in (0, len(s) - 1))
    # A competing peak must be a separate hump: at least `min_sep_ms` away, with a dip between
    # it and the main peak of at least `min_prominence` x the main peak height.
    maxima = local_maxima(s)
    cands = []
    for j in maxima:
        if abs(lv[j] - theta) < min_sep_ms:
            continue
        a, b = sorted((j, k))
        dip = float(s[a:b + 1].min())
        if s[k] > 0 and (s[j] - dip) >= min_prominence * s[k]:
            cands.append(j)
    if cands:
        j = max(cands, key=lambda q: s[q])
        second_ms, second_val = float(lv[j]), float(s[j])
        ambiguous = bool(second_val >= ambiguity_ratio * s[k])
    else:
        second_ms = second_val = None
        ambiguous = False
    peak = Peak(theta, float(s[k]), bool(near_edge or rising_to_edge), ambiguous, second_ms, second_val)
    return peak, lv, s


def llr_squared(lags_ms: np.ndarray, rho: np.ndarray, lagset_ms: np.ndarray) -> float:
    """Huth–Abergel LLR: sum rho(l)^2 over l in lagset divided by sum rho(-l)^2.

    Lags are matched to the nearest grid point. `lagset_ms` holds positive values.
    """
    lags_ms = np.asarray(lags_ms, dtype=float)
    pos = [int(np.argmin(np.abs(lags_ms - l))) for l in lagset_ms]
    neg = [int(np.argmin(np.abs(lags_ms + l))) for l in lagset_ms]
    num, den = float(np.sum(rho[pos] ** 2)), float(np.sum(rho[neg] ** 2))
    return num / den if den > 0 else float("nan")


def llr_unsquared_abs(lags_ms: np.ndarray, v: np.ndarray, lagset_ms: np.ndarray) -> float:
    """lead-lag library style: sum |V(l)| over l in lagset divided by sum |V(-l)|."""
    lags_ms = np.asarray(lags_ms, dtype=float)
    pos = [int(np.argmin(np.abs(lags_ms - l))) for l in lagset_ms]
    neg = [int(np.argmin(np.abs(lags_ms + l))) for l in lagset_ms]
    num, den = float(np.sum(np.abs(v[pos]))), float(np.sum(np.abs(v[neg])))
    return num / den if den > 0 else float("nan")


DEFAULT_LLR_LAGSET_MS = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750,
                                  1000, 1500, 2000], dtype=float)


def far_lag_z(lags_ms: np.ndarray, v: np.ndarray, peak_value: float, far_ms: float = 5000.0) -> float:
    """Signal test: (peak - mean of far lags) / sd of far lags, using |lag| >= far_ms.

    Far lags carry no true lead-lag signal under the model, so they give a null.
    Returns NaN if fewer than 20 far lags exist.
    """
    lags_ms = np.asarray(lags_ms, dtype=float)
    far = np.abs(lags_ms) >= far_ms
    if far.sum() < 20:
        return float("nan")
    null = np.asarray(v, dtype=float)[far]
    sd = float(null.std(ddof=1))
    return float((peak_value - null.mean()) / sd) if sd > 0 else float("inf")


def far_band_z(lags_ms: np.ndarray, v: np.ndarray, peak_value: float, far_ms: float = 2000.0) -> float:
    """Peak height above the far-lag pedestal, in units of far-band noise.

    The baseline is the larger of the two far-band means (|lag| >= far_ms on each side), which
    is conservative. The noise is the pooled within-side standard deviation. Unlike a single
    pooled sd, this is not inflated when long intervals create different pedestal levels on the
    two sides. Returns NaN if either side has fewer than 20 points.
    """
    lags_ms = np.asarray(lags_ms, dtype=float)
    v = np.asarray(v, dtype=float)
    left, right = v[lags_ms <= -far_ms], v[lags_ms >= far_ms]
    if len(left) < 20 or len(right) < 20:
        return float("nan")
    base = max(left.mean(), right.mean())
    sd = np.sqrt(((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()) / np.sqrt(len(left) + len(right) - 2)
    return float((peak_value - base) / sd) if sd > 0 else float("inf")
