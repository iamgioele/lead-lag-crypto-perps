"""Shifted Hayashi–Yoshida contrast for all lags at once.

For X-intervals I_i = (a0_i, a1_i] with increments dx_i and Y-intervals J_j = (b0_j, b1_j]
with increments dy_j, the signed contrast at lag theta is

    V(theta) = sum_{i,j} dx_i dy_j 1{ I_i ∩ (J_j - theta) != ∅ }.

theta > 0 means X leads Y: Y reacts at time t + theta to information that X shows at t.
A pair (i, j) contributes exactly for theta in the open interval
(b0_j - a1_i, b1_j - a0_i), so V on a lag grid is a sum of boxes. We enumerate the pairs
whose box meets the grid range and accumulate them in a difference array.
The cost is O(n + m + K + |G|), where K is the number of such pairs.

All times and lags are int64 nanoseconds. Nothing is snapped to a grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class Increments:
    """Price increments over half-open intervals (start, end], sorted and non-overlapping."""

    start: np.ndarray  # int64 ns
    end: np.ndarray    # int64 ns
    dx: np.ndarray     # float64 (log-price increments)

    def __post_init__(self) -> None:
        if not (len(self.start) == len(self.end) == len(self.dx)):
            raise ValueError("start, end and dx must have equal length")

    def __len__(self) -> int:
        return len(self.dx)

    @property
    def sum_sq(self) -> float:
        return float(np.dot(self.dx, self.dx))


def increments(t_ns: np.ndarray, logp: np.ndarray, *, change_only: bool = False,
               segment: np.ndarray | None = None, keep_zero: bool = False) -> Increments:
    """Build increment intervals from observation times and log prices.

    `t_ns` must be sorted, with unique values; dedupe equal timestamps beforehand
    (keep last).
    - `change_only=False` (default): intervals run between consecutive observations,
      whatever they are (all quote updates). Zero increments are dropped unless
      `keep_zero`. They contribute nothing to V, but dropping them keeps each nonzero
      interval's true left end.
    - `change_only=True`: first keep only the observations where the price differs
      from the previous one, then form intervals between those. This is the
      "keep only price changes" rule, and its intervals are longer.
    - `segment`: optional integer id per observation. Increments that would cross a
      segment boundary (an outage) are dropped.
    """
    t = np.asarray(t_ns, dtype=np.int64)
    p = np.asarray(logp, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError("t_ns and logp must have the same shape")
    if len(t) and np.any(np.diff(t) <= 0):
        raise ValueError("t_ns must be strictly increasing (dedupe equal timestamps first)")
    seg = None if segment is None else np.asarray(segment)
    if change_only and len(p):
        keep = np.ones(len(p), dtype=bool)
        keep[1:] = p[1:] != p[:-1]
        if seg is not None:
            keep[1:] |= seg[1:] != seg[:-1]      # always keep the first obs of a segment
        t, p = t[keep], p[keep]
        seg = None if seg is None else seg[keep]
    if len(t) < 2:
        e = np.empty(0, dtype=np.int64)
        return Increments(e, e.copy(), np.empty(0, dtype=np.float64))
    dx = np.diff(p)
    ok = np.isfinite(dx)
    if seg is not None:
        ok &= seg[1:] == seg[:-1]
    if not keep_zero:
        ok &= dx != 0.0
    return Increments(t[:-1][ok], t[1:][ok], dx[ok])


@dataclass(frozen=True)
class LagGrid:
    """Lags theta_k = offset + k*step (ns), k = 0..n-1."""

    offset: int
    step: int
    n: int

    @classmethod
    def symmetric(cls, max_lag_ns: int, step_ns: int, half_step: bool = False) -> "LagGrid":
        """Grid covering [-max_lag, max_lag].

        With `half_step=True` the grid sits at (k + 1/2)*step, which avoids exact ties
        with integer-millisecond timestamp differences.
        """
        if step_ns <= 0 or max_lag_ns < 0:
            raise ValueError("step must be > 0 and max_lag >= 0")
        if half_step:
            if step_ns % 2:
                raise ValueError("half_step needs an even step in ns")
            k = int(np.ceil((max_lag_ns - step_ns // 2) / step_ns))
            return cls(offset=-(k * step_ns + step_ns // 2), step=step_ns, n=2 * k + 2)
        k = max_lag_ns // step_ns
        return cls(offset=-k * step_ns, step=step_ns, n=2 * k + 1)

    @property
    def lags(self) -> np.ndarray:
        return self.offset + self.step * np.arange(self.n, dtype=np.int64)

    @property
    def lags_ms(self) -> np.ndarray:
        return self.lags / NS_PER_MS


def _pair_ranges(X: Increments, Y: Increments, lo_lag: int, hi_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """For each X interval, the half-open range [lo, hi) of Y intervals that can overlap it
    for some lag in [lo_lag, hi_lag]: b1_j > a0_i + lo_lag and b0_j < a1_i + hi_lag."""
    lo = np.searchsorted(Y.end, X.start + lo_lag, side="right")
    hi = np.searchsorted(Y.start, X.end + hi_lag, side="left")
    return lo, np.maximum(hi, lo)


def contrast(X: Increments, Y: Increments, grid: LagGrid, *, block_of_x: np.ndarray | None = None,
             n_blocks: int | None = None, max_pairs: int = 4_000_000) -> np.ndarray:
    """Signed contrast V on `grid`, shape (n,), or (n_blocks, n) if `block_of_x` is given.

    `block_of_x[i]` assigns X interval i to a time block. Summing the block rows gives the
    full curve exactly. Pairs are processed in chunks of at most about `max_pairs`.
    """
    n = grid.n
    if len(X) == 0 or len(Y) == 0:
        return np.zeros((n_blocks, n) if block_of_x is not None else n)
    if np.any(np.diff(Y.start) < 0) or np.any(Y.start[1:] < Y.end[:-1]):
        raise ValueError("Y intervals must be sorted and non-overlapping")
    lo_lag = grid.offset
    hi_lag = grid.offset + grid.step * (n - 1)
    lo, hi = _pair_ranges(X, Y, lo_lag, hi_lag)
    cnt = hi - lo
    blocks = block_of_x is not None
    if blocks:
        block_of_x = np.asarray(block_of_x, dtype=np.int64)
        if n_blocks is None:
            n_blocks = int(block_of_x.max()) + 1
        acc = np.zeros(n_blocks * (n + 1))
    else:
        acc = np.zeros(n + 1)
    csum = np.cumsum(cnt)
    starts = np.concatenate([[0], np.searchsorted(csum, np.arange(max_pairs, csum[-1], max_pairs), side="left")])
    bounds = list(starts) + [len(cnt)]
    for c0, c1 in zip(bounds[:-1], bounds[1:]):
        if c1 <= c0:
            continue
        cc = cnt[c0:c1]
        tot = int(cc.sum())
        if tot == 0:
            continue
        i = np.repeat(np.arange(c0, c1), cc)
        first = np.repeat(np.cumsum(cc) - cc, cc)
        j = np.repeat(lo[c0:c1], cc) + (np.arange(tot) - first)
        left = Y.start[j] - X.end[i]       # open interval (left, right) of lags
        right = Y.end[j] - X.start[i]
        k0 = np.floor_divide(left - grid.offset, grid.step) + 1
        k1 = -np.floor_divide(grid.offset - right, grid.step) - 1
        k0 = np.maximum(k0, 0)
        k1 = np.minimum(k1, n - 1)
        ok = k0 <= k1
        if not ok.any():
            continue
        w = X.dx[i[ok]] * Y.dx[j[ok]]
        k0, k1 = k0[ok], k1[ok] + 1
        if blocks:
            base = block_of_x[i[ok]] * (n + 1)
            k0, k1 = base + k0, base + k1
        size = acc.shape[0]
        acc += np.bincount(k0, weights=w, minlength=size)[:size]
        acc -= np.bincount(k1, weights=w, minlength=size)[:size]
    if blocks:
        return np.cumsum(acc.reshape(n_blocks, n + 1), axis=1)[:, :n]
    return np.cumsum(acc)[:n]


def contrast_per_lag(X: Increments, Y: Increments, lags_ns: np.ndarray) -> np.ndarray:
    """Reference implementation: one vectorised pass per lag, O(n log m) each.

    Uses the same interval semantics as `contrast`, with no assumption that intervals are
    contiguous.
    """
    out = np.empty(len(lags_ns))
    for q, th in enumerate(np.asarray(lags_ns, dtype=np.int64)):
        lo, hi = _pair_ranges(X, Y, int(th), int(th))
        cnt = hi - lo
        tot = int(cnt.sum())
        if tot == 0:
            out[q] = 0.0
            continue
        i = np.repeat(np.arange(len(X)), cnt)
        j = np.repeat(lo, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        ov = (Y.start[j] - th < X.end[i]) & (Y.end[j] - th > X.start[i])
        out[q] = float(np.sum(X.dx[i[ov]] * Y.dx[j[ov]]))
    return out


def contrast_bruteforce(X: Increments, Y: Increments, lags_ns: np.ndarray) -> np.ndarray:
    """O(n m) definition, for tests on small inputs only."""
    a0, a1 = X.start[:, None], X.end[:, None]
    w = X.dx[:, None] * Y.dx[None, :]
    out = []
    for th in np.asarray(lags_ns, dtype=np.int64):
        b0, b1 = Y.start[None, :] - th, Y.end[None, :] - th
        out.append(float(np.sum(w * (np.maximum(a0, b0) < np.minimum(a1, b1)))))
    return np.array(out)
