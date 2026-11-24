"""Lead-lag price model sampled exactly at the observation times.

X_t = sigma W_t and Y_t = sigma (rho W_{t - theta} + sqrt(1 - rho^2) B_t), with times in ms.
With theta > 0, X leads Y. W is simulated only at the merged times {t_i} ∪ {s_j - theta}
and B only at {s_j}, so there is no discretisation grid and no approximation.
"""
from __future__ import annotations

import numpy as np

NS_PER_MS = 1_000_000


def _bm_at(times: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Brownian motion (variance sigma^2 per ms) at arbitrary times, in the original order."""
    o = np.argsort(times, kind="stable")
    ts = times[o]
    start = min(0.0, ts[0]) if len(ts) else 0.0
    inc = rng.normal(size=len(ts)) * sigma * np.sqrt(np.diff(np.concatenate([[start], ts])))
    out = np.empty(len(ts))
    out[o] = np.cumsum(inc)
    return out


def lead_lag_pair(tx_ms: np.ndarray, ty_ms: np.ndarray, theta_ms: float, rho: float, sigma_per_sqrt_ms: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Log-prices of X at tx and of Y at ty (both relative to 0)."""
    w = _bm_at(np.concatenate([tx_ms, ty_ms - theta_ms]), sigma_per_sqrt_ms, rng)
    x, wy = w[:len(tx_ms)], w[len(tx_ms):]
    b = _bm_at(ty_ms, sigma_per_sqrt_ms, rng)
    return x, rho * wy + np.sqrt(1.0 - rho ** 2) * b


def observe(t_ms: np.ndarray, logp: np.ndarray, *, p0: float = 100.0, tick: float | None = None,
            noise_sd: float = 0.0, rng: np.random.Generator | None = None,
            ms_timestamps: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Turn a latent log-price path into observed (t_ns, log price).

    Options: add i.i.d. noise in log units, round the price level to a tick, truncate
    timestamps to whole milliseconds. Equal timestamps keep the last observation.
    """
    lp = np.asarray(logp, dtype=float).copy()
    if noise_sd > 0:
        lp = lp + rng.normal(0.0, noise_sd, len(lp))
    if tick:
        lp = np.log(np.round(p0 * np.exp(lp) / tick) * tick / p0)
    t = np.floor(t_ms) if ms_timestamps else np.asarray(t_ms, dtype=float)
    t_ns = np.round(t * NS_PER_MS).astype(np.int64)
    keep = np.ones(len(t_ns), dtype=bool)
    keep[:-1] = t_ns[1:] != t_ns[:-1]
    return t_ns[keep], lp[keep]
