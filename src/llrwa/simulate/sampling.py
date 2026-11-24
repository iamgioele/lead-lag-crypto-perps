"""Observation-time schemes on [0, T) in milliseconds (float).

All schemes are stationary: renewal processes start from a long burn-in, and grids get a
uniformly random phase unless one is given.
"""
from __future__ import annotations

import numpy as np


def poisson(rate_per_s: float, T_ms: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.poisson(rate_per_s * T_ms / 1e3)
    return np.sort(rng.uniform(0.0, T_ms, n))


def gamma_renewal(rate_per_s: float, cv: float, T_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Renewal process with Gamma inter-arrivals of mean 1/rate and coefficient of variation `cv`.

    cv > 1 is bursty, cv = 1 is Poisson.
    """
    mean = 1e3 / rate_per_s
    shape = 1.0 / cv ** 2
    burn = 50 * mean * max(1.0, cv ** 2)
    n = int((T_ms + burn) / mean * 1.5) + 100
    t = np.cumsum(rng.gamma(shape, mean / shape, n)) - burn
    while t[-1] < T_ms:                       # extend if needed
        t = np.concatenate([t, t[-1] + np.cumsum(rng.gamma(shape, mean / shape, n))])
    return t[(t >= 0) & (t < T_ms)]


def grid(period_ms: float, T_ms: float, rng: np.random.Generator | None = None,
         phase_ms: float | None = None) -> np.ndarray:
    """Deterministic snapshot grid. The phase is uniform on [0, period) unless given (locked grids)."""
    if phase_ms is None:
        phase_ms = float(rng.uniform(0, period_ms)) if rng is not None else 0.0
    return np.arange(phase_ms, T_ms, period_ms)


def triggered(leader_ms: np.ndarray, delay_ms: float, jitter_ms: float, share: float, background_rate_per_s: float,
              T_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Follower update times: a fraction `share` of leader updates trigger a follower update after
    delay + |N(0, jitter)| ms, plus independent Poisson background updates."""
    pick = leader_ms[rng.uniform(size=len(leader_ms)) < share]
    trig = pick + delay_ms + np.abs(rng.normal(0.0, jitter_ms, len(pick)))
    t = np.concatenate([trig, poisson(background_rate_per_s, T_ms, rng)])
    t = np.sort(t[(t >= 0) & (t < T_ms)])
    return t
