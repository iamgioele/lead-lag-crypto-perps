"""Expected shape of the shifted Hayashi–Yoshida contrast (candidate proposition, chapter 3).

Take stationary sampling schemes that are independent of each other and of the prices.
Then E V(theta + h) ≈ sigma^2 rho T k(h), with k(h) = P(A + D > |h|). Here A is the
backward waiting time of X's sampling and D the forward waiting time of Y's sampling at
a typical instant (by stationarity, backward and forward times have the same law).
"""
from __future__ import annotations

import numpy as np


def k_poisson(h_ms: np.ndarray, lam_x_per_s: float, lam_y_per_s: float) -> np.ndarray:
    """Survival function of Exp(lam_x) + Exp(lam_y), evaluated at |h|."""
    h = np.abs(np.asarray(h_ms, dtype=float)) / 1e3
    l, m = lam_x_per_s, lam_y_per_s
    if np.isclose(l, m):
        return (1 + l * h) * np.exp(-l * h)
    return (m * np.exp(-l * h) - l * np.exp(-m * h)) / (m - l)


def k_poisson_grid(h_ms: np.ndarray, lam_x_per_s: float, period_ms: float) -> np.ndarray:
    """X Poisson(lam), Y on a grid with independent uniform phase: P(Exp(lam) + U(0, D) > |h|)."""
    h = np.abs(np.asarray(h_ms, dtype=float)) / 1e3
    lam, D = lam_x_per_s, period_ms / 1e3
    inside = ((1 - np.exp(-lam * np.minimum(h, D))) / lam + (D - np.minimum(h, D))) / D
    beyond = np.exp(-lam * h) * (np.exp(lam * D) - 1) / (lam * D)
    return np.where(h <= D, inside, beyond)


def k_empirical(tx_ms: np.ndarray, ty_ms: np.ndarray, theta_ms: float, h_ms: np.ndarray,
                n_points: int = 20000, rng: np.random.Generator | None = None) -> np.ndarray:
    """k(h) estimated from the sampling times alone, via recurrence times at random instants."""
    rng = rng or np.random.default_rng(0)
    tys = np.asarray(ty_ms) - theta_ms
    lo, hi = max(tx_ms[0], tys[0]), min(tx_ms[-1], tys[-1])
    u = rng.uniform(lo, hi, n_points)
    a = u - tx_ms[np.searchsorted(tx_ms, u, "right") - 1]      # X backward
    b = tx_ms[np.searchsorted(tx_ms, u, "left")] - u           # X forward
    c = u - tys[np.searchsorted(tys, u, "right") - 1]          # Y backward
    d = tys[np.searchsorted(tys, u, "left")] - u               # Y forward
    h = np.asarray(h_ms, dtype=float)
    pos = np.array([np.mean(a + d > x) for x in np.abs(h)])
    neg = np.array([np.mean(b + c > x) for x in np.abs(h)])
    return np.where(h >= 0, pos, neg)


def previous_tick_kernel_samples(d_x_ms: np.ndarray, d_y_ms: np.ndarray, theta_ms: float = 0.0) -> np.ndarray:
    """Samples whose density is the expected shape of the fine-bar previous-tick cross-correlation.

    With previous-tick returns on fine bars, corr(r_x[t], r_y[t+L]) sums covariances of
    interval pairs whose right end points differ by L. The expected curve is therefore the
    density of theta + D_Y - D_X, where D_X and D_Y are the forward waiting times at a
    typical instant: a difference kernel, not the symmetric sum kernel of HRY.
    """
    return theta_ms + np.asarray(d_y_ms) - np.asarray(d_x_ms)


def previous_tick_centroid_shift_ms(mean_wait_x_ms: float, mean_wait_y_ms: float) -> float:
    """The previous-tick curve's centroid sits at theta + E[D_Y] - E[D_X], but its mode can stay at theta.

    So the baseline mostly biases mass-based measures (LLR, centroid): the faster venue
    appears to lead. Poisson(lam) gives E[D] = 1000/lam ms; a grid of spacing D gives D/2.
    """
    return mean_wait_y_ms - mean_wait_x_ms
