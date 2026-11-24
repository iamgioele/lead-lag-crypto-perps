"""Monte Carlo runner for the simulation study (Phase 7, RQ4).

A setting is a dict. One repetition simulates X, Y with true lag theta, observes them
(sampling, rounding, noise, staleness), and records HRY and baseline estimates.

Sampling specs: ("poisson", rate/s) | ("gamma", rate/s, cv) | ("grid", period_ms) |
("grid_locked", period_ms, phase_ms). A setting may add `stale_y_ms` (Y's snapshot shows the
price as of t - stale) and a price observation model: `tick_bp` (relative tick size),
`noise_bp` (i.i.d. noise sd), `change_only`.
"""
from __future__ import annotations

import numpy as np

from ..hry.engine import increments
from ..hry.pairwise import pair_curve
from ..hry.peaks import DEFAULT_LLR_LAGSET_MS, llr_squared, llr_unsquared_abs, pick_peak
from . import baselines, prices, sampling

BASE = dict(T_min=60.0, theta_ms=50.0, rho=0.8, vol_bp_per_sqrt_s=1.0, x=("poisson", 10.0), y=("poisson", 5.0),
            tick_bp=None, noise_bp=0.0, change_only=False, stale_y_ms=0.0, max_lag_ms=1000.0, width=9,
            xcorr_bars_ms=())


def _times(spec, T_ms, rng):
    kind = spec[0]
    if kind == "poisson":
        return sampling.poisson(spec[1], T_ms, rng)
    if kind == "gamma":
        return sampling.gamma_renewal(spec[1], spec[2], T_ms, rng)
    if kind == "grid":
        return sampling.grid(spec[1], T_ms, rng)
    if kind == "grid_locked":
        return sampling.grid(spec[1], T_ms, phase_ms=spec[2])
    raise KeyError(kind)


def one_rep(setting: dict, seed: int) -> dict:
    s = {**BASE, **setting}
    rng = np.random.default_rng(seed)
    T_ms = s["T_min"] * 60_000.0
    tx = _times(s["x"], T_ms, rng)
    ty = _times(s["y"], T_ms, rng)
    sigma = s["vol_bp_per_sqrt_s"] * 1e-4 / np.sqrt(1000.0)          # per sqrt(ms)
    # staleness: Y's observation at s shows the price as of s - stale
    x, y = prices.lead_lag_pair(tx, ty - s["stale_y_ms"], s["theta_ms"], s["rho"], sigma, rng)
    tick = None if s["tick_bp"] is None else 100.0 * s["tick_bp"] * 1e-4
    ox = prices.observe(tx, x, tick=tick, noise_sd=s["noise_bp"] * 1e-4, rng=rng)
    oy = prices.observe(ty, y, tick=tick, noise_sd=s["noise_bp"] * 1e-4, rng=rng)
    X = increments(*ox, change_only=s["change_only"])
    Y = increments(*oy, change_only=s["change_only"])
    out = {"seed": seed, "n_x": len(X), "n_y": len(Y)}
    if len(X) < 10 or len(Y) < 10:
        return {**out, "theta_hat": np.nan}
    lags, rho = pair_curve(X, Y, s["max_lag_ms"], 1.0, s["width"])
    pk, lv, sm = pick_peak(lags, rho, width=s["width"], max_lag_ms=s["max_lag_ms"])
    lagset = DEFAULT_LLR_LAGSET_MS[DEFAULT_LLR_LAGSET_MS <= s["max_lag_ms"]]
    out.update(theta_hat=pk.theta_ms, peak=pk.value, censored=pk.censored, ambiguous=pk.ambiguous,
               llr_ha=llr_squared(lv, sm, lagset), llr_lib=llr_unsquared_abs(lv, sm, lagset))
    for bar in s["xcorr_bars_ms"]:
        bl, c = baselines.xcorr_grid(ox[0], ox[1], oy[0], oy[1], bar_ms=bar, max_lag_ms=s["max_lag_ms"])
        w = max(1, int(round(9 / bar))) | 1
        cs = np.convolve(c, np.ones(w) / w, "same")
        out[f"xc{bar:g}_argmax"] = float(bl[np.argmax(cs)])
        pos, neg = c[bl > 0].clip(0).sum(), c[bl < 0].clip(0).sum()
        out[f"xc{bar:g}_centroid"] = float(np.sum(bl * c) / np.sum(c)) if np.sum(c) != 0 else np.nan
        out[f"xc{bar:g}_mass_ratio"] = float(pos / neg) if neg > 0 else np.nan
    return out


def summarise(df, theta_ms: float) -> dict:
    err = df["theta_hat"] - theta_ms
    ok = err.notna()
    e = err[ok]
    return {"reps": int(ok.sum()), "bias_ms": float(e.mean()), "median_bias_ms": float(e.median()),
            "rmse_ms": float(np.sqrt((e ** 2).mean())), "mae_ms": float(e.abs().mean()),
            "p_within_1ms": float((e.abs() <= 1).mean()), "p_within_5ms": float((e.abs() <= 5).mean()),
            "p_within_10ms": float((e.abs() <= 10).mean()), "q05_ms": float(e.quantile(0.05)),
            "q95_ms": float(e.quantile(0.95)), "censored_rate": float(df["censored"].fillna(False).mean()),
            "llr_ha_median": float(df["llr_ha"].median()), "n_x_mean": float(df["n_x"].mean()),
            "n_y_mean": float(df["n_y"].mean())}
