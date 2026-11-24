"""All-pairs lead-lag estimation for one asset and window (Phase 4).

Convention: for the pair (i, j), theta > 0 means venue i leads venue j.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .engine import Increments, LagGrid, NS_PER_MS, contrast, increments
from .peaks import DEFAULT_LLR_LAGSET_MS, llr_squared, llr_unsquared_abs, pick_peak


@dataclass
class PairResult:
    leader_candidate: str      # venue i
    follower_candidate: str    # venue j
    theta_ms: float            # > 0: i leads j
    peak_rho: float            # smoothed normalised contrast at the peak (tick-level normalisation)
    censored: bool
    ambiguous: bool
    second_ms: float | None
    llr_ha: float              # squared, on the fixed lag set
    llr_lib: float             # unsquared |V| ratio, on the fixed lag set
    n_i: int
    n_j: int


def series_increments(df: pd.DataFrame, price: str = "mid", change_only: bool = False,
                      coarsen_ms: float | None = None) -> Increments:
    """Log-price increments from a clean frame (ts_ns, mid/micro, segment)."""
    t = df["ts_ns"].to_numpy(np.int64)
    p = np.log(df[price].to_numpy(float))
    seg = df["segment"].to_numpy() if "segment" in df else None
    if coarsen_ms:
        step = int(round(coarsen_ms * NS_PER_MS))
        t = (t + step // 2) // step * step               # round to nearest, not floor (DECISIONS D8)
        keep = np.ones(len(t), dtype=bool)
        keep[:-1] = t[1:] != t[:-1]                      # keep last per coarsened stamp
        t, p = t[keep], p[keep]
        seg = None if seg is None else seg[keep]
    return increments(t, p, change_only=change_only, segment=seg)


def pair_curve(X: Increments, Y: Increments, max_lag_ms: float = 2500, step_ms: float = 1.0,
               width: int = 9, half_step: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Normalised signed contrast on an extended grid (±max_lag plus the smoothing margin)."""
    margin = (width - 1) // 2 * step_ms
    grid = LagGrid.symmetric(int((max_lag_ms + margin) * NS_PER_MS), int(step_ms * NS_PER_MS), half_step=half_step)
    v = contrast(X, Y, grid)
    norm = np.sqrt(X.sum_sq * Y.sum_sq)
    return grid.lags_ms, (v / norm if norm > 0 else v)


def estimate_pairs(series: dict[str, pd.DataFrame], *, price: str = "mid", change_only: bool = False,
                   max_lag_ms: float = 2500, step_ms: float = 1.0, width: int = 9,
                   coarsen_ms: float | None = None, llr_lagset_ms=DEFAULT_LLR_LAGSET_MS
                   ) -> tuple[pd.DataFrame, dict]:
    incs = {v: series_increments(df, price, change_only, coarsen_ms) for v, df in series.items()}
    rows, curves = [], {}
    for a, b in combinations(list(series), 2):
        lags, rho = pair_curve(incs[a], incs[b], max_lag_ms, step_ms, width)
        pk, lv, s = pick_peak(lags, rho, width=width, max_lag_ms=max_lag_ms)
        rows.append(asdict(PairResult(a, b, pk.theta_ms, pk.value, pk.censored, pk.ambiguous, pk.second_ms,
                                      llr_squared(lv, s, llr_lagset_ms), llr_unsquared_abs(lv, s, llr_lagset_ms),
                                      len(incs[a]), len(incs[b]))))
        curves[(a, b)] = (lv, s)
    return pd.DataFrame(rows), curves


def lag_matrix(res: pd.DataFrame, venues: list[str], value: str = "theta_ms") -> pd.DataFrame:
    """Antisymmetric lag matrix (row leads column when positive). Censored pairs are NaN."""
    M = pd.DataFrame(np.nan, index=venues, columns=venues)
    for _, r in res.iterrows():
        x = np.nan if r["censored"] else r[value]
        M.loc[r["leader_candidate"], r["follower_candidate"]] = x
        M.loc[r["follower_candidate"], r["leader_candidate"]] = -x if value == "theta_ms" else x
    return M


def venue_offsets(res: pd.DataFrame, venues: list[str], ref: str) -> tuple[pd.Series, pd.DataFrame]:
    """Least-squares offsets tau with theta_ij ≈ tau_j - tau_i and tau_ref = 0. Uses uncensored pairs.

    Returns tau (ms; a larger tau reacts later) and the residual table.
    """
    use = res[~res["censored"]]
    idx = {v: k for k, v in enumerate(venues)}
    A = np.zeros((len(use) + 1, len(venues)))
    y = np.zeros(len(use) + 1)
    for r, (_, row) in enumerate(use.iterrows()):
        A[r, idx[row["follower_candidate"]]] = 1.0
        A[r, idx[row["leader_candidate"]]] = -1.0
        y[r] = row["theta_ms"]
    A[-1, idx[ref]] = 1.0                               # constraint tau_ref = 0
    tau, *_ = np.linalg.lstsq(A, y, rcond=None)
    # venues not connected to `ref` through the used pairs have no identified offset
    reach, frontier = {ref}, [ref]
    edges = list(zip(use["leader_candidate"], use["follower_candidate"]))
    while frontier:
        v = frontier.pop()
        for a_, b_ in edges:
            for u, w in ((a_, b_), (b_, a_)):
                if u == v and w not in reach:
                    reach.add(w)
                    frontier.append(w)
    tau = np.array([t if v in reach else np.nan for v, t in zip(venues, tau)])
    fit = A[:-1] @ np.nan_to_num(tau)
    resid = use[["leader_candidate", "follower_candidate", "theta_ms"]].copy()
    resid["fitted_ms"] = fit
    resid["residual_ms"] = resid["theta_ms"] - fit
    return pd.Series(tau, index=venues, name="tau_ms"), resid
