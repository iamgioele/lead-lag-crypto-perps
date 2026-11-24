"""Moving-block bootstrap on stored per-block curves, shared across pairs (joint resampling).

Blocks (5 min) inside the selected scope are grouped into consecutive superblocks
(default 6 blocks = 30 min) that are resampled with replacement. The same draws are used
for every pair, so venue offsets can be refitted per replicate. HRY's estimator has no
limit law (Proposition 2 of HRY 2013), so these intervals are a secondary, heuristic
measure; their coverage is checked by simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .peaks import pick_peak
from .store import PairStore


def superblocks(mask: np.ndarray, size: int) -> list[np.ndarray]:
    idx = np.flatnonzero(mask)
    return [idx[i:i + size] for i in range(0, len(idx), size)]


def bootstrap_pairs(stores: dict[tuple[str, str], PairStore], mask: np.ndarray, *, B: int = 500, size: int = 6,
                    seed: int = 0) -> pd.DataFrame:
    """Return one row per (replicate, pair) with theta* and the censored/ambiguous flags."""
    rng = np.random.default_rng(seed)
    sbs = superblocks(mask, size)
    nsb = len(sbs)
    counts = rng.multinomial(nsb, np.full(nsb, 1.0 / nsb), size=B)       # (B, nsb) resampling weights
    rows = []
    for (x, y), st in stores.items():
        SV = np.stack([st.V[s].sum(axis=0, dtype=np.float64) for s in sbs])   # (nsb, n_lags)
        sx = np.array([st.sx2[s].sum() for s in sbs])
        sy = np.array([st.sy2[s].sum() for s in sbs])
        V = counts @ SV
        norm = np.sqrt((counts @ sx) * (counts @ sy))
        for b in range(B):
            pk, _, _ = pick_peak(st.lags_ms, V[b] / norm[b], width=st.width, max_lag_ms=st.max_lag_ms)
            rows.append({"rep": b, "leader_candidate": x, "follower_candidate": y, "theta_ms": pk.theta_ms,
                         "censored": pk.censored, "ambiguous": pk.ambiguous})
    return pd.DataFrame(rows)


def summarise_bootstrap(bs: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    g = bs.groupby(["leader_candidate", "follower_candidate"])["theta_ms"]
    out = pd.DataFrame({"ci_lo_ms": g.quantile(lo), "ci_hi_ms": g.quantile(hi), "boot_sd_ms": g.std(),
                        "share_censored": bs.groupby(["leader_candidate", "follower_candidate"])["censored"].mean()})
    return out.reset_index()


def mode_shares(bs: pd.DataFrame, split_ms: float) -> pd.DataFrame:
    """For bimodal pairs: share of replicates on each side of `split_ms`."""
    g = bs.groupby(["leader_candidate", "follower_candidate"])["theta_ms"]
    return g.apply(lambda s: pd.Series({"share_below": float((s < split_ms).mean()),
                                        "share_above": float((s >= split_ms).mean())})).unstack().reset_index()
