"""Phase 5 robustness for one asset: data knobs, estimator knobs, and the feed-resolution test.

Usage: python scripts/robustness.py --asset WTI
Data knobs require the variant stores (build_curves.py --price micro / --change-only /
--coarsen-ms). Estimator knobs post-process the default store. Output:
results/tables/robustness_{asset}.csv and results/tables/feedtest_{asset}.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from llrwa.config import DATA_CLEAN, DATA_CURVES, RESULTS, STUDY
from llrwa.hry.engine import NS_PER_MS, increments
from llrwa.hry.pairwise import estimate_pairs
from llrwa.hry.peaks import pick_peak
from llrwa.hry.store import PairStore

VENUES = ["binance", "okx", "kraken", "hyperliquid"]
DEFAULT = dict(variant="mid_all", width=9, max_lag_ms=2500.0, step=1, window="pooled", min_n=200)


def pairs_of(asset: str, variant: str):
    base = DATA_CURVES / asset / variant
    for f in sorted(base.glob("*.npz")):
        x, y = f.stem.split("__")
        yield x, y, PairStore.load(f)


def peak_with(st: PairStore, mask, width: int, max_lag_ms: float, step: int):
    rho, _ = st.curve(mask)
    lags = st.lags_ms
    keep = np.abs(lags) <= max_lag_ms + (width - 1) // 2 * step + 1e-9
    lags, rho = lags[keep], rho[keep]
    if step > 1:
        lags, rho = lags[::step], rho[::step]
    return pick_peak(lags, rho, width=width, max_lag_ms=max_lag_ms)[0]


def window_masks(st: PairStore, kind: str, min_n: int):
    n = len(st.sx2)
    if kind == "pooled":
        return [np.ones(n, dtype=bool)]
    size = {"10min": 2, "1h": 12, "1d": 288}[kind]
    out = []
    for s0 in range(0, n, size):
        m = np.zeros(n, dtype=bool)
        m[s0:s0 + size] = True
        if st.nx[m].sum() >= min_n and st.ny[m].sum() >= min_n:
            out.append(m)
    return out


def run_setting(asset: str, knob: str, value, **kw) -> list[dict]:
    p = {**DEFAULT, **kw}
    rows = []
    for x, y, st in pairs_of(asset, p["variant"]):
        ests = []
        for m in window_masks(st, p["window"], p["min_n"]):
            pk = peak_with(st, m, p["width"], p["max_lag_ms"], p["step"])
            if not pk.censored:
                ests.append(pk.theta_ms)
        e = np.array(ests)
        rows.append({"asset": asset, "knob": knob, "value": str(value), "leader_candidate": x, "follower_candidate": y,
                     "n_windows": len(e), "theta_median_ms": float(np.median(e)) if len(e) else np.nan,
                     "theta_iqr_ms": float(np.subtract(*np.percentile(e, [75, 25]))) if len(e) > 1 else np.nan})
    return rows


def feed_test(asset: str, day: str = STUDY.ref_day.replace("-", "")) -> pd.DataFrame:
    """Binance vs degraded copies of itself, and each degraded copy vs the other venues."""
    b = pd.read_parquet(DATA_CLEAN / f"{asset}_binance_{day}.parquet")
    t, mid = b["ts_ns"].to_numpy(np.int64), b["mid"].to_numpy(float)
    step = 100 * NS_PER_MS
    grid = np.arange((t[0] // step + 1) * step, t[-1], step, dtype=np.int64)
    idx = np.searchsorted(t, grid, side="right") - 1
    grid_stamped = pd.DataFrame({"ts_ns": grid, "mid": mid[idx], "segment": 0})
    cell = t // step
    last = np.r_[cell[1:] != cell[:-1], True]
    thinned = pd.DataFrame({"ts_ns": t[last], "mid": mid[last], "segment": 0})
    stale_idx = np.searchsorted(t, grid - 65 * NS_PER_MS, side="right") - 1
    stale = pd.DataFrame({"ts_ns": grid[stale_idx >= 0], "mid": mid[stale_idx[stale_idx >= 0]], "segment": 0})
    series = {"binance": b[["ts_ns", "mid"]].assign(segment=0), "b100_grid": grid_stamped, "b100_thin": thinned,
              "b100_grid_stale65": stale}
    for v in ("okx", "kraken"):
        f = DATA_CLEAN / f"{asset}_{v}_{day}.parquet"
        if f.exists():
            series[v] = pd.read_parquet(f)[["ts_ns", "mid", "segment"]]
    res, _ = estimate_pairs(series, max_lag_ms=1000)
    keep = res["leader_candidate"].isin(["binance", "b100_grid", "b100_thin", "b100_grid_stale65"])
    return res[keep][["leader_candidate", "follower_candidate", "theta_ms", "peak_rho", "censored", "ambiguous"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    a = ap.parse_args()
    rows = []
    rows += run_setting(a.asset, "baseline", "default")
    for w in (1, 5, 9, 21):
        rows += run_setting(a.asset, "smoothing_width", w, width=w)
    for L in (250.0, 1000.0, 2500.0):
        rows += run_setting(a.asset, "max_lag_ms", L, max_lag_ms=L)
    for s in (1, 5):
        rows += run_setting(a.asset, "lag_step_ms", s, step=s)
    for win in ("10min", "1h", "1d", "pooled"):
        rows += run_setting(a.asset, "window", win, window=win)
    for mn in (200, 500, 1000):
        rows += run_setting(a.asset, "min_increments", mn, window="1h", min_n=mn)
    for var in ("micro_all", "mid_chg", "mid_all_c5ms", "mid_all_c10ms"):
        if (DATA_CURVES / a.asset / var).exists():
            rows += run_setting(a.asset, "data_variant", var, variant=var)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "tables" / f"robustness_{a.asset}.csv", index=False)
    wide = df.pivot_table(index=["knob", "value"], columns=["leader_candidate", "follower_candidate"],
                          values="theta_median_ms", aggfunc="first", sort=False)
    pd.set_option("display.width", 250)
    print(wide.round(1).to_string())
    ft = feed_test(a.asset)
    ft.to_csv(RESULTS / "tables" / f"feedtest_{a.asset}.csv", index=False)
    print("\nFeed-resolution test (reference day):")
    print(ft.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
