"""Build per-pair curve stores for one asset over the whole window (Phase 4 input).

Usage: python scripts/build_curves.py --asset WTI [--price mid|micro] [--change-only] [--coarsen-ms 5]
Stores go to <DATA_CURVES>/{asset}/{variant}/{a}__{b}.npz (per study, see llrwa/study.py).
"""
from __future__ import annotations

import argparse
import time
from itertools import combinations

import numpy as np
import pandas as pd

from llrwa.config import DATA_CLEAN, DATA_CURVES, STUDY
from llrwa.hry.pairwise import series_increments
from llrwa.hry.store import build_pair_store

VENUES = ["binance", "okx", "kraken", "hyperliquid"]
T0, T1 = STUDY.T0, STUDY.T1


def load_venue(asset: str, venue: str, segment_gap_s: float = 60.0) -> pd.DataFrame:
    files = sorted(DATA_CLEAN.glob(f"{asset}_{venue}_2026*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("ts_ns", kind="stable")
    df = df.drop_duplicates("ts_ns", keep="last")
    gap = np.diff(df["ts_ns"].to_numpy(), prepend=df["ts_ns"].iloc[0]) / 1e9
    df["segment"] = np.cumsum(gap > segment_gap_s)
    return df.reset_index(drop=True)


def variant_tag(price: str, change_only: bool, coarsen_ms: float | None) -> str:
    return f"{price}_{'chg' if change_only else 'all'}" + (f"_c{coarsen_ms:g}ms" if coarsen_ms else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--price", default="mid")
    ap.add_argument("--change-only", action="store_true")
    ap.add_argument("--coarsen-ms", type=float, default=None)
    ap.add_argument("--max-lag-ms", type=float, default=2500.0)
    a = ap.parse_args()
    tag = variant_tag(a.price, a.change_only, a.coarsen_ms)
    series = {v: load_venue(a.asset, v) for v in VENUES}
    incs = {v: series_increments(df, a.price, a.change_only, a.coarsen_ms) for v, df in series.items() if len(df)}
    out = DATA_CURVES / a.asset / tag
    for x, y in combinations([v for v in VENUES if v in incs], 2):
        t = time.monotonic()
        st = build_pair_store(incs[x], incs[y], T0.value, T1.value, max_lag_ms=a.max_lag_ms)
        st.save(out / f"{x}__{y}.npz")
        print(f"{a.asset} {tag} {x}-{y}: n={len(incs[x])},{len(incs[y])} blocks={len(st.sx2)} {time.monotonic() - t:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
