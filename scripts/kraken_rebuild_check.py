"""Check the rebuilt Kraken best quotes against Amberdata's ~1 s Kraken tickers on the reference day.

Usage: LLRWA_STUDY=oct26 python scripts/kraken_rebuild_check.py
A ticker matches if some rebuilt state within +-50 ms has the same bid and ask. For tickers that
do not match, we report the gap between the ticker mid and the rebuilt mid prevailing at its time (bp).
Numbers quoted in Section 3.2.1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llrwa.config import DATA_CLEAN, STUDY
from llrwa.data.clean import load_raw
from llrwa.study import et_day_bounds
from llrwa.symbols import ASSETS

WINDOW_NS = 50_000_000


def check(asset: str, day: str) -> dict:
    lo, hi = et_day_bounds(day)
    b = pd.read_parquet(DATA_CLEAN / f"{asset}_kraken_{day.replace('-', '')}.parquet",
                        columns=["ts_ns", "bid", "ask"]).sort_values("ts_ns")
    t = load_raw("tickers", asset, "kraken", lo, hi, columns=["bid", "ask"])[["ts_ns", "bid", "ask"]].dropna()
    t = t.sort_values("ts_ns")
    bt, bb, ba = b["ts_ns"].to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy()
    tt, tb, ta = t["ts_ns"].to_numpy(), t["bid"].to_numpy(), t["ask"].to_numpy()
    i0 = np.searchsorted(bt, tt - WINDOW_NS, side="left") - 1
    i1 = np.searchsorted(bt, tt + WINDOW_NS, side="right")
    ok = np.zeros(len(t), bool)
    gap_bp = np.full(len(t), np.nan)
    for k in range(len(t)):
        s = slice(max(i0[k], 0), i1[k])
        ok[k] = bool(((bb[s] == tb[k]) & (ba[s] == ta[k])).any())
        if not ok[k]:
            j = max(np.searchsorted(bt, tt[k], side="right") - 1, 0)
            mid = (tb[k] + ta[k]) / 2
            gap_bp[k] = abs((bb[j] + ba[j]) / 2 - mid) / mid * 1e4
    return {"asset": asset, "day": day, "tickers": len(t), "match_share_50ms": ok.mean(),
            "median_gap_bp_unmatched": float(np.nanmedian(gap_bp))}


def main() -> None:
    df = pd.DataFrame([check(a, STUDY.ref_day) for a in ASSETS])
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
