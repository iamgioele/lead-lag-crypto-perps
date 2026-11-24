"""Hyperliquid quotes versus Hyperliquid trades, on the same window and against the same venues.

Usage: LLRWA_STUDY=oct26 python scripts/quotes_vs_trades.py
Writes results/<study>/tables/quotes_vs_trades.csv and paper/tables/quotes_vs_trades.tex.

Quotes: the clean per-block best bid/offer series (mid-price). Trades: our recorded trades, last
trade price per block time, with the same 60 s segment rule. For Binance, OKX and Kraken against
each Hyperliquid series we report the pooled lag over the window and the median and interquartile
range of the estimates of single ET days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llrwa.config import DATA_CLEAN, RESULTS, ROOT, STUDY
from llrwa.data.hl_ws import load_trades
from llrwa.hry.pairwise import estimate_pairs
from llrwa.study import et_day_bounds
from llrwa.symbols import ASSET_NAMES_TEX, ASSETS

CEX = ["binance", "okx", "kraken"]
LBL = {"binance": "Binance", "okx": "OKX", "kraken": "Kraken"}
OUT = ROOT / "paper" / "tables"
SEGMENT_GAP_NS = 60 * 10**9


def clean_series(asset: str, venue: str) -> pd.DataFrame:
    fs = sorted(DATA_CLEAN.glob(f"{asset}_{venue}_2026*.parquet"))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f, columns=["ts_ns", "mid"]) for f in fs]).drop_duplicates("ts_ns", keep="last")
    return df[(df["ts_ns"] >= STUDY.T0.value) & (df["ts_ns"] < STUDY.T1.value)].sort_values("ts_ns")


def trade_series(asset: str) -> pd.DataFrame:
    tr = load_trades(asset, STUDY.T0.to_pydatetime(), STUDY.T1.to_pydatetime())
    tr = tr.drop_duplicates("time_ms", keep="last")              # last trade of each block
    return pd.DataFrame({"ts_ns": tr["time_ms"].to_numpy(np.int64) * 1_000_000, "mid": tr["px"].to_numpy()})


def with_segments(df: pd.DataFrame) -> pd.DataFrame:
    gap = np.diff(df["ts_ns"].to_numpy(), prepend=df["ts_ns"].iloc[0])
    return df.assign(segment=np.cumsum(gap > SEGMENT_GAP_NS))


def lags(series: dict[str, pd.DataFrame]) -> dict[tuple[str, str], float]:
    res, _ = estimate_pairs({k: with_segments(v) for k, v in series.items() if len(v) > 200})
    return {(r.leader_candidate, r.follower_candidate): (np.nan if r.censored else r.theta_ms)
            for r in res.itertuples()}


def main() -> None:
    rows = []
    for asset in ASSETS:
        s = {v: clean_series(asset, v) for v in CEX}
        s["hl_quotes"], s["hl_trades"] = clean_series(asset, "hyperliquid"), trade_series(asset)
        pooled = lags(s)
        daily = []
        for day in STUDY.et_days:
            lo, hi = (pd.Timestamp(x).value for x in et_day_bounds(day))
            if hi - lo < 20 * 3600 * 10**9 or lo < STUDY.T0.value or hi > STUDY.T1.value:
                continue                                          # full ET days only
            daily.append(lags({k: v[(v["ts_ns"] >= lo) & (v["ts_ns"] < hi)] for k, v in s.items()}))
        for v in CEX:
            for hl in ("hl_quotes", "hl_trades"):
                d = np.array([x.get((v, hl), np.nan) for x in daily], dtype=float)
                d = d[np.isfinite(d)]
                rows.append({"asset": asset, "venue": v, "hyperliquid": hl, "pooled_ms": pooled.get((v, hl), np.nan),
                             "daily_median_ms": np.median(d) if len(d) else np.nan,
                             "daily_iqr_ms": float(np.subtract(*np.percentile(d, [75, 25]))) if len(d) > 1 else np.nan,
                             "n_days": len(d)})
        print(asset, "done", flush=True)
    df = pd.DataFrame(rows)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "tables" / "quotes_vs_trades.csv", index=False)

    f = lambda x: "--" if not np.isfinite(x) else f"{x:+.0f}"
    lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{tabular}{llrrrr}", r"\hline",
             r"Asset & Leader & \multicolumn{2}{c}{Hyperliquid quotes} & \multicolumn{2}{c}{Hyperliquid trades} \\",
             r" & & pooled & daily median & pooled & daily median \\", r"\hline"]
    for a in ASSETS:
        first = True
        for v in CEX:
            q = df[(df["asset"] == a) & (df["venue"] == v) & (df["hyperliquid"] == "hl_quotes")].iloc[0]
            t = df[(df["asset"] == a) & (df["venue"] == v) & (df["hyperliquid"] == "hl_trades")].iloc[0]
            lines.append(f"{ASSET_NAMES_TEX[a] if first else ''} & {LBL[v]} & {f(q.pooled_ms)} & {f(q.daily_median_ms)} & "
                         f"{f(t.pooled_ms)} & {f(t.daily_median_ms)} \\\\")
            first = False
        lines.append(r"\hline")
    n_days = int(df["n_days"].max())
    lines += [r"\end{tabular}",
              r"\caption{Lag (ms) of Hyperliquid behind each centralised venue, measured with Hyperliquid's per-block "
              r"quotes and with its trades, both from our recording, " + STUDY.label.replace("-", "--") + r". Pooled: "
              r"whole window. Daily median: median over the " + str(n_days) + r" full days in New York time. "
              r"Positive: the centralised venue leads.}",
              r"\label{tab:quotes_vs_trades}", r"\end{table}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "quotes_vs_trades.tex").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 200)
    print(df.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
