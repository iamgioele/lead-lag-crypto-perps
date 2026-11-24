"""Feed table for Section 3.2 from the clean series of the selected study (no hand-copied numbers).

Usage: LLRWA_STUDY=oct26 python scripts/feed_table.py
Writes results/<study>/tables/feeds.csv, paper/tables/feeds.tex and paper/tables/feeds_macros.tex
(\\newcommand values for the Hyperliquid capture quoted in the text).

Per venue and asset, over the study window: quote updates per second, mid-price changes per
second, median gap between updates, and the 100 ms grid concentration R (1 = every timestamp on
the same phase of a 100 ms grid, 0 = no grid). For Hyperliquid also: how many of the three
websocket connections delivered each update, and the expected number missed by all of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llrwa.config import DATA_CLEAN, RESULTS, ROOT, STUDY
from llrwa.data.feedstats import grid_phase
from llrwa.data.hl_ws import load_bbo
from llrwa.symbols import ASSETS

VENUES = ["binance", "okx", "kraken", "hyperliquid"]
LBL = {"binance": "Binance", "okx": "OKX", "kraken": "Kraken", "hyperliquid": "Hyperliquid"}
SOURCE = {"binance": ("Amberdata", "top-of-book tickers"), "okx": ("Amberdata", "top-of-book tickers"),
          "kraken": ("Amberdata", "rebuilt from book events")}
OUT = ROOT / "paper" / "tables"


def series(asset: str, venue: str) -> pd.DataFrame:
    fs = sorted(DATA_CLEAN.glob(f"{asset}_{venue}_2026*.parquet"))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f, columns=["ts_ns", "mid"]) for f in fs]).drop_duplicates("ts_ns")
    lo, hi = STUDY.T0.value, STUDY.T1.value
    return df[(df["ts_ns"] >= lo) & (df["ts_ns"] < hi)].sort_values("ts_ns")


def main() -> None:
    hours = (STUDY.T1 - STUDY.T0).total_seconds()
    rows = []
    for a in ASSETS:
        for v in VENUES:
            df = series(a, v)
            if df.empty:
                continue
            t = df["ts_ns"].to_numpy()
            rows.append({"asset": a, "venue": v, "updates_per_s": len(df) / hours,
                         "mid_changes_per_s": int((np.diff(df["mid"].to_numpy()) != 0).sum()) / hours,
                         "median_gap_ms": float(np.median(np.diff(t))) / 1e6, "grid_R": grid_phase(t)[0]})
    st = pd.DataFrame(rows)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    st.to_csv(RESULTS / "tables" / "feeds.csv", index=False)

    def rng(x, d=0):
        lo, hi = x.min(), x.max()
        return f"{lo:.{d}f}" if round(lo, d) == round(hi, d) else f"{lo:.{d}f}--{hi:.{d}f}"

    src = dict(SOURCE)
    src["hyperliquid"] = (("Own capture", "per-block best bid/offer") if STUDY.hl_source == "ws_bbo"
                          else ("Amberdata", "trades"))
    lines = [r"\begin{table}[ht]", r"\centering\footnotesize", r"\begin{adjustbox}{max width=\textwidth}",
             r"\begin{tabular}{lllrrrr}", r"\hline",
             r"Venue & Source & Content & Updates/s & Mid changes/s & Median gap (ms) & Grid $R$ \\", r"\hline"]
    for v in VENUES:
        g = st[st["venue"] == v]
        if g.empty:
            continue
        lines.append(f"{LBL[v]} & {src[v][0]} & {src[v][1]} & {rng(g['updates_per_s'], 1)} & "
                     f"{rng(g['mid_changes_per_s'], 1)} & {rng(g['median_gap_ms'])} & {rng(g['grid_R'], 2)} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{adjustbox}",
              r"\caption{Data feeds, " + STUDY.label.replace("-", "--") + r". Ranges over the four assets. Updates/s counts every "
              r"change of the best quotes, including size-only changes; Mid changes/s counts changes of the "
              r"mid-price. Grid $R$ is the concentration of timestamps modulo 100\,ms (1: all on one phase of a "
              r"100\,ms grid, 0: no grid).}",
              r"\label{tab:feeds}", r"\end{table}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feeds.tex").write_text("\n".join(lines) + "\n")

    if STUDY.hl_source == "ws_bbo":
        tot = all3 = one = 0
        missed = 0.0
        for a in ASSETS:
            b = load_bbo(a, STUDY.T0.to_pydatetime(), STUDY.T1.to_pydatetime())
            if b.empty:
                continue
            m = b["conn_mask"].to_numpy()
            n = len(b)
            p_miss = [1 - ((m >> k) & 1).sum() / n for k in range(3)]   # share of the union each connection missed
            tot += n
            all3 += int((b["n_conn"] == 3).sum())
            one += int((b["n_conn"] == 1).sum())
            missed += n * float(np.prod(p_miss))                         # assuming independent misses
            print(f"{a}: union {n:,}, missed per connection {[round(100 * p, 2) for p in p_miss]} %")
        mac = {"HLUnionUpdates": f"{tot:,}".replace(",", "{,}"),
               "HLShareAllThree": f"{100 * all3 / tot:.1f}", "HLShareOnlyOne": f"{100 * one / tot:.2f}",
               "HLMissedAll": f"{missed:.0f}", "HLMissedAllShare": f"{100 * missed / tot:.4f}"}
        (OUT / "feeds_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
        print(mac)
    pd.set_option("display.width", 200)
    print(st.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
