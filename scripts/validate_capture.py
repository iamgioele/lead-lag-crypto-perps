"""Check our Hyperliquid recording against an independent source: Amberdata's Hyperliquid trades.

Usage: LLRWA_STUDY=oct26 python scripts/validate_capture.py
Needs Amberdata trades for the study window (pull_window.py --venues hyperliquid).
Writes results/<study>/tables/capture_validation.csv and paper/tables/capture_macros.tex.

Trades are matched on (block time in ms, trade id). For matched trades we compare price and size.
Our trades are the union over connections; the replay of the last 30 trades after each reconnect
is removed by deduplicating on (coin, time, tid).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llrwa.config import RESULTS, ROOT, STUDY
from llrwa.data.clean import load_raw
from llrwa.data.hl_ws import load_trades
from llrwa.symbols import ASSETS

OUT = ROOT / "paper" / "tables"


def our_trades(asset: str) -> pd.DataFrame:
    return load_trades(asset, STUDY.T0.to_pydatetime(), STUDY.T1.to_pydatetime())


def amber_trades(asset: str) -> pd.DataFrame:
    a = load_raw("trades", asset, "hyperliquid", STUDY.T0.to_pydatetime(), STUDY.T1.to_pydatetime(),
                 columns=["price", "volume", "tradeId"])
    return pd.DataFrame({"time_ms": a["ts_ns"] // 10**6, "tid": a["tradeId"].astype(str),
                         "px": a["price"], "sz": a["volume"]}).drop_duplicates(["time_ms", "tid"])


def main() -> None:
    rows = []
    for asset in ASSETS:
        o, a = our_trades(asset), amber_trades(asset)
        if o.empty or a.empty:
            continue
        m = o.merge(a, on=["time_ms", "tid"], suffixes=("_o", "_a"))
        rows.append({"asset": asset, "ours": len(o), "amberdata": len(a), "matched": len(m),
                     "same_tid_other_time": int(o.merge(a, on="tid").query("time_ms_x != time_ms_y").shape[0]),
                     "price_equal": int(np.isclose(m["px_o"], m["px_a"], rtol=0, atol=1e-9).sum()),
                     "size_equal": int(np.isclose(m["sz_o"], m["sz_a"], rtol=0, atol=1e-9).sum())})
    df = pd.DataFrame(rows)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "tables" / "capture_validation.csv", index=False)
    tot = df.sum(numeric_only=True)
    big = lambda x: f"{int(x):,}".replace(",", "{,}")
    pct = lambda x: f"{x:.2f}".rstrip("0").rstrip(".")          # "100", not "100.00"
    mac = {"CVOurTrades": big(tot["ours"]), "CVAmberTrades": big(tot["amberdata"]),
           "CVMatchedTrades": big(tot["matched"]),
           "CVShareAmberMatched": pct(100 * tot['matched'] / tot['amberdata']),
           "CVShareOursMatched": pct(100 * tot['matched'] / tot['ours']),
           "CVPriceSizeEqual": pct(100 * min(tot['price_equal'], tot['size_equal']) / tot['matched'])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "capture_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    print(df.to_string(index=False))
    print(mac)


if __name__ == "__main__":
    main()
