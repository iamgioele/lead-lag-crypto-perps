"""Activity by venue and trading state: price changes per second (clean data of the selected study).

Writes results/tables/activity.csv and paper/tables/activity.tex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llrwa.calendars.buckets import BUCKETS, CME_ASSETS, EQUITY_ASSETS, bucket_intervals
from llrwa.config import DATA_CLEAN, RESULTS, ROOT, STUDY
from llrwa.study import state_mask
from llrwa.symbols import ASSET_NAMES_TEX, ASSETS

VENUES = ["binance", "okx", "kraken", "hyperliquid"]
HL_WORDS = "per-block quotes" if STUDY.hl_source == "ws_bbo" else "trade prices"
LBL = {"binance": "Binance", "okx": "OKX", "kraken": "Kraken", "hyperliquid": "Hyperliquid"}
OPEN = {a: [BUCKETS[a][0]] for a in ASSETS}
CLOSED_WEEKDAY = {**{a: ["daily_break"] for a in CME_ASSETS}, **{a: ["overnight"] for a in EQUITY_ASSETS}}
WEEKEND = {a: [BUCKETS[a][-1]] for a in ASSETS}
T0, T1 = STUDY.T0, STUDY.T1


def main() -> None:
    rows = []
    for asset in ASSETS:
        bk = bucket_intervals(asset, T0.to_pydatetime(), T1.to_pydatetime())
        starts = bk["start_utc"].values.astype("datetime64[ns]").astype(np.int64)
        ends = bk["end_utc"].values.astype("datetime64[ns]").astype(np.int64)
        for v in VENUES:
            files = sorted(DATA_CLEAN.glob(f"{asset}_{v}_2026*.parquet"))
            if not files:
                continue
            df = pd.concat([pd.read_parquet(f, columns=["ts_ns", "mid"]) for f in files]).sort_values("ts_ns")
            df = df[(df["ts_ns"] >= T0.value) & (df["ts_ns"] < T1.value)]
            chg = df["ts_ns"].to_numpy()[1:][np.diff(df["mid"].to_numpy()) != 0]
            idx = np.searchsorted(starts, chg, side="right") - 1
            idx = idx[(idx >= 0)]
            counts = np.bincount(idx, minlength=len(bk))
            # seconds of each bucket interval during which the venue has data (Kraken MU lists mid-sample)
            lo, hi = df["ts_ns"].iloc[0], df["ts_ns"].iloc[-1]
            dur = np.clip(np.minimum(ends, hi) - np.maximum(starts, lo), 0, None) / 1e9
            for name, labels in (("open", OPEN[asset]), ("closed_weekday", CLOSED_WEEKDAY[asset]), ("weekend", WEEKEND[asset])):
                m = (bk["bucket"].isin(labels).values & ~bk["transition"].values
                     & state_mask(STUDY, asset, pd.DatetimeIndex(bk["start_utc"])))
                rows.append({"asset": asset, "venue": v, "state": name,
                             "changes_per_s": counts[m].sum() / dur[m].sum() if dur[m].sum() else np.nan})
    df = pd.DataFrame(rows)
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "tables" / "activity.csv", index=False)
    piv = df.pivot_table(index=["asset", "venue"], columns="state", values="changes_per_s", sort=False)
    names = ASSET_NAMES_TEX
    lines = [r"\begin{table}[ht]", r"\centering\small", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{llrrr}", r"\hline",
             r"Asset & Venue & Open & Closed (weekday) & Weekend \\", r"\hline"]
    last = None
    for (a, v), r in piv.iterrows():
        cells = " & ".join("--" if np.isnan(r[c]) else f"{r[c]:.2f}" for c in ("open", "closed_weekday", "weekend"))
        lines.append(f"{names[a] if a != last else ''} & {LBL[v]} & {cells} \\\\")
        last = a
    lines += [r"\hline", r"\end{tabular}", r"\end{adjustbox}",
              r"\caption{Price changes per second by trading state (" + STUDY.label.replace("-", "--") + r"). Open: US day (WTI, gold), "
              r"regular session (stocks). Closed (weekday): CME daily break (WTI, gold), overnight (stocks). "
              r"Kraken counts refer to the rebuilt book; Hyperliquid to " + HL_WORDS + r". --: no data in that state.}", r"\label{tab:activity}", r"\end{table}"]
    (ROOT / "paper" / "tables" / "activity.tex").write_text("\n".join(lines) + "\n")
    pd.set_option("display.width", 200)
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
