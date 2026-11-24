"""Phase 1: pull two 10-minute windows per cell from Amberdata and measure feed resolution.

Exchanges (Binance, OKX, Kraken) use top-of-book tickers. Amberdata has no Hyperliquid
top-of-book, so Hyperliquid uses trades. Output: results/tables/resolution.csv and
results/tables/coverage.csv.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from llrwa.config import DATA_RAW, RESULTS, api_key
from llrwa.data.amber import AmberClient, pull
from llrwa.data.feedstats import feed_stats, load_amber_window
from llrwa.symbols import load_instruments

WINDOWS = {
    "weekday Thu 24 Sep 14:00-14:10Z": (datetime(2026, 9, 24, 14, 0, tzinfo=timezone.utc),
                                        datetime(2026, 9, 24, 14, 10, tzinfo=timezone.utc)),
    "weekend Sat 19 Sep 14:00-14:10Z": (datetime(2026, 9, 19, 14, 0, tzinfo=timezone.utc),
                                        datetime(2026, 9, 19, 14, 10, tzinfo=timezone.utc)),
}


def main() -> None:
    client = AmberClient(api_key("amberdata"), rate_per_s=15)
    inst = load_instruments()
    rows, log = [], DATA_RAW / "pull_log.jsonl"
    for _, r in inst.iterrows():
        kind = "trades" if r["venue"] == "hyperliquid" else "tickers"
        for wname, (t0, t1) in WINDOWS.items():
            recs = pull(client, kind, asset=r["asset"], venue=r["venue"], exchange=r["amber_exchange"],
                        instrument=r["amber_instrument"], t0=t0, t1=t1, root=DATA_RAW, chunk_minutes=10,
                        workers=1, log_path=log)
            for rec in recs:
                print(f"{r['asset']:6s} {r['venue']:12s} {kind:8s} {wname[:7]} {rec.status:6s} rows={rec.rows:>7,d} "
                      f"pages={rec.pages:>3d} {rec.seconds:6.1f}s {rec.error[:60]}")
            df = load_amber_window(kind, r["asset"], r["venue"], t0, t1, DATA_RAW)
            window_s = (t1 - t0).total_seconds()
            if kind == "tickers" and len(df):
                st = feed_stats(df["ts_ns"].to_numpy(), ((df["bid"] + df["ask"]) / 2).to_numpy(), window_s,
                                df["bid"].to_numpy(), df["ask"].to_numpy())
            elif len(df):
                st = feed_stats(df["ts_ns"].to_numpy(), df["price"].to_numpy(), window_s)
            else:
                st = {"rows": 0}
            rows.append({"asset": r["asset"], "venue": r["venue"], "vendor": "amberdata", "dataset": kind,
                         "window": wname, **st})
    res = pd.DataFrame(rows)
    # seconds per page from the manifest
    man = pd.DataFrame([json.loads(l) for l in log.read_text().splitlines() if l.strip()])
    man = man[man["status"].isin(["ok", "empty"]) & man["asset"].isin(set(inst["asset"]))].groupby(["asset", "venue", "kind"]).agg(
        pages=("pages", "sum"), seconds=("seconds", "sum")).reset_index()
    man["seconds_per_page"] = man["seconds"] / man["pages"].clip(lower=1)
    out = RESULTS / "tables"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "resolution.csv", index=False)
    cov = inst[["asset", "venue", "native_symbol", "asset_type", "amber_instrument"]].copy()
    wk = res[res["window"].str.startswith("weekday")].set_index(["asset", "venue"])
    cov["amber_dataset"] = [("trades" if v == "hyperliquid" else "tickers") for v in cov["venue"]]
    cov["amber_rows_10min_weekday"] = [wk.loc[(a, v), "rows"] for a, v in zip(cov["asset"], cov["venue"])]
    cov["top_of_book_source"] = ["none on Amberdata (trades only); Tardis account inactive" if v == "hyperliquid"
                                 else "Amberdata tickers" for v in cov["venue"]]
    cov.to_csv(out / "coverage.csv", index=False)
    man.to_csv(out / "pull_speed.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.drop(columns=["vendor"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
