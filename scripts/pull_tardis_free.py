"""Download free first-of-month Tardis days (no key needed) and convert them to parquet."""
from __future__ import annotations

import sys
from datetime import date

from llrwa.config import DATA_RAW
from llrwa.data.tardis import pull_day
from llrwa.symbols import load_instruments

DAY = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2026, 9, 1)
ASSETS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["WTI"]
TYPES = {"binance": ["book_ticker"], "okx": ["book_ticker", "quotes"], "kraken": ["quotes"], "hyperliquid": ["book_ticker"]}
VENUES = sys.argv[3].split(",") if len(sys.argv) > 3 else ["okx", "kraken", "hyperliquid"]

inst = load_instruments()
for asset in ASSETS:
    for venue in VENUES:
        r = inst[(inst["asset"] == asset) & (inst["venue"] == venue)].iloc[0]
        for dt in TYPES[venue]:
            rec = pull_day(asset=asset, venue=venue, exchange=r["tardis_exchange"], symbol=r["tardis_symbol"],
                           data_type=dt, day=DAY, root=DATA_RAW, api_key="")
            print(f"{asset} {venue} {dt}: {rec.status} rows={rec.rows:,} {rec.seconds:.0f}s {rec.error[:150]}", flush=True)
