"""Background Amberdata pulls: resumable, logged, chunked.

Usage:
    python scripts/pull_window.py --start 2026-09-24T04:00 --end 2026-09-25T04:00 --assets WTI
    python scripts/pull_window.py --start 2026-09-18T20:00 --end 2026-09-26T00:00

Per venue: Binance/OKX tickers; Kraken tickers + order-book events + snapshots (to rebuild the
event-level best bid/offer); Hyperliquid trades (Amberdata has no Hyperliquid top-of-book).
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

from llrwa.config import DATA_RAW, api_key
from llrwa.data.amber import AmberClient, pull
from llrwa.symbols import ASSETS, load_instruments

KINDS_BY_VENUE = {"binance": ["tickers"], "okx": ["tickers"],
                  "kraken": ["tickers", "order-book-snapshots", "order-book-events"],
                  "hyperliquid": ["trades"]}


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--assets", nargs="*", default=list(ASSETS))
    ap.add_argument("--venues", nargs="*", default=list(KINDS_BY_VENUE))
    ap.add_argument("--chunk-minutes", type=int, default=60)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rate", type=float, default=15.0)
    a = ap.parse_args()
    t0, t1 = utc(a.start), utc(a.end)
    client = AmberClient(api_key("amberdata"), rate_per_s=a.rate)
    inst = load_instruments()
    for asset in a.assets:
        for venue in a.venues:
            r = inst[(inst["asset"] == asset) & (inst["venue"] == venue)].iloc[0]
            for kind in KINDS_BY_VENUE[venue]:
                started = time.monotonic()
                recs = pull(client, kind, asset=asset, venue=venue, exchange=r["amber_exchange"],
                            instrument=r["amber_instrument"], t0=t0, t1=t1, root=DATA_RAW,
                            chunk_minutes=a.chunk_minutes, workers=a.workers)
                bad = [x for x in recs if x.status == "error"]
                print(f"{datetime.now(timezone.utc):%H:%M:%S} {asset:6s} {venue:12s} {kind:21s} chunks={len(recs):4d} "
                      f"rows={sum(x.rows for x in recs):>11,d} pages={sum(x.pages for x in recs):>6,d} "
                      f"errors={len(bad)} {time.monotonic() - started:7.0f}s"
                      + (f" first error: {bad[0].error[:120]}" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
