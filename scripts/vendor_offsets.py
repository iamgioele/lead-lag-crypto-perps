"""Same-venue vendor offsets for every asset on one free Tardis day (default 2026-09-01).

For each (asset, venue) with both an Amberdata and a Tardis copy, the lag between the two copies
isolates vendor timestamp effects (see scripts/crossvendor_check.py). Negative means the
Amberdata copy is late. Writes results/tables/vendor_offsets_<day>.csv.

Usage: python scripts/vendor_offsets.py [2026-09-01]
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from llrwa.config import RESULTS
from llrwa.data.clean import clean_cell, clean_tardis
from llrwa.hry.pairwise import estimate_pairs
from llrwa.symbols import ASSETS, VENUES, instrument

DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-09-01"
T0 = datetime.fromisoformat(DAY).replace(tzinfo=timezone.utc)
TARDIS = {"binance": "book_ticker", "okx": "book_ticker", "kraken": "quotes", "hyperliquid": "book_ticker"}


def main() -> None:
    rows = []
    for asset in ASSETS:
        cs = float(instrument(asset, "okx")["contract_size"])
        for v in VENUES:
            try:
                a, _, _ = clean_cell(asset, v, DAY, window=(T0, T0 + timedelta(days=1)),
                                     okx_contract_size=cs if v == "okx" else None)
                t, _ = clean_tardis(asset, v, TARDIS[v], DAY, okx_contract_size=cs if v == "okx" else None)
            except FileNotFoundError:
                continue
            if len(a) < 100 or len(t) < 100:
                continue
            res, _ = estimate_pairs({f"amber:{v}": a, f"tardis:{v}": t}, max_lag_ms=1500)
            r = res.iloc[0]
            rows.append({"asset": asset, "venue": v, "amber": a["source"].iloc[0], "amber_rows": len(a),
                         "tardis_rows": len(t), "offset_ms": r["theta_ms"], "peak_rho": round(r["peak_rho"], 3),
                         "censored": r["censored"]})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "tables" / f"vendor_offsets_{DAY.replace('-', '')}.csv", index=False)


if __name__ == "__main__":
    main()
