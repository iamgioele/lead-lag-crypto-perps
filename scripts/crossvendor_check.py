"""Same-venue, cross-vendor lead-lag on one UTC day (default 2026-09-01, a free Tardis day).

For one venue, the lag between the Amberdata copy and the Tardis copy isolates vendor
timestamp effects (staleness, stamping convention). No market behaviour enters.
theta > 0 means the Amberdata copy leads the Tardis copy, so a negative value means the
Amberdata copy is stale. Also compares the two four-venue matrices.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from llrwa.config import RESULTS
from llrwa.data.clean import clean_cell, clean_tardis
from llrwa.hry.pairwise import estimate_pairs
from llrwa.symbols import instrument

DAY = sys.argv[1] if len(sys.argv) > 1 else "2026-09-01"
ASSET = sys.argv[2] if len(sys.argv) > 2 else "WTI"
T0 = datetime.fromisoformat(DAY).replace(tzinfo=timezone.utc)
TARDIS = {"binance": "book_ticker", "okx": "book_ticker", "kraken": "quotes", "hyperliquid": "book_ticker"}


def main() -> None:
    cs = float(instrument(ASSET, "okx")["contract_size"])
    amber, tard, rows = {}, {}, []
    for v in TARDIS:
        a, log, _ = clean_cell(ASSET, v, DAY, window=(T0, T0 + timedelta(days=1)),
                               okx_contract_size=cs if v == "okx" else None)
        t, _ = clean_tardis(ASSET, v, TARDIS[v], DAY, okx_contract_size=cs if v == "okx" else None)
        amber[v], tard[v] = a, t
        res, _ = estimate_pairs({f"amber:{v}": a, f"tardis:{v}": t}, max_lag_ms=1500)
        r = res.iloc[0]
        rows.append({"venue": v, "amber_source": a["source"].iloc[0] if len(a) else "-", "tardis_source": TARDIS[v],
                     "amber_rows": len(a), "tardis_rows": len(t), "theta_amber_vs_tardis_ms": r["theta_ms"],
                     "peak_rho": round(r["peak_rho"], 3), "censored": r["censored"]})
    same = pd.DataFrame(rows)
    ra, _ = estimate_pairs(amber)
    rt, _ = estimate_pairs(tard)
    cmp = ra[["leader_candidate", "follower_candidate", "theta_ms", "peak_rho"]].rename(
        columns={"theta_ms": "theta_amber_ms", "peak_rho": "rho_amber"})
    cmp["theta_tardis_ms"] = rt["theta_ms"].values
    cmp["rho_tardis"] = rt["peak_rho"].values
    out = RESULTS / "tables"
    same.to_csv(out / f"crossvendor_same_venue_{ASSET}_{DAY.replace('-', '')}.csv", index=False)
    cmp.to_csv(out / f"crossvendor_matrix_{ASSET}_{DAY.replace('-', '')}.csv", index=False)
    pd.set_option("display.width", 200)
    print("Same venue, Amberdata copy vs Tardis copy (theta > 0: Amberdata copy leads):")
    print(same.to_string(index=False))
    print("\nFour-venue pairs, Amberdata-based vs Tardis-based:")
    print(cmp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
