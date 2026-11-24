"""Phase 3 batch: clean every (asset, venue, ET day) and collect the cleaning log.

Usage: python scripts/clean_all.py --assets WTI --venues binance okx [--days 2026-10-26 ...]
ET days are America/New_York calendar days (DST-aware); default: the days of the selected study
(LLRWA_STUDY, see llrwa/study.py). The first and last days of the window are partial.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from llrwa.config import DATA_CLEAN, RESULTS, STUDY
from llrwa.data.clean import clean_cell, write_clean
from llrwa.symbols import ASSETS, instrument

DAYS = STUDY.et_days


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=list(ASSETS))
    ap.add_argument("--venues", nargs="*", default=["binance", "okx", "kraken", "hyperliquid"])
    ap.add_argument("--days", nargs="*", default=DAYS)
    a = ap.parse_args()
    logdir = DATA_CLEAN / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    for asset in a.assets:
        cs = float(instrument(asset, "okx")["contract_size"])
        for venue in a.venues:
            for day in a.days:
                t = time.monotonic()
                df, log, _ = clean_cell(asset, venue, day, okx_contract_size=cs if venue == "okx" else None)
                log["seconds"] = round(time.monotonic() - t, 1)
                log["mid_changes"] = int((df["mid"].diff() != 0).sum() - 1) if len(df) else 0
                if len(df):
                    write_clean(df, asset, venue, day)
                (logdir / f"{asset}_{venue}_{day}.json").write_text(json.dumps(log, default=float))
                print(f"{asset:6s} {venue:12s} {day} rows_out={log.get('rows_out', 0):>9,d} "
                      f"changes={log['mid_changes']:>8,d} {log['seconds']:5.1f}s", flush=True)
    rows = [json.loads(p.read_text()) for p in sorted(logdir.glob("*.json")) if p.name.split("_")[0] in ASSETS]
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "tables" / "cleaning_log.csv", index=False)


if __name__ == "__main__":
    main()
