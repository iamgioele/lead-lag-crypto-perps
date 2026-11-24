#!/usr/bin/env bash
# Pulls for the assets added on 28 Sep 2026 (D27): the main window (one process per asset, in parallel),
# then the Phase-1 windows and the 1 Sep vendor-check day from both vendors.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ASSETS="${*:-GOLD SNDK MU}"
for A in $ASSETS; do
  $PY scripts/pull_window.py --start 2026-09-18T20:00 --end 2026-09-26T00:00 --assets "$A" &
done
wait
$PY scripts/phase1_resolution.py > data/raw/logs/phase1_rerun.log 2>&1
$PY scripts/pull_window.py --start 2026-09-01T00:00 --end 2026-09-02T00:00 --assets $ASSETS
$PY scripts/pull_tardis_free.py 2026-09-01 "$(echo $ASSETS | tr ' ' ',')" binance,okx,kraken,hyperliquid
echo "PULLS DONE"
