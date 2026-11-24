#!/usr/bin/env bash
# Full per-asset analysis after its raw data is pulled: clean -> curves -> scopes -> figures -> robustness -> bootstrap.
set -euo pipefail
ASSET="$1"
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/clean_all.py --assets "$ASSET"
$PY scripts/build_curves.py --asset "$ASSET"
for v in "--price micro" "--change-only" "--coarsen-ms 5" "--coarsen-ms 10"; do $PY scripts/build_curves.py --asset "$ASSET" $v; done
$PY scripts/matrices.py --asset "$ASSET"
$PY scripts/robustness.py --asset "$ASSET"
$PY scripts/bootstrap_asset.py --asset "$ASSET" --B 300
# after the bootstrap: the offset fit and the heatmaps leave out poorly located pairs (95% CI wider than 100 ms)
$PY scripts/figures_asset.py --asset "$ASSET"
echo "PIPELINE DONE $ASSET"
