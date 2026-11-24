#!/usr/bin/env bash
# After the D27 pulls: run the remaining per-asset pipelines one at a time (memory), then the shared tables.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for A in "$@"; do
  scripts/asset_pipeline.sh "$A" > "data/raw/logs/pipeline_${A}.log" 2>&1 || echo "PIPELINE FAILED $A"
  tail -1 "data/raw/logs/pipeline_${A}.log"
done
$PY scripts/phase1_resolution.py > data/raw/logs/phase1_rerun.log 2>&1 || echo "PHASE1 FAILED"
$PY scripts/vendor_offsets.py 2026-09-01 > data/raw/logs/vendor_offsets.log 2>&1 || echo "VENDOR FAILED"
$PY scripts/activity_table.py > data/raw/logs/activity.log 2>&1 || echo "ACTIVITY FAILED"
$PY scripts/paper_tables.py > data/raw/logs/paper_tables.log 2>&1 || echo "TABLES FAILED"
echo "FINISH DONE"
