#!/usr/bin/env bash
# For each asset in order: wait until its Hyperliquid trades line for the main window appears in the pull log, then run the pipeline.
# Usage: scripts/wait_and_run.sh <pull log> <ASSET> [<ASSET> ...]
cd "$(dirname "$0")/.."
LOG="$1"; shift
for A in "$@"; do
  until grep -qE "^[0-9:]+ ${A} +hyperliquid +trades +chunks= *172 " "$LOG"; do sleep 20; done
  scripts/asset_pipeline.sh "$A" > "data/raw/logs/pipeline_${A}.log" 2>&1 || echo "PIPELINE FAILED $A"
  tail -1 "data/raw/logs/pipeline_${A}.log"
done
