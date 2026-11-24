#!/usr/bin/env bash
# Copy the capture from the droplet to the Mac (incremental; safe to rerun any time)
# and print the server's health report.
#   bash collector/pull_from_server.sh
set -euo pipefail
HOST=${HOST:?set HOST=user@your-server}
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/hyperliquid_ws.nosync"   # .nosync: keep out of iCloud
mkdir -p "$DEST"
rsync -a --partial -e "ssh -o BatchMode=yes" "$HOST:/home/hl/data/" "$DEST/"
ssh -o BatchMode=yes "$HOST" 'systemctl is-active hl-collector; sudo -u hl /home/hl/collector/.venv/bin/python /home/hl/collector/health.py /home/hl/data "$(date -u +%Y%m%d)"'
du -sh "$DEST"
