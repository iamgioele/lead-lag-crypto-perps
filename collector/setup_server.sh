#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 24.04 droplet. Run as root from the folder holding
# hl_collector.py and hl-collector.service:   bash setup_server.sh
set -euo pipefail

timedatectl set-timezone UTC
apt-get update -q
apt-get install -y -q python3-venv chrony   # chrony keeps the clock synced (recv_ns quality)
systemctl enable --now chrony

id hl >/dev/null 2>&1 || useradd -m -s /bin/bash hl
install -d -o hl -g hl /home/hl/collector /home/hl/data
install -o hl -g hl -m 644 hl_collector.py /home/hl/collector/
sudo -u hl python3 -m venv /home/hl/collector/.venv
sudo -u hl /home/hl/collector/.venv/bin/pip install -q "websockets==15.0.1"

install -m 644 hl-collector.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now hl-collector
systemctl status hl-collector --no-pager

# security upgrades must not restart the capture (a restart drops all connections at once)
cat > /etc/needrestart/conf.d/50-hl-collector.conf <<'CONF'
$nrconf{override_rc}{qr(^hl-collector\.service$)} = 0;
CONF
