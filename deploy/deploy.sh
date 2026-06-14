#!/bin/bash
# Ghost Sweep Cloud Deploy Script
# Usage: ./deploy.sh <ssh-user>@<ip> [ssh-key-path]
#
# Example:
#   ./deploy.sh ubuntu@129.146.xx.xx ~/.ssh/oracle_key
#   ./deploy.sh root@your-vps.com

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <ssh-user>@<host> [ssh-key-path]"
    exit 1
fi

TARGET="$1"
SSH_KEY="${2:-}"
SSH_OPTS="-o StrictHostKeyChecking=accept-new"
[ -n "$SSH_KEY" ] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Ghost Sweep Cloud Deploy ==="
echo "Target: $TARGET"
echo "Source: $SCRIPT_DIR"
echo ""

# 1. Install Python + deps on remote
echo "[1/5] Installing Python on remote..."
ssh $SSH_OPTS "$TARGET" bash <<'REMOTE_SETUP'
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip > /dev/null
sudo useradd -r -s /bin/false ghost 2>/dev/null || true
sudo mkdir -p /opt/ghost-sweep/data
sudo chown -R ghost:ghost /opt/ghost-sweep
REMOTE_SETUP

# 2. Upload files
echo "[2/5] Uploading project files..."
scp $SSH_OPTS \
    "$SCRIPT_DIR/ghost_listener.py" \
    "$SCRIPT_DIR/ghost_db.py" \
    "$SCRIPT_DIR/ghost_utils.py" \
    "$SCRIPT_DIR/requirements.txt" \
    "$TARGET:/tmp/"

ssh $SSH_OPTS "$TARGET" "sudo cp /tmp/ghost_listener.py /tmp/ghost_db.py /tmp/ghost_utils.py /tmp/requirements.txt /opt/ghost-sweep/ && sudo chown ghost:ghost /opt/ghost-sweep/*"

# 3. Create venv + install deps
echo "[3/5] Creating virtual environment..."
ssh $SSH_OPTS "$TARGET" bash <<'REMOTE_VENV'
sudo -u ghost python3 -m venv /opt/ghost-sweep/venv
sudo -u ghost /opt/ghost-sweep/venv/bin/pip install -q cryptography requests
REMOTE_VENV

# 4. Install systemd service
echo "[4/5] Installing systemd service..."
scp $SSH_OPTS "$SCRIPT_DIR/deploy/ghost-listener.service" "$TARGET:/tmp/"
ssh $SSH_OPTS "$TARGET" bash <<'REMOTE_SERVICE'
sudo cp /tmp/ghost-listener.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ghost-listener
sudo systemctl start ghost-listener
REMOTE_SERVICE

# 5. Verify
echo "[5/5] Verifying..."
sleep 5
ssh $SSH_OPTS "$TARGET" bash <<'REMOTE_VERIFY'
echo "Service status:"
sudo systemctl status ghost-listener --no-pager -l | head -15
echo ""
echo "Log tail:"
sudo tail -5 /var/log/ghost-listener.log 2>/dev/null || echo "(waiting for first log...)"
echo ""
echo "Data directory:"
sudo ls -la /opt/ghost-sweep/data/ 2>/dev/null || echo "(empty - first cycle pending)"
REMOTE_VERIFY

echo ""
echo "=== Deploy Complete ==="
echo "Monitor:  ssh $SSH_OPTS $TARGET 'sudo journalctl -u ghost-listener -f'"
echo "Stats:    ssh $SSH_OPTS $TARGET 'sudo tail -20 /var/log/ghost-listener.log'"
echo "Pull DB:  scp $SSH_OPTS $TARGET:/opt/ghost-sweep/data/ghost_alerts.db ."
echo "Pull JSON: scp $SSH_OPTS $TARGET:/opt/ghost-sweep/data/ghost_alerts.json ."
