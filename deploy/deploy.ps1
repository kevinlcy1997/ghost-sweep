# Ghost Sweep Cloud Deploy (PowerShell version for Windows)
# Usage: .\deploy.ps1 -Target "ubuntu@129.146.xx.xx" -KeyPath "~\.ssh\oracle_key"
param(
    [Parameter(Mandatory)][string]$Target,
    [string]$KeyPath
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $ProjectDir) { $ProjectDir = Split-Path -Parent $PSScriptRoot }

$SshOpts = @("-o", "StrictHostKeyChecking=accept-new")
if ($KeyPath) { $SshOpts += @("-i", $KeyPath) }

Write-Host "=== Ghost Sweep Cloud Deploy ===" -ForegroundColor Cyan
Write-Host "Target: $Target"
Write-Host "Source: $ProjectDir"
Write-Host ""

# 1. Install Python on remote
Write-Host "[1/5] Installing Python on remote..." -ForegroundColor Yellow
ssh @SshOpts $Target "sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-venv python3-pip > /dev/null && sudo useradd -r -s /bin/false ghost 2>/dev/null; sudo mkdir -p /opt/ghost-sweep/data && sudo chown -R ghost:ghost /opt/ghost-sweep"

# 2. Upload files
Write-Host "[2/5] Uploading project files..." -ForegroundColor Yellow
$files = @("ghost_listener.py", "ghost_db.py", "ghost_utils.py", "requirements.txt")
foreach ($f in $files) {
    scp @SshOpts "$ProjectDir\$f" "${Target}:/tmp/"
}
ssh @SshOpts $Target "sudo cp /tmp/ghost_listener.py /tmp/ghost_db.py /tmp/ghost_utils.py /tmp/requirements.txt /opt/ghost-sweep/ && sudo chown ghost:ghost /opt/ghost-sweep/*"

# 3. Create venv
Write-Host "[3/5] Creating virtual environment..." -ForegroundColor Yellow
ssh @SshOpts $Target "sudo -u ghost python3 -m venv /opt/ghost-sweep/venv && sudo -u ghost /opt/ghost-sweep/venv/bin/pip install -q cryptography requests"

# 4. Install systemd service
Write-Host "[4/5] Installing systemd service..." -ForegroundColor Yellow
scp @SshOpts "$ProjectDir\deploy\ghost-listener.service" "${Target}:/tmp/"
ssh @SshOpts $Target "sudo cp /tmp/ghost-listener.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable ghost-listener && sudo systemctl start ghost-listener"

# 5. Verify
Write-Host "[5/5] Verifying..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
ssh @SshOpts $Target "sudo systemctl status ghost-listener --no-pager -l | head -15; echo ''; sudo tail -5 /var/log/ghost-listener.log 2>/dev/null; echo ''; sudo ls -la /opt/ghost-sweep/data/ 2>/dev/null"

Write-Host ""
Write-Host "=== Deploy Complete ===" -ForegroundColor Green
Write-Host "Monitor:  ssh $Target 'sudo journalctl -u ghost-listener -f'"
Write-Host "Pull DB:  scp $Target`:/opt/ghost-sweep/data/ghost_alerts.db ."
Write-Host "Pull JSON: scp $Target`:/opt/ghost-sweep/data/ghost_alerts.json ."
