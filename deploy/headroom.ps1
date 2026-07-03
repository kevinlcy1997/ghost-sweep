[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet("preflight")]
    [string]$Action,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CopilotArgs
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$HeadroomVenv = Join-Path $RepoRoot ".headroom-venv"
$RepoPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$HeadroomPython = Join-Path $HeadroomVenv "Scripts\python.exe"
$HeadroomExe = Join-Path $HeadroomVenv "Scripts\headroom.exe"

function Write-Info([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Fail([string]$Message) {
    throw $Message
}

function Test-Tool([string]$Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Get-BootstrapPython {
    if (Test-Path $RepoPython) {
        return $RepoPython
    }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    Fail "Python 3.10+ not found. Create the repo .venv or install Python first."
}

function Assert-PythonVersion([string]$PythonPath) {
    $versionText = & $PythonPath -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0) {
        Fail "Unable to query Python version from $PythonPath."
    }

    $version = [version]"$versionText.0"
    if ($version -lt [version]"3.10.0") {
        Fail "Python 3.10+ required. Found $versionText at $PythonPath."
    }
}

function Invoke-Preflight {
    Write-Info "=== Headroom Copilot Preflight ==="

    if (-not (Test-Tool "copilot")) {
        Fail "GitHub Copilot CLI not found. Install @github/copilot first."
    }

    $pythonPath = Get-BootstrapPython
    Assert-PythonVersion $pythonPath

    Write-Info "Copilot CLI: OK"
    Write-Info "Python: $pythonPath"

    if (-not (Test-Path $HeadroomExe)) {
        $missing = @()
        if (-not (Test-Tool "link.exe")) {
            $missing += "link.exe (MSVC Build Tools)"
        }
        if (-not (Test-Tool "rustc")) {
            $missing += "rustc (Rust toolchain)"
        }
        if ($missing.Count -gt 0) {
            Fail "Headroom install prerequisites missing: $($missing -join ', ')."
        }
    }

    Write-Info "Preflight: OK"
}

switch ($Action) {
    "preflight" { Invoke-Preflight }
}
