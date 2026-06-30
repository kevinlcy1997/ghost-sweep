param(
    [int]$Port = 8765,
    [int]$MlflowPort = 5000,
    [string]$PythonPath = "",
    [switch]$InstallRequirements,
    [switch]$RefreshArtifacts,
    [switch]$RetrainModels,
    [switch]$Restart,
    [switch]$NoMlflow,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ProjectDir "analysis"
$OutLog = Join-Path $LogDir "dashboard_service.out.log"
$ErrLog = Join-Path $LogDir "dashboard_service.err.log"
$MlflowOutLog = Join-Path $LogDir "mlflow_service.out.log"
$MlflowErrLog = Join-Path $LogDir "mlflow_service.err.log"
$MlflowDb = Join-Path $LogDir "mlflow_tracking.db"
$MlflowArtifactRoot = Join-Path $LogDir "mlruns"
$Url = "http://127.0.0.1:$Port/"
$MlflowUrl = "http://127.0.0.1:$MlflowPort/"

function Resolve-Python {
    param([string]$RequestedPath)

    $candidates = @()
    if ($RequestedPath) { $candidates += $RequestedPath }
    $candidates += Join-Path $ProjectDir ".venv\Scripts\python.exe"
    $candidates += "C:\Users\Kevin\AppData\Local\Programs\Python\Python312\python.exe"

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return (Resolve-Path $candidate).Path
        }
    }

    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }

    throw "Could not find Python. Pass -PythonPath C:\Path\To\python.exe"
}

function Test-Dashboard {
    param([string]$HealthUrl)
    try {
        $summary = Invoke-RestMethod "$HealthUrl/api/summary" -TimeoutSec 3
        return $summary
    } catch {
        return $null
    }
}

function Test-HttpReady {
    param([string]$ReadyUrl)
    try {
        Invoke-WebRequest $ReadyUrl -UseBasicParsing -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Stop-PortOwner {
    param([int]$LocalPort)
    $connections = Get-NetTCPConnection -LocalPort $LocalPort -ErrorAction SilentlyContinue
    if (-not $connections) { return }

    $processIds = $connections | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($processId in $processIds) {
        if ($processId -and $processId -ne $PID) {
            Write-Host "Stopping existing process on port $LocalPort (PID $processId)..." -ForegroundColor Yellow
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
}

function Run-Step {
    param(
        [string]$Label,
        [string[]]$Arguments
    )
    Write-Host $Label -ForegroundColor Yellow
    & $Python @Arguments
}

function Initialize-LogFile {
    param(
        [string]$Path,
        [string]$Value
    )
    try {
        Set-Content -Path $Path -Value $Value -ErrorAction Stop
        return $Path
    } catch [System.IO.IOException] {
        $directory = Split-Path -Parent $Path
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($Path)
        $extension = [System.IO.Path]::GetExtension($Path)
        $stamp = Get-Date -Format "yyyyMMddHHmmss"
        $alternate = Join-Path $directory "$stem.$PID.$stamp$extension"
        Set-Content -Path $alternate -Value $Value
        return $alternate
    }
}

Set-Location $ProjectDir
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Python = Resolve-Python $PythonPath

Write-Host "=== Ghost Sweep Dev Environment ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectDir"
Write-Host "Python:  $Python"
Write-Host "URL:     $Url"
if (-not $NoMlflow) {
    Write-Host "MLflow:  $MlflowUrl"
}
Write-Host ""

$existing = Test-Dashboard "http://127.0.0.1:$Port"
if ($existing -and -not $Restart) {
    Write-Host "Dashboard service is already running." -ForegroundColor Green
    Write-Host "Mode: $($existing.coverage_mode), res$($existing.h3_resolution), cells $($existing.coverage_cells), events $($existing.event_count)"
    if (-not $NoBrowser) { Start-Process $Url }
    exit 0
}

if ($InstallRequirements) {
    Run-Step "Installing Python requirements..." @("-m", "pip", "install", "-r", "requirements.txt")
}

$shouldBuildManifest = $false
if ($RetrainModels) {
    Run-Step "Running multi-horizon experiment..." @("analysis/run_multi_horizon_experiment.py")
    $shouldBuildManifest = $true
}

$roadCoverage = Join-Path $ProjectDir "analysis\geo\hk_h3_road_coverage_res9.csv"
$roadGeoJson = Join-Path $ProjectDir "analysis\geo\hk_drivable_roads.geojson"
if ($RefreshArtifacts -or -not (Test-Path $roadCoverage)) {
    Run-Step "Building res9 fixed HK coverage grid..." @("analysis/build_hk_coverage_grid.py")

    $roadArgs = @("analysis/build_hk_road_coverage_grid.py", "--buffer-m", "60")
    if ($RefreshArtifacts -or -not (Test-Path $roadGeoJson)) {
        $roadArgs += "--fetch-roads"
    }
    Run-Step "Building road-access H3 coverage grid..." $roadArgs

    Run-Step "Building road-access feature marts..." @(
        "analysis/build_fixed_grid_feature_mart.py",
        "--grid-path", "analysis\geo\hk_h3_road_coverage_res9.csv",
        "--output-suffix", "road"
    )
    $shouldBuildManifest = $true
}

if ($shouldBuildManifest) {
    Run-Step "Building dashboard manifest..." @("analysis/build_dashboard_manifest.py")
}

Stop-PortOwner $Port
if (-not $NoMlflow) {
    Stop-PortOwner $MlflowPort
}
Start-Sleep -Milliseconds 500

$OutLog = Initialize-LogFile -Path $OutLog -Value "Starting Ghost Sweep dashboard service at $(Get-Date -Format o)`n"
$ErrLog = Initialize-LogFile -Path $ErrLog -Value ""
if (-not $NoMlflow) {
    $MlflowOutLog = Initialize-LogFile -Path $MlflowOutLog -Value "Starting Ghost Sweep MLflow UI at $(Get-Date -Format o)`n"
    $MlflowErrLog = Initialize-LogFile -Path $MlflowErrLog -Value ""
    New-Item -ItemType Directory -Force -Path $MlflowArtifactRoot | Out-Null
}

$mlflowProcess = $null
if (-not $NoMlflow) {
    Write-Host "Starting MLflow service..." -ForegroundColor Yellow
    $mlflowProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList @(
            "-m", "mlflow", "ui",
            "--backend-store-uri", "sqlite:///$($MlflowDb.Replace('\', '/'))",
            "--default-artifact-root", $MlflowArtifactRoot,
            "--host", "127.0.0.1",
            "--port", "$MlflowPort"
        ) `
        -WorkingDirectory $ProjectDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $MlflowOutLog `
        -RedirectStandardError $MlflowErrLog `
        -PassThru
}

Write-Host "Starting dashboard service..." -ForegroundColor Yellow
$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @("analysis/dashboard_service.py", "--port", "$Port") `
    -WorkingDirectory $ProjectDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -PassThru

$summary = $null
for ($attempt = 1; $attempt -le 20; $attempt++) {
    Start-Sleep -Milliseconds 500
    $summary = Test-Dashboard "http://127.0.0.1:$Port"
    if ($summary) { break }
}

if (-not $summary) {
    Write-Host "Dashboard service did not become ready. Check logs:" -ForegroundColor Red
    Write-Host "  $OutLog"
    Write-Host "  $ErrLog"
    exit 1
}

$mlflowReady = $false
if (-not $NoMlflow) {
    for ($attempt = 1; $attempt -le 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        $mlflowReady = Test-HttpReady $MlflowUrl
        if ($mlflowReady) { break }
    }
}

Write-Host ""
Write-Host "Dashboard ready." -ForegroundColor Green
Write-Host "PID: $($process.Id)"
Write-Host "URL: $Url"
Write-Host "Mode: $($summary.coverage_mode), res$($summary.h3_resolution), edge $($summary.average_hex_edge_m)m"
Write-Host "Cells: $($summary.coverage_cells), road-access: $($summary.road_access_cells), events: $($summary.event_count)"
Write-Host "Logs:"
Write-Host "  $OutLog"
Write-Host "  $ErrLog"

if (-not $NoMlflow) {
    if ($mlflowReady) {
        Write-Host ""
        Write-Host "MLflow ready." -ForegroundColor Green
        Write-Host "PID: $($mlflowProcess.Id)"
        Write-Host "URL: $MlflowUrl"
    } else {
        Write-Host ""
        Write-Host "MLflow service did not become ready yet. Check logs:" -ForegroundColor Yellow
    }
    Write-Host "  $MlflowOutLog"
    Write-Host "  $MlflowErrLog"
}

if (-not $NoBrowser) {
    Start-Process $Url
    if (-not $NoMlflow) {
        Start-Process $MlflowUrl
    }
}
