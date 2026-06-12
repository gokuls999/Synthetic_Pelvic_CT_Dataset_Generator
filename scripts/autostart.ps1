# Auto-launch wrapper for the PFD production pipeline.
#
# Fires from the "PelvicCT-Generator-Pipeline" Windows scheduled task at boot
# (install via scripts/install_autostart.ps1). After a power cut / restart,
# this script:
#
#   1. Checks if production is already complete (roster_summary.json with
#      n_succeeded >= $expectedPatients) -> exits, leaves the box alone.
#   2. Checks if the production dashboard is already responding on port 8766
#      -> exits, another instance is running.
#   3. Waits $bootSettleSec for drivers / GPU, then resumes:
#      python scripts\pfd_production_v2.py --skip-done --no-browser
#      Output lands in synthetic_dataset/production_350 (inside the project).
#
# Per-boot log: logs\autostart_YYYYMMDD_HHMMSS.log under the project dir.

$ErrorActionPreference = 'Continue'

# --- Edit these two paths if you move the project or use a different Python ---
$projectDir = 'D:\Muthu kumar\gen_ai_ct_pelvic'
$pythonExe  = 'C:\Python314\python.exe'
# --- Pipeline command (kept here so it's easy to change without editing the task) ---
$expectedPatients = 350     # treat production as done once this many patients succeed
$dashboardPort    = 8766    # PFD production dashboard port
$bootSettleSec    = 60      # let drivers / GPU / disks come up before launching

# --- Logging --------------------------------------------------------------
$logDir = Join-Path $projectDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("autostart_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")
function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}
Write-Log "autostart triggered"

# --- Guard 1: production already completed --------------------------------
$summaryPath = Join-Path $projectDir 'synthetic_dataset\production_350\roster_summary.json'
if (Test-Path $summaryPath) {
    try {
        $rpt = Get-Content $summaryPath -Raw | ConvertFrom-Json
        if ($rpt.n_succeeded -ge $expectedPatients) {
            Write-Log "Production already completed ($($rpt.n_succeeded)/$expectedPatients patients). Not restarting."
            Write-Log "Run scripts\uninstall_autostart.ps1 to remove the scheduled task."
            exit 0
        }
        Write-Log "roster_summary.json found but only $($rpt.n_succeeded)/$expectedPatients patients done; will resume."
    } catch {
        Write-Log "roster_summary.json exists but unparseable; proceeding with launch"
    }
}

# --- Guard 2: production already running (dashboard on 8766) -------------
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$dashboardPort/api/state" -TimeoutSec 2 -UseBasicParsing
    if ($r.StatusCode -eq 200) {
        Write-Log "PFD production dashboard already responding on port $dashboardPort; another run is live. Exiting."
        exit 0
    }
} catch { }

# --- Wait for system to settle (drivers, GPU, network) -------------------
Write-Log "waiting $bootSettleSec s for system services"
Start-Sleep -Seconds $bootSettleSec

Set-Location $projectDir
Write-Log "cwd: $projectDir"

# --- Resume PFD production (always targets synthetic_dataset/production_350) ---
# The .production_in_progress marker is written at run start and removed at
# completion.  After a power cut the marker is still on disk, so this guard
# also catches runs that were mid-flight when the machine lost power.
# If the marker is absent but roster_summary shows fewer than $expectedPatients,
# we still resume (the marker may have been deleted before the script finished).
$prodRoot   = Join-Path $projectDir 'synthetic_dataset\production_350'
$prodMarker = Join-Path $prodRoot '.production_in_progress'

$shouldResume = (Test-Path $prodMarker)
if (-not $shouldResume -and (Test-Path $summaryPath)) {
    try {
        $rpt2 = Get-Content $summaryPath -Raw | ConvertFrom-Json
        if ($rpt2.n_succeeded -lt $expectedPatients) { $shouldResume = $true }
    } catch { }
}
if (-not $shouldResume -and (Test-Path $prodRoot)) {
    # Folder exists but no summary yet -> run was started but never logged
    $shouldResume = $true
}

if ($shouldResume) {
    Write-Log "Resuming PFD production v2 -> $prodRoot"
    Write-Log "exec: $pythonExe scripts\pfd_production_v2.py --num-patients 350 --n-plain 175 --n-hilly 175 --out synthetic_dataset/production_350 --port 8766 --skip-done --no-browser"
    & $pythonExe scripts\pfd_production_v2.py `
        --num-patients 350 --n-plain 175 --n-hilly 175 `
        --out synthetic_dataset/production_350 `
        --port 8766 --skip-done --no-browser 2>&1 |
        Tee-Object -FilePath $logFile -Append
} else {
    Write-Log "PFD production complete or not yet started; checking AI generation..."
}

# --- Resume AI generation if in progress or not complete -------------------
$aiRoot    = Join-Path $projectDir 'synthetic_dataset\ai_generated_350'
$aiMarker  = Join-Path $aiRoot '.ai_generation_in_progress'
$aiSummary = Join-Path $aiRoot 'generation_summary.json'
$aiPort    = 8769
$aiExpected = 350

$aiShouldResume = $false

# Check if generation dashboard already running
try {
    $ar = Invoke-WebRequest -Uri "http://127.0.0.1:$aiPort/api/state" -TimeoutSec 2 -UseBasicParsing
    if ($ar.StatusCode -eq 200) {
        Write-Log "AI generation dashboard already responding on port $aiPort; another run is live. Exiting."
        exit 0
    }
} catch { }

if (Test-Path $aiMarker) {
    Write-Log "AI generation in-progress marker found; will resume."
    $aiShouldResume = $true
}
if (-not $aiShouldResume -and (Test-Path $aiSummary)) {
    try {
        $aiRpt = Get-Content $aiSummary -Raw | ConvertFrom-Json
        if ($aiRpt.done -lt $aiExpected) {
            Write-Log "AI summary shows $($aiRpt.done)/$aiExpected; will resume."
            $aiShouldResume = $true
        } else {
            Write-Log "AI generation already complete ($($aiRpt.done)/$aiExpected)."
        }
    } catch { $aiShouldResume = $true }
}
if (-not $aiShouldResume -and (Test-Path $aiRoot)) {
    $aiDone = (Get-ChildItem $aiRoot -Directory | Where-Object { $_.Name -match '^PF_\d+$' }).Count
    if ($aiDone -gt 0 -and $aiDone -lt $aiExpected) {
        Write-Log "AI folder exists with $aiDone/$aiExpected patients; will resume."
        $aiShouldResume = $true
    }
}

if ($aiShouldResume) {
    Write-Log "Resuming AI generation -> $aiRoot  (port $aiPort)"
    & $pythonExe scripts\run_ai_generation.py `
        --num-patients $aiExpected --port $aiPort --no-browser 2>&1 |
        Tee-Object -FilePath $logFile -Append
}

# --- Resume PFD v4 all-sources production if in progress -------------------
$v4Root    = Join-Path $projectDir 'synthetic_dataset\production_v4_350'
$v4Marker  = Join-Path $v4Root '.production_in_progress'
$v4Summary = Join-Path $v4Root 'roster_summary.json'
$v4Port    = 8770
$v4Expected = 350

$v4ShouldResume = $false
try {
    $vr = Invoke-WebRequest -Uri "http://127.0.0.1:$v4Port/api/state" -TimeoutSec 2 -UseBasicParsing
    if ($vr.StatusCode -eq 200) {
        Write-Log "PFD v4 dashboard already responding on port $v4Port; run is live. Exiting."
        exit 0
    }
} catch { }

if (Test-Path $v4Marker) { $v4ShouldResume = $true; Write-Log "PFD v4 in-progress marker found." }
if (-not $v4ShouldResume -and (Test-Path $v4Summary)) {
    try {
        $v4Rpt = Get-Content $v4Summary -Raw | ConvertFrom-Json
        if ($v4Rpt.n_succeeded -lt $v4Expected) { $v4ShouldResume = $true }
    } catch { $v4ShouldResume = $true }
}
if (-not $v4ShouldResume -and (Test-Path $v4Root)) {
    $v4Done = (Get-ChildItem $v4Root -Directory | Where-Object { $_.Name -match '^PFD_\d+' }).Count
    if ($v4Done -gt 0 -and $v4Done -lt $v4Expected) { $v4ShouldResume = $true }
}

if ($v4ShouldResume) {
    Write-Log "Resuming PFD v4 all-sources -> $v4Root"
    Set-Location $projectDir
    & $pythonExe scripts\pfd_production_v4.py `
        --num-patients $v4Expected --n-plain 175 --n-hilly 175 `
        --out synthetic_dataset/production_v4_350 `
        --port $v4Port --skip-done --no-browser 2>&1 |
        Tee-Object -FilePath $logFile -Append
}

# --- Resume PFD v6 population-split production if in progress ---------------
$v6Root    = Join-Path $projectDir 'synthetic_dataset\PFD_Synthetic_Dataset_350'
$v6Marker  = Join-Path $v6Root '.production_in_progress'
$v6Summary = Join-Path $v6Root 'roster_summary.json'
$v6Port    = 8773
$v6Expected = 350

$v6ShouldResume = $false
try {
    $v6r = Invoke-WebRequest -Uri "http://127.0.0.1:$v6Port/api/state" -TimeoutSec 2 -UseBasicParsing
    if ($v6r.StatusCode -eq 200) {
        Write-Log "PFD v6 dashboard already responding on port $v6Port; run is live. Exiting."
        exit 0
    }
} catch { }

if (Test-Path $v6Marker) { $v6ShouldResume = $true; Write-Log "PFD v6 in-progress marker found." }
if (-not $v6ShouldResume -and (Test-Path $v6Summary)) {
    try {
        $v6Rpt = Get-Content $v6Summary -Raw | ConvertFrom-Json
        if ($v6Rpt.n_succeeded -lt $v6Expected) { $v6ShouldResume = $true }
    } catch { $v6ShouldResume = $true }
}
if (-not $v6ShouldResume -and (Test-Path $v6Root)) {
    $v6Done = (Get-ChildItem (Join-Path $v6Root 'Plain_175') -Directory -ErrorAction SilentlyContinue).Count +
              (Get-ChildItem (Join-Path $v6Root 'Hilly_175') -Directory -ErrorAction SilentlyContinue).Count
    if ($v6Done -gt 0 -and $v6Done -lt $v6Expected) { $v6ShouldResume = $true }
}

if ($v6ShouldResume) {
    Write-Log "Resuming PFD v6 population-split -> $v6Root  (port $v6Port)"
    Set-Location $projectDir
    & $pythonExe scripts\pfd_production_v6.py `
        --num-patients $v6Expected --n-plain 175 --n-hilly 175 `
        --port $v6Port --skip-done --no-browser 2>&1 |
        Tee-Object -FilePath $logFile -Append
}

$rc = $LASTEXITCODE
Write-Log "pipeline process exited with code $rc"
Write-Log "if the run is not complete, this script will fire again at the next boot (or by Task Scheduler retry policy)"
exit $rc
