# Auto-launch wrapper for the synthetic pelvic CT pipeline.
#
# Fires from the "PelvicCT-Generator-Pipeline" Windows scheduled task at boot
# (install via scripts/install_autostart.ps1). After a power cut / restart /
# holiday, this script:
#
#   1. Checks if the pipeline already finished (anatomy_report.json with the
#      expected patient count) -> exits, leaves the box alone.
#   2. Checks if the pipeline is already running on the dashboard port ->
#      exits, no double-launch.
#   3. Otherwise waits 60 s for system services to settle, then runs
#      `python scripts\run_all.py --preset overnight --device cuda --skip
#      cache,labels --no-browser`. The --resume default in run_all.py picks
#      up training from the highest-numbered epoch checkpoint on disk.
#
# Per-boot log: logs\autostart_YYYYMMDD_HHMMSS.log under the project dir.

$ErrorActionPreference = 'Continue'

# --- Edit these two paths if you move the project or use a different Python ---
$projectDir = 'D:\Muthu kumar\gen_ai_ct_pelvic'
$pythonExe  = 'C:\Python314\python.exe'
# --- Pipeline command (kept here so it's easy to change without editing the task) ---
$preset     = 'overnight'
$expectedPatients = 50      # treat run as "done" once anatomy_report has this many
$dashboardPort    = 8765
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

# --- Guard 1: pipeline already completed ----------------------------------
$doneMarker = Join-Path $projectDir 'synthetic_dataset\anatomy_report.json'
if (Test-Path $doneMarker) {
    try {
        $rpt = Get-Content $doneMarker -Raw | ConvertFrom-Json
        if ($rpt.n_patients -ge $expectedPatients) {
            Write-Log "pipeline already completed ($($rpt.n_patients)/$expectedPatients patients). Not restarting. Run scripts\uninstall_autostart.ps1 to clean up the scheduled task."
            exit 0
        }
    } catch {
        Write-Log "anatomy_report.json exists but unparseable; proceeding with launch"
    }
}

# --- Guard 2: pipeline already running (dashboard answers) ---------------
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$dashboardPort/api/state" -TimeoutSec 2 -UseBasicParsing
    if ($r.StatusCode -eq 200) {
        Write-Log "dashboard already responding on port $dashboardPort; another pipeline is running. Exiting."
        exit 0
    }
} catch { }

# --- Wait for system to settle (drivers, GPU, network) -------------------
Write-Log "waiting $bootSettleSec s for system services"
Start-Sleep -Seconds $bootSettleSec

# --- Launch the pipeline -------------------------------------------------
Set-Location $projectDir
Write-Log "cwd: $projectDir"
Write-Log "exec: $pythonExe scripts\run_all.py --preset $preset --device cuda --skip cache,labels --no-browser"
& $pythonExe scripts\run_all.py --preset $preset --device cuda --skip cache,labels --no-browser 2>&1 |
    Tee-Object -FilePath $logFile -Append

$rc = $LASTEXITCODE
Write-Log "pipeline process exited with code $rc"
Write-Log "if the run is not complete, this script will fire again at the next boot (or by Task Scheduler retry policy)"
exit $rc
