# autostart_real_v1.ps1 — Boot-time resume for pfd_real_v1.py
#
# Registered as a Windows Scheduled Task (trigger: At startup).
# After any power cut / reboot:
#   1. Prevents sleep (AC + display)
#   2. Starts HTTP server on port 8899
#   3. Opens progress UI in browser
#   4. Resumes pfd_real_v1.py (skips already-done patients)
#   5. Exits cleanly once all 350 are done

$ErrorActionPreference = 'Continue'

$projectDir  = 'D:\Muthu kumar\gen_ai_ct_pelvic'
$pythonExe   = 'C:\Python314\python.exe'
$bootSettle  = 45      # seconds to wait after boot for drivers
$expected    = 350
$httpPort    = 8899
$progressUrl = "http://127.0.0.1:$httpPort/synthetic_dataset/PFD_Real_Dataset_350/progress.html"

# ── Logging ───────────────────────────────────────────────────────────────────
$logDir  = Join-Path $projectDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("autostart_realv1_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")
function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}
Log "=== autostart_real_v1 triggered ==="

# ── Guard: already complete? ──────────────────────────────────────────────────
$outRoot  = Join-Path $projectDir 'synthetic_dataset\PFD_Real_Dataset_350'
$plainDir = Join-Path $outRoot 'Plain_175'
$hillyDir = Join-Path $outRoot 'Hilly_175'

function Count-Done {
    $p = 0; $h = 0
    if (Test-Path $plainDir) {
        $p = (Get-ChildItem $plainDir -Directory |
              Where-Object { Test-Path (Join-Path $_.FullName 'COMPARISON.png') }).Count
    }
    if (Test-Path $hillyDir) {
        $h = (Get-ChildItem $hillyDir -Directory |
              Where-Object { Test-Path (Join-Path $_.FullName 'COMPARISON.png') }).Count
    }
    return $p + $h
}

$alreadyDone = Count-Done
Log "Patients done so far: $alreadyDone / $expected"

if ($alreadyDone -ge $expected) {
    Log "All $expected patients complete. Nothing to do — exiting."
    exit 0
}

# ── Guard: already running? ───────────────────────────────────────────────────
$running = Get-Process python -ErrorAction SilentlyContinue |
           Where-Object { $_.CommandLine -like '*pfd_real_v1*' }
if ($running) {
    Log "pfd_real_v1.py already running (PID $($running.Id)). Exiting."
    exit 0
}

# ── Prevent sleep (AC power: never sleep, never hibernate, keep display on) ──
Log "Disabling sleep & hibernate..."
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 0
# Also keep the thread alive (SetThreadExecutionState via PowerShell)
$keepAwakeJob = Start-Job -ScriptBlock {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class PowerMgmt {
    [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS       = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED  = 0x00000001;
    public const uint ES_DISPLAY_REQUIRED = 0x00000002;
}
"@
    while ($true) {
        [PowerMgmt]::SetThreadExecutionState(
            [PowerMgmt]::ES_CONTINUOUS -bor
            [PowerMgmt]::ES_SYSTEM_REQUIRED -bor
            [PowerMgmt]::ES_DISPLAY_REQUIRED) | Out-Null
        Start-Sleep -Seconds 30
    }
}
Log "Keep-awake job started (ID $($keepAwakeJob.Id))"

# ── Boot settle ───────────────────────────────────────────────────────────────
Log "Waiting $bootSettle s for system to settle..."
Start-Sleep -Seconds $bootSettle

Set-Location $projectDir

# ── Start HTTP server (kill stale instances first) ────────────────────────────
Log "Starting HTTP server on port $httpPort..."
Get-NetTCPConnection -LocalPort $httpPort -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 1
Start-Process $pythonExe `
    -ArgumentList "-m","http.server",$httpPort `
    -WorkingDirectory $projectDir `
    -WindowStyle Hidden
Start-Sleep -Seconds 3
Log "HTTP server started"

# ── Open progress UI in browser ───────────────────────────────────────────────
Log "Opening progress UI: $progressUrl"
Start-Process $progressUrl
Start-Sleep -Seconds 2

# ── Run pfd_real_v1.py ────────────────────────────────────────────────────────
$genLog = Join-Path $projectDir 'logs\pfd_real_v1_new.log'
$genErr = Join-Path $projectDir 'logs\pfd_real_v1_err.log'
Log "Starting pfd_real_v1.py (log -> $genLog)"

$env:PYTHONUTF8    = "1"
$env:PYTHONUNBUFFERED = "1"

& $pythonExe -u (Join-Path $projectDir 'scripts\pfd_real_v1.py') `
    *>> $genLog

$rc = $LASTEXITCODE
Log "pfd_real_v1.py exited with code $rc"

# ── Check completion ──────────────────────────────────────────────────────────
$finalDone = Count-Done
Log "Final count: $finalDone / $expected"

if ($finalDone -ge $expected) {
    Log "SUCCESS — all $expected patients generated."
    # Restore default power settings
    powercfg /change standby-timeout-ac 30
    powercfg /change monitor-timeout-ac 15
} else {
    Log "INCOMPLETE ($finalDone/$expected). Will retry at next boot via scheduled task."
}

Stop-Job $keepAwakeJob -ErrorAction SilentlyContinue
Remove-Job $keepAwakeJob -ErrorAction SilentlyContinue
Log "Done."
exit $rc
