# Open the pelvic-CT pipeline dashboard in the default browser.
#
# Fires from the "PelvicCT-Generator-Dashboard" scheduled task at user logon
# (installed by scripts/install_autostart.ps1). After power-on:
#   1. autostart.ps1 launches the pipeline at system startup (no browser)
#   2. user logs in
#   3. this script fires, waits up to 10 min for http://127.0.0.1:8765/ to
#      respond, then opens the URL in the default browser.
#
# If the dashboard never comes up, the script gives up silently after 10 min
# so it doesn't pop a stuck browser tab.

$ErrorActionPreference = 'Continue'

$projectDir   = 'D:\Muthu kumar\gen_ai_ct_pelvic'
$dashboardUrl = 'http://127.0.0.1:8765/'
$waitSeconds  = 600     # 10 minutes -- enough for the pipeline to import torch
                        # + load datasets after a fresh boot
$pollInterval = 3

$logDir = Join-Path $projectDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("open_dashboard_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")
function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

Write-Log "waiting for dashboard at $dashboardUrl (up to $waitSeconds s)"
$deadline = (Get-Date).AddSeconds($waitSeconds)
$found = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "$dashboardUrl`api/state" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            $found = $true
            break
        }
    } catch {
        Start-Sleep -Seconds $pollInterval
    }
}

if ($found) {
    Write-Log "dashboard up; launching browser"
    try {
        Start-Process $dashboardUrl
        Write-Log "browser launched"
    } catch {
        Write-Log "Start-Process failed: $_"
    }
} else {
    Write-Log "gave up after $waitSeconds s; dashboard never responded"
}
