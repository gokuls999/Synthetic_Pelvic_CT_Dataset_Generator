# Register a Windows scheduled task that runs scripts/autostart.ps1 at every
# system boot. Combined with the --resume default in run_all.py, this means
# the pipeline survives unattended power cuts / reboots during holidays:
#
#   power back on  ->  Windows boots  ->  task fires  ->  autostart.ps1
#   ->  resume training from last epoch checkpoint  ->  dashboard back up.
#
# Run this script ONCE in an ELEVATED PowerShell window:
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
#
# To uninstall later:   scripts\uninstall_autostart.ps1

$ErrorActionPreference = 'Stop'

$taskName     = 'PelvicCT-Generator-Pipeline'
$projectDir   = 'D:\Muthu kumar\gen_ai_ct_pelvic'
$scriptPath   = Join-Path $projectDir 'scripts\autostart.ps1'
$maxRuntime   = New-TimeSpan -Days 30      # let it run for weeks if needed
$restartGap   = New-TimeSpan -Minutes 5    # gap between automatic retries

if (-not (Test-Path $scriptPath)) {
    throw "autostart.ps1 not found at $scriptPath"
}

# Are we admin? S4U / "run whether user is logged in or not" needs it.
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "Must run from an ELEVATED PowerShell (admin). Right-click PowerShell -> Run as administrator."
}

# Action: run our wrapper script under powershell.exe, hidden.
$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -NonInteractive -File `"$scriptPath`"" `
    -WorkingDirectory $projectDir

# Trigger: at every system startup. Also add a logon trigger as belt-and-braces
# so it kicks in if the system is already up but the user just logged on.
$triggerStart = New-ScheduledTaskTrigger -AtStartup
$triggerLogon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Settings: keep it running on battery, restart on failure a few times.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 -RestartInterval $restartGap `
    -ExecutionTimeLimit $maxRuntime `
    -MultipleInstances IgnoreNew

# Principal: run as the current user with highest privileges, even if not
# logged in (S4U). No stored password.
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME -LogonType S4U -RunLevel Highest

# Register (replace if exists)
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task '$taskName' before reinstall..."
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask -TaskName $taskName `
    -Action $action `
    -Trigger @($triggerStart, $triggerLogon) `
    -Settings $settings `
    -Principal $principal `
    -Description "Auto-launch pelvic CT generator pipeline (resume-on-restart)."

Write-Host ""
Write-Host "Installed scheduled task: $taskName"
Write-Host "  Triggers: At system startup AND at logon ($env:USERNAME)"
Write-Host "  Action:   $scriptPath"
Write-Host "  Restart on failure: 3 times, 5 min apart"
Write-Host ""
Write-Host "Useful commands:"
Write-Host "  Inspect:   Get-ScheduledTask -TaskName '$taskName' | Get-ScheduledTaskInfo"
Write-Host "  Run now:   Start-ScheduledTask -TaskName '$taskName'"
Write-Host "  Stop now:  Stop-ScheduledTask -TaskName '$taskName'"
Write-Host "  Remove:    powershell -ExecutionPolicy Bypass -File scripts\uninstall_autostart.ps1"
Write-Host ""
Write-Host "The task will skip launching if:"
Write-Host "  - synthetic_dataset/anatomy_report.json shows the run is already complete"
Write-Host "  - another pipeline is already responding on http://127.0.0.1:8765"
