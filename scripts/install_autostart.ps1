# Register Windows scheduled tasks that survive power cuts / reboots:
#
#   PelvicCT-Generator-Pipeline   (at system startup)
#     Runs scripts/autostart.ps1 -- launches run_all.py with --resume.
#
#   PelvicCT-Generator-Dashboard  (at user logon)
#     Runs scripts/open_dashboard.ps1 -- waits for the dashboard server to
#     respond on http://127.0.0.1:8765/ then opens it in the default browser.
#
# So after power-on:
#   boot -> pipeline relaunches in background (resumes from last epoch)
#   logon -> dashboard tab opens by itself when the server is ready.
#
# Run ONCE in an ELEVATED PowerShell window (right-click PowerShell -> Run as
# administrator):
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_autostart.ps1
#
# To uninstall:   powershell -ExecutionPolicy Bypass -File scripts\uninstall_autostart.ps1

$ErrorActionPreference = 'Stop'

$projectDir         = 'D:\Muthu kumar\gen_ai_ct_pelvic'
$pipelineTaskName   = 'PelvicCT-Generator-Pipeline'
$dashboardTaskName  = 'PelvicCT-Generator-Dashboard'
$autostartPs1       = Join-Path $projectDir 'scripts\autostart.ps1'
$openDashboardPs1   = Join-Path $projectDir 'scripts\open_dashboard.ps1'
$maxRuntime         = New-TimeSpan -Days 30
$restartGap         = New-TimeSpan -Minutes 5

if (-not (Test-Path $autostartPs1))      { throw "Not found: $autostartPs1" }
if (-not (Test-Path $openDashboardPs1))  { throw "Not found: $openDashboardPs1" }

# Admin check -- if not elevated, relaunch ourselves with Start-Process -Verb RunAs
# (triggers the UAC prompt) and exit the unprivileged copy.
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Not elevated -- requesting admin privileges via UAC..."
    $argList = "-ExecutionPolicy Bypass -NoExit -File `"$PSCommandPath`""
    try {
        Start-Process powershell -Verb RunAs -ArgumentList $argList
        Write-Host "A new elevated PowerShell window opened. Click 'Yes' on the UAC prompt if you haven't already."
        Write-Host "This window will close in 5 seconds."
        Start-Sleep -Seconds 5
        exit 0
    } catch {
        throw "Failed to self-elevate: $_`nOpen PowerShell as Administrator manually and re-run."
    }
}

function Register-PelvicCTTask {
    param(
        [string]   $TaskName,
        [string]   $ScriptPath,
        [object[]] $Triggers,
        [bool]     $RunInteractive
    )
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Write-Host "Removing existing task '$TaskName' before reinstall..."
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    $action = New-ScheduledTaskAction `
        -Execute 'powershell.exe' `
        -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -NonInteractive -File `"$ScriptPath`"" `
        -WorkingDirectory $projectDir

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 -RestartInterval $restartGap `
        -ExecutionTimeLimit $maxRuntime `
        -MultipleInstances IgnoreNew

    # Pipeline task: S4U so it can fire at boot even if no one is logged in.
    # Dashboard task: must be interactive to open a browser, so use Interactive logon.
    if ($RunInteractive) {
        $principal = New-ScheduledTaskPrincipal `
            -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
    } else {
        $principal = New-ScheduledTaskPrincipal `
            -UserId $env:USERNAME -LogonType S4U -RunLevel Highest
    }

    Register-ScheduledTask -TaskName $TaskName `
        -Action $action `
        -Trigger $Triggers `
        -Settings $settings `
        -Principal $principal `
        -Description "Pelvic CT generator pipeline ($TaskName)" | Out-Null
    Write-Host "Installed: $TaskName"
}

# 1) Pipeline launch at every system startup.
Register-PelvicCTTask `
    -TaskName  $pipelineTaskName `
    -ScriptPath $autostartPs1 `
    -Triggers  @(New-ScheduledTaskTrigger -AtStartup) `
    -RunInteractive $false

# 2) Dashboard browser opens at user logon (interactive session).
Register-PelvicCTTask `
    -TaskName  $dashboardTaskName `
    -ScriptPath $openDashboardPs1 `
    -Triggers  @(New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME) `
    -RunInteractive $true

Write-Host ""
Write-Host "Both scheduled tasks installed."
Write-Host ""
Write-Host "What will happen after you power off and back on:"
Write-Host "  Windows boots -> '$pipelineTaskName' fires -> pipeline resumes in background."
Write-Host "  You log in    -> '$dashboardTaskName' fires -> waits up to 10 min for the"
Write-Host "                   dashboard server, then opens http://127.0.0.1:8765/ in your"
Write-Host "                   default browser."
Write-Host ""
Write-Host "Test without rebooting:"
Write-Host "  Start-ScheduledTask -TaskName '$pipelineTaskName'"
Write-Host "  Start-ScheduledTask -TaskName '$dashboardTaskName'"
Write-Host ""
Write-Host "Remove when training is done:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\uninstall_autostart.ps1"
Write-Host ""
Write-Host "Per-trigger logs land under: $projectDir\logs\"
