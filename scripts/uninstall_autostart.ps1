# Remove the auto-restart scheduled task.
#
# Run in an ELEVATED PowerShell window:
#   powershell -ExecutionPolicy Bypass -File scripts\uninstall_autostart.ps1

$taskName = 'PelvicCT-Generator-Pipeline'

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    throw "Must run from an ELEVATED PowerShell (admin). Right-click PowerShell -> Run as administrator."
}

if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed scheduled task: $taskName"
} else {
    Write-Host "No scheduled task named '$taskName' found. Nothing to do."
}
