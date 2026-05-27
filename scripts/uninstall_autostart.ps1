# Remove BOTH auto-restart scheduled tasks (pipeline + dashboard).
#
# Run in an ELEVATED PowerShell window:
#   powershell -ExecutionPolicy Bypass -File scripts\uninstall_autostart.ps1

$tasks = @('PelvicCT-Generator-Pipeline', 'PelvicCT-Generator-Dashboard')

$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Not elevated -- requesting admin privileges via UAC..."
    $argList = "-ExecutionPolicy Bypass -NoExit -File `"$PSCommandPath`""
    try {
        Start-Process powershell -Verb RunAs -ArgumentList $argList
        Start-Sleep -Seconds 3
        exit 0
    } catch {
        throw "Failed to self-elevate: $_`nOpen PowerShell as Administrator manually and re-run."
    }
}

foreach ($t in $tasks) {
    if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $t -Confirm:$false
        Write-Host "Removed scheduled task: $t"
    } else {
        Write-Host "No task named '$t' found."
    }
}
