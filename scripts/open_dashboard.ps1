# Open the active pipeline dashboard in the default browser at user logon.
#
# Fires from the "PelvicCT-Generator-Dashboard" scheduled task (installed by
# scripts/install_autostart.ps1). Checks two ports in priority order:
#
#   Port 8766  -- PFD production dashboard (pfd_production_v2.py)
#   Port 8765  -- Training dashboard (run_all.py / train.py)
#
# Whichever responds first is opened.  If neither responds within
# $waitSeconds the script exits silently (no stuck browser tab).
#
# Per-logon log: logs\open_dashboard_YYYYMMDD_HHMMSS.log

$ErrorActionPreference = 'Continue'

$projectDir   = 'D:\Muthu kumar\gen_ai_ct_pelvic'
$waitSeconds  = 600     # 10 min -- enough for Python/torch import after cold boot
$pollInterval = 3

# Ports to check, in priority order (AI generation first, then PFD production, then training).
$candidates = @(
    [pscustomobject]@{ Port=8770; Label='PFD v4 All-sources' },
    [pscustomobject]@{ Port=8769; Label='AI Generation'      },
    [pscustomobject]@{ Port=8766; Label='PFD Production'     },
    [pscustomobject]@{ Port=8765; Label='Training'           }
)

$logDir = Join-Path $projectDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("open_dashboard_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")
function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

Write-Log "Dashboard watcher started. Polling ports: $($candidates.Port -join ', ') for up to $waitSeconds s"

$deadline = (Get-Date).AddSeconds($waitSeconds)
$found    = $null   # will hold the winning candidate

while ((Get-Date) -lt $deadline -and $null -eq $found) {
    foreach ($c in $candidates) {
        try {
            $url   = "http://127.0.0.1:$($c.Port)/"
            $state = "${url}api/state"
            $r = Invoke-WebRequest -Uri $state -TimeoutSec 2 -UseBasicParsing
            if ($r.StatusCode -eq 200) {
                $found = [pscustomobject]@{ Url=$url; Label=$c.Label; Port=$c.Port }
                break
            }
        } catch { }
    }
    if ($null -eq $found) {
        Start-Sleep -Seconds $pollInterval
    }
}

if ($null -ne $found) {
    Write-Log "$($found.Label) dashboard up on port $($found.Port); opening $($found.Url)"
    try {
        Start-Process $found.Url
        Write-Log "Browser launched -> $($found.Url)"
    } catch {
        Write-Log "Start-Process failed: $_"
    }
} else {
    Write-Log "No dashboard responded after $waitSeconds s. Pipeline may still be starting -- check logs."
}
