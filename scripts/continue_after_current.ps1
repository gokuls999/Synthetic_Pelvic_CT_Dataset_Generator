# continue_after_current.ps1
#
# Runs silently in the background.  Waits for the active pfd_production_v6.py
# run to complete, reads how many patients succeeded, then launches a
# continuation pass (--skip-done) in the SAME output folder to fill the gap
# up to 350 total successes.
#
# Safe to start at any time - it does nothing until the current run finishes.

$ErrorActionPreference = 'Continue'
$projectDir  = 'D:\Muthu kumar\gen_ai_ct_pelvic'
$pythonExe   = 'C:\Python314\python.exe'
$outDir      = 'synthetic_dataset\PFD_Synthetic_Dataset_350'
$outRoot     = Join-Path $projectDir $outDir
$marker      = Join-Path $outRoot '.production_in_progress'
$summaryFile = Join-Path $outRoot 'roster_summary.json'
$port        = 8773
$targetOK    = 350         # how many successful patients we want in total
$safetyMult  = 1.45        # pad extra slots to absorb ~25-30% TotalSeg failure rate

$logDir  = Join-Path $projectDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir ("continuation_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")

function Write-Log($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $logFile -Value $line
}

Write-Log "Continuation watcher started. Waiting for current production run to finish..."
Write-Log "  Watching for marker: $marker"

# ── Wait for the in-progress marker to disappear ──────────────────────────
$pollSec = 30
while ($true) {
    if (-not (Test-Path $marker)) { break }
    Write-Log "  Still running... (polls every ${pollSec}s)"
    Start-Sleep -Seconds $pollSec
}

Write-Log "Marker gone — current run finished. Reading results..."

# ── Read success count from roster_summary.json ───────────────────────────
$nOK      = 0
$nPlainOK = 0
$nHillyOK = 0
if (Test-Path $summaryFile) {
    try {
        $rpt      = Get-Content $summaryFile -Raw | ConvertFrom-Json
        $nOK      = [int]$rpt.n_succeeded
        $nPlainOK = [int]$rpt.n_plain
        $nHillyOK = [int]$rpt.n_hilly
        Write-Log "  Succeeded: $nOK  (plain=$nPlainOK  hilly=$nHillyOK)"
    } catch {
        Write-Log "  WARNING: could not parse roster_summary.json; assuming 0 successes"
    }
} else {
    Write-Log "  WARNING: roster_summary.json not found; assuming 0 successes"
}

if ($nOK -ge $targetOK) {
    Write-Log "Already have $nOK >= $targetOK successful patients. No continuation needed."
    exit 0
}

# ── Calculate how many extra slots to generate ────────────────────────────
$needed      = $targetOK - $nOK                        # successes still required
$extraSlots  = [int][Math]::Ceiling($needed * $safetyMult)  # slots to add (with safety margin)

# Count patient folders that already exist in both population subdirs
$existingPlain = @(Get-ChildItem (Join-Path $outRoot 'Plain_175') -Directory -ErrorAction SilentlyContinue).Count
$existingHilly = @(Get-ChildItem (Join-Path $outRoot 'Hilly_175') -Directory -ErrorAction SilentlyContinue).Count
$existingTotal = $existingPlain + $existingHilly

# New total slot count the roster needs to cover
$newTotal = $existingTotal + $extraSlots

# ALWAYS keep the same n-plain/n-hilly split as the original run (175/175).
# Only increase --num-patients.  Changing the split causes _plan_roster() to
# reassign pattern+grade to existing slots, producing new IDs that --skip-done
# cannot match, causing full regeneration.
$newPlain = 175
$newHilly = 175

Write-Log "Need $needed more successes -> adding $extraSlots slots"
Write-Log "  Existing folders : plain=$existingPlain  hilly=$existingHilly"
Write-Log "  New roster total : $newTotal  (plain=$newPlain  hilly=$newHilly fixed)"

# ── Wait until the dashboard port is free (previous process fully exited) ─
$waitedFor = 0
while ($waitedFor -lt 60) {
    try {
        $chk = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/state" -TimeoutSec 2 -UseBasicParsing
        if ($chk.StatusCode -eq 200) {
            Write-Log "  Dashboard still alive on port $port, waiting 10 s..."
            Start-Sleep -Seconds 10
            $waitedFor += 10
            continue
        }
    } catch { break }
}

# ── Launch continuation ────────────────────────────────────────────────────
Write-Log "Launching continuation pass..."
Write-Log "  cmd: $pythonExe scripts\pfd_production_v6.py --num-patients $newTotal --n-plain $newPlain --n-hilly $newHilly --out $outDir --port $port --skip-done --no-browser"

Set-Location $projectDir

& $pythonExe scripts\pfd_production_v6.py `
    --num-patients $newTotal `
    --n-plain      $newPlain `
    --n-hilly      $newHilly `
    --out          $outDir `
    --port         $port `
    --skip-done `
    --no-browser 2>&1 | Tee-Object -FilePath $logFile -Append

$rc = $LASTEXITCODE
Write-Log "Continuation finished with exit code $rc"
exit $rc
