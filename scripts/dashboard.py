# -*- coding: utf-8 -*-
"""dashboard.py — Live ETA dashboard for PFD CT generation.
Run:  python scripts/dashboard.py
Opens browser at http://localhost:8765  (auto-refreshes every 10 s)
"""
import json, os, sys, time, threading, webbrowser
import numpy as np
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime, timedelta

BASE   = Path(__file__).resolve().parent.parent
LOG    = BASE / "logs" / "pfd_full_run.log"
PLAIN  = BASE / "synthetic_dataset" / "PFD_Real_Dataset_350" / "Plain_175"
HILLY  = BASE / "synthetic_dataset" / "PFD_Real_Dataset_350" / "Hilly_175"
N_TOTAL = 350
PORT    = 8765

# ── Data collection ───────────────────────────────────────────────────────────

def _completed(pop_dir: Path):
    """Return list of (patient_id, mtime) for completed patients."""
    out = []
    if not pop_dir.exists():
        return out
    for d in pop_dir.iterdir():
        if not d.is_dir():
            continue
        meta = d / "metadata_real.json"
        if meta.exists():
            out.append((d.name, meta.stat().st_mtime))
    return sorted(out, key=lambda x: x[1])


def _qa_stats(pop_dir: Path):
    """Count qa_attempts from metadata files. Returns (total_attempts, patients)."""
    total_att, patients = 0, 0
    if not pop_dir.exists():
        return 0, 0
    for meta in pop_dir.glob("*/metadata_real.json"):
        try:
            d = json.loads(meta.read_text())
            total_att += d.get("qa_attempts", 1)
            patients  += 1
        except Exception:
            pass
    return total_att, patients


def _read_log(n=60):
    """Read last n lines from the log, handling utf-16 / utf-8 / latin-1."""
    if not LOG.exists():
        return ["Log not found — generation may not have started yet."]
    for enc in ("utf-16", "utf-8-sig", "utf-8", "latin-1"):
        try:
            text = LOG.read_text(encoding=enc)
            # utf-16 sometimes has wide-spaced chars — strip them
            if enc == "utf-16" and " " * 3 in text[:200]:
                text = text.replace(" ", "")
            lines = [l.rstrip() for l in text.splitlines() if l.strip()]
            return lines[-n:]
        except Exception:
            continue
    return ["Could not decode log file."]


def _current_patient(log_lines):
    """Detect in-progress patient: folder exists but no metadata_real.json yet."""
    # Prefer scanning folders over log (log may not be redirected)
    for pop_dir in [PLAIN, HILLY]:
        if not pop_dir.exists():
            continue
        for d in sorted(pop_dir.iterdir()):
            if d.is_dir() and not (d / "metadata_real.json").exists():
                # Check it has some files (truly in-progress, not empty failed folder)
                any_file = any(d.rglob("*"))
                if any_file:
                    elapsed = time.time() - d.stat().st_mtime
                    em = int(elapsed // 60); es = int(elapsed % 60)
                    return f"{d.name}  (in progress — {em}m {es}s elapsed)"
    # Fall back to log
    for line in reversed(log_lines):
        if line.startswith("[") and "/350]" in line:
            return line.strip("[] ").split("]")[-1].strip()
        if "REAL_" in line and ("Loading" in line or "Deforming" in line or "Writing" in line):
            return line.strip()
    return "Idle / scanning sources…"


def _eta(plain_done, hilly_done):
    """Compute speed + ETA using MEDIAN inter-arrival time to eliminate outliers."""
    all_done = sorted(
        [t for _, t in plain_done] + [t for _, t in hilly_done]
    )
    n_done = len(all_done)
    n_left = N_TOTAL - n_done

    if n_done < 2:
        return n_done, n_left, None, None, None

    # Compute per-patient intervals
    intervals = [all_done[i+1] - all_done[i] for i in range(len(all_done)-1)]

    # Use last 10 intervals (most recent pace), then take median to drop outliers
    recent = intervals[max(0, len(intervals)-10):]
    median_secs = float(np.median(recent))
    mean_secs   = float(np.mean(recent))

    # Use the SLOWER of median/mean so ETA is conservative but not blown out by 1 outlier
    secs_per_patient = max(median_secs, min(mean_secs, median_secs * 3))
    pph = 3600 / max(secs_per_patient, 1)

    eta_secs = n_left * secs_per_patient
    eta_dt   = datetime.now() + timedelta(seconds=eta_secs)

    return n_done, n_left, pph, eta_dt, median_secs


# ── HTML builder ──────────────────────────────────────────────────────────────

def _bar(val, total, color="#4ade80"):
    pct = round(val / max(total, 1) * 100, 1)
    return f"""
<div class="bar-wrap">
  <div class="bar-fill" style="width:{pct}%;background:{color}"></div>
  <span class="bar-label">{val} / {total} &nbsp; ({pct}%)</span>
</div>"""


def build_page():
    plain_done = _completed(PLAIN)
    hilly_done = _completed(HILLY)
    n_plain    = len(plain_done)
    n_hilly    = len(hilly_done)
    n_done, n_left, pph, eta_dt, med_secs = _eta(plain_done, hilly_done)

    qa_att_p, qa_pat_p = _qa_stats(PLAIN)
    qa_att_h, qa_pat_h = _qa_stats(HILLY)
    total_att  = qa_att_p + qa_att_h
    total_pats = qa_pat_p + qa_pat_h
    qa_rej     = max(0, total_att - total_pats)

    log_lines   = _read_log(50)
    cur_patient = _current_patient(log_lines)

    # Detect TotalSegmentator (multiprocessing workers)
    ts_workers = 0
    try:
        import subprocess, re
        r = subprocess.run(
            ["wmic","process","where","name='python.exe'","get","commandline"],
            capture_output=True, text=True, timeout=5)
        ts_workers = len(re.findall(r'multiprocessing-fork', r.stdout))
    except Exception:
        pass

    pph_str    = f"{pph:.1f} pts/hr" if pph else "—"
    med_str    = f"{int(med_secs//60)}m {int(med_secs%60)}s" if med_secs else "—"
    if eta_dt:
        eta_str  = eta_dt.strftime("%d %b %H:%M")
        rem_secs = (eta_dt - datetime.now()).total_seconds()
        rem_h    = int(rem_secs // 3600)
        rem_m    = int((rem_secs % 3600) // 60)
        rem_str  = f"{rem_h}h {rem_m}m remaining"
    else:
        eta_str = "Calculating…"
        rem_str = "Need ≥2 completed patients"

    # colour-code log lines
    def _fmt(line):
        cls = ""
        if "QA PASS" in line or "OK ->" in line:
            cls = "log-ok"
        elif "QA FAIL" in line or "SKIP" in line or "FAIL" in line or "ERROR" in line:
            cls = "log-warn"
        elif "Deforming" in line or "Writing DICOM" in line or "Loading" in line:
            cls = "log-info"
        line_esc = line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        return f'<span class="{cls}">{line_esc}</span>'

    log_html = "\n".join(_fmt(l) for l in log_lines)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PFD CT Generation Dashboard</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0f1117;color:#e2e8f0;font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;min-height:100vh}}
header{{background:linear-gradient(135deg,#1e3a5f,#0f2942);padding:18px 28px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #1e3a5f}}
header h1{{font-size:20px;font-weight:700;color:#7dd3fc;letter-spacing:.5px}}
header .ts{{font-size:12px;color:#64748b}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;padding:20px 24px}}
.card{{background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;padding:18px 20px}}
.card h2{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:#64748b;margin-bottom:14px}}
.big-num{{font-size:42px;font-weight:800;line-height:1;margin-bottom:4px}}
.sub{{font-size:12px;color:#94a3b8;margin-top:4px}}
.bar-wrap{{position:relative;background:#1e293b;border-radius:6px;height:22px;overflow:hidden;margin-top:10px}}
.bar-fill{{height:100%;border-radius:6px;transition:width .4s ease}}
.bar-label{{position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:12px;font-weight:600;color:#fff;text-shadow:0 1px 2px #000}}
.eta-time{{font-size:28px;font-weight:700;color:#f59e0b;margin-top:6px}}
.eta-rem{{font-size:13px;color:#94a3b8;margin-top:4px}}
.cur-pat{{font-size:12px;color:#a5f3fc;word-break:break-all;padding:8px 10px;background:#0f1a2e;border-radius:6px;margin-top:8px;line-height:1.5}}
.log-box{{grid-column:1/-1;background:#0d1117;border:1px solid #2d3748;border-radius:12px;padding:16px 18px}}
.log-box h2{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.8px;color:#64748b;margin-bottom:10px}}
.log-inner{{font-family:'Cascadia Code','Consolas',monospace;font-size:12px;line-height:1.6;max-height:340px;overflow-y:auto;white-space:pre-wrap;word-break:break-all}}
.log-ok{{color:#4ade80}}
.log-warn{{color:#f87171}}
.log-info{{color:#93c5fd}}
.badge{{display:inline-block;border-radius:4px;padding:2px 7px;font-size:11px;font-weight:600;margin-left:6px}}
.badge-green{{background:#14532d;color:#4ade80}}
.badge-red{{background:#450a0a;color:#f87171}}
.badge-blue{{background:#172554;color:#93c5fd}}
.stat-row{{display:flex;justify-content:space-between;margin-top:8px;font-size:13px;color:#cbd5e1}}
.stat-row span:last-child{{font-weight:700;color:#e2e8f0}}
footer{{padding:12px 24px;text-align:center;font-size:11px;color:#334155;border-top:1px solid #1e293b}}
</style>
</head>
<body>
<header>
  <h1>&#9881; PFD CT Generation &mdash; Live Dashboard</h1>
  <span class="ts">Auto-refreshes every 10s &bull; Last update: {now_str}</span>
</header>

<div class="grid">

  <!-- Overall Progress -->
  <div class="card">
    <h2>Overall Progress</h2>
    <div class="big-num" style="color:#7dd3fc">{n_done}</div>
    <div class="sub">of {N_TOTAL} patients generated</div>
    {_bar(n_done, N_TOTAL, "#3b82f6")}
  </div>

  <!-- Plain -->
  <div class="card">
    <h2>Plain (Gynecoid) &nbsp;<span class="badge badge-blue">175 target</span></h2>
    <div class="big-num" style="color:#4ade80">{n_plain}</div>
    <div class="sub">patients completed</div>
    {_bar(n_plain, 175, "#22c55e")}
  </div>

  <!-- Hilly -->
  <div class="card">
    <h2>Hilly (Android) &nbsp;<span class="badge badge-blue">175 target</span></h2>
    <div class="big-num" style="color:#a78bfa">{n_hilly}</div>
    <div class="sub">patients completed</div>
    {_bar(n_hilly, 175, "#8b5cf6")}
  </div>

  <!-- ETA -->
  <div class="card">
    <h2>ETA &amp; Speed <span style="font-size:10px;color:#64748b">(median-based, outliers removed)</span></h2>
    <div class="eta-time">&#128336; {eta_str}</div>
    <div class="eta-rem">{rem_str}</div>
    <div class="stat-row" style="margin-top:14px">
      <span>Speed</span><span>{pph_str}</span>
    </div>
    <div class="stat-row">
      <span>Median time/patient</span><span>{med_str}</span>
    </div>
    <div class="stat-row">
      <span>Remaining</span><span>{n_left} patients</span>
    </div>
    {'<div class="stat-row" style="margin-top:8px;padding:6px 8px;background:#1c1a0d;border-radius:5px;border:1px solid #854d0e"><span style="color:#fbbf24">&#9888; TotalSegmentator running</span><span style="color:#fbbf24">' + str(ts_workers) + ' workers</span></div>' if ts_workers > 0 else '<div class="stat-row" style="margin-top:8px;padding:6px 8px;background:#0d1c12;border-radius:5px"><span style="color:#4ade80">&#10003; Masks cached — fast mode</span><span></span></div>'}
  </div>

  <!-- QA Stats -->
  <div class="card">
    <h2>QA Stats</h2>
    <div class="stat-row">
      <span>Patients accepted</span>
      <span>{total_pats} <span class="badge badge-green">PASS</span></span>
    </div>
    <div class="stat-row">
      <span>QA-rejected attempts</span>
      <span>{qa_rej} <span class="badge badge-red">FAIL</span></span>
    </div>
    <div class="stat-row">
      <span>Total source tries</span>
      <span>{total_att}</span>
    </div>
    <div class="stat-row">
      <span>Avg tries per patient</span>
      <span>{"%.1f" % (total_att / max(total_pats,1))}</span>
    </div>
  </div>

  <!-- Current Activity -->
  <div class="card">
    <h2>Current Activity</h2>
    <div style="color:#fbbf24;font-size:13px;margin-top:4px">&#9679; Generating&hellip;</div>
    <div class="cur-pat">{cur_patient}</div>
    <div class="stat-row" style="margin-top:12px">
      <span>Log file</span><span style="color:#4ade80">{'&#10003; Found' if LOG.exists() else '&#10007; Missing'}</span>
    </div>
    <div class="stat-row">
      <span>Plain folder</span><span style="color:#4ade80">{'&#10003;' if PLAIN.exists() else '&#10007;'}</span>
    </div>
    <div class="stat-row">
      <span>Hilly folder</span><span style="color:#4ade80">{'&#10003;' if HILLY.exists() else '&#10007;'}</span>
    </div>
  </div>

  <!-- Log -->
  <div class="log-box">
    <h2>Live Log (last 50 lines)</h2>
    <div class="log-inner">{log_html}</div>
  </div>

</div>
<footer>PFD Synthetic CT Pipeline &bull; 175 Plain + 175 Hilly = 350 patients &bull; Pelvic Floor Disorder Study</footer>
<script>
// Auto-scroll log to bottom
document.querySelector('.log-inner').scrollTop = 99999;
</script>
</body>
</html>"""


# ── HTTP server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        html = build_page().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def log_message(self, *_):
        pass  # suppress access log noise


def _open_browser():
    time.sleep(1.2)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open_browser, daemon=True).start()
    server = HTTPServer(("localhost", PORT), Handler)
    print(f"Dashboard running at http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
