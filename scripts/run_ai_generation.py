"""Power-cut-resilient AI generation runner with live web dashboard.

Usage:
    python scripts/run_ai_generation.py [--num-patients 350] [--port 8769] [--no-browser]

State is written to synthetic_dataset/ai_generated_350/progress.json after every
patient so the HTTP server never contends with the generation loop.

Auto-resumes: already-complete PF_XXXX folders are skipped on restart.
Dashboard:    http://127.0.0.1:8769/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR      = ROOT / "synthetic_dataset" / "ai_generated_350"
LOG_DIR      = ROOT / "logs"
MARKER       = OUT_DIR / ".ai_generation_in_progress"
PROGRESS_FILE = OUT_DIR / "progress.json"

# ── safe stdout (WindowStyle Hidden sets sys.stdout to None on Windows) ─────
def _safe_print(msg: str) -> None:
    try:
        if sys.stdout:
            print(msg, flush=True)
    except Exception:
        pass

# ── state (written to file; HTTP server reads file — zero lock contention) ──
_state = {
    "total": 350,
    "done": 0,
    "skipped": 0,
    "failed": 0,
    "current_patient": "",
    "started_at": 0.0,
    "rate_s_per_pt": 0.0,
    "eta_epoch": 0.0,
    "log": [],
    "status": "starting",
}
_TIMES: list[float] = []


def _write_state() -> None:
    try:
        PROGRESS_FILE.write_text(json.dumps(_state))
    except Exception:
        pass


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _safe_print(line)
    _state["log"].append(line)
    if len(_state["log"]) > 200:
        _state["log"] = _state["log"][-200:]
    _write_state()


def _update_rate() -> None:
    now = time.time()
    _TIMES.append(now)
    window = [t for t in _TIMES if now - t < 600]
    _TIMES[:] = window
    if len(window) >= 2:
        rate = (window[-1] - window[0]) / (len(window) - 1)
    else:
        elapsed = max(now - _state["started_at"], 1)
        rate = elapsed / max(_state["done"], 1)
    _state["rate_s_per_pt"] = rate
    remaining = _state["total"] - _state["done"]
    _state["eta_epoch"] = now + rate * remaining
    _write_state()


# ── HTML ──────────────────────────────────────────────────────────────────────
_HTML = b"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>AI Generation - Pelvic CT</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d1117;color:#e6edf3;font-family:monospace;padding:24px}
h1{font-size:1.4rem;margin-bottom:18px;color:#58a6ff}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px;margin-bottom:16px}
.row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid #21262d}
.row:last-child{border-bottom:none}
.lbl{color:#8b949e;font-size:.88rem}
.val{color:#e6edf3;font-weight:bold;font-size:.92rem}
.bar-wrap{background:#21262d;border-radius:6px;height:28px;width:100%;margin-top:14px;overflow:hidden;position:relative}
.bar{background:linear-gradient(90deg,#1a7f37,#2ea043);height:100%;transition:width 1s ease;min-width:2px;border-radius:6px}
.bar-text{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
          color:#fff;font-size:.85rem;font-weight:bold;white-space:nowrap;text-shadow:0 0 4px #000}
.status-running{color:#3fb950}.status-done{color:#58a6ff}.status-starting{color:#e3b341}.status-error{color:#f85149}
pre{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;
    font-size:.75rem;max-height:300px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;margin-top:8px}
.refresh{color:#484f58;font-size:.72rem;text-align:right;margin-top:6px}
</style></head><body>
<h1>&#129504; Synthetic Pelvic CT &mdash; AI Generation</h1>
<div class="card" id="main"><div style="color:#8b949e">Connecting to server&hellip;</div></div>
<div class="card"><div style="color:#8b949e;font-size:.85rem;margin-bottom:4px">Recent log</div><pre id="log"></pre></div>
<div class="refresh" id="ts"></div>
<script>
function fmt(s){
  if(!s||!isFinite(s)||s<0)return'--';
  const h=Math.floor(s/3600),m=Math.floor(s%3600/60),sec=Math.floor(s%60);
  return h?h+'h '+m+'m ':m?m+'m '+sec+'s ':sec+'s';
}
function eta(ep){
  if(!ep||ep<Date.now()/1000)return'--';
  return new Date(ep*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});
}
async function tick(){
  try{
    const r=await fetch('/api/state');
    if(!r.ok)throw new Error(r.status);
    const s=await r.json();
    const pct=s.total>0?Math.min(100,s.done/s.total*100):0;
    const elapsed=Date.now()/1000-s.started_at;
    document.getElementById('main').innerHTML=`
      <div class="row"><span class="lbl">Status</span>
        <span class="val status-${s.status}">${s.status.toUpperCase()}</span></div>
      <div class="row"><span class="lbl">Progress</span>
        <span class="val">${s.done} / ${s.total} patients &nbsp;(${pct.toFixed(1)}%)</span></div>
      <div class="row"><span class="lbl">Skipped (already done)</span><span class="val">${s.skipped}</span></div>
      <div class="row"><span class="lbl">Failed</span><span class="val">${s.failed}</span></div>
      <div class="row"><span class="lbl">Current patient</span>
        <span class="val">${s.current_patient||'&mdash;'}</span></div>
      <div class="row"><span class="lbl">Speed</span>
        <span class="val">${s.rate_s_per_pt?s.rate_s_per_pt.toFixed(0)+'s / patient':'&mdash;'}</span></div>
      <div class="row"><span class="lbl">Elapsed</span><span class="val">${fmt(elapsed)}</span></div>
      <div class="row"><span class="lbl">ETA (finish time)</span><span class="val">${eta(s.eta_epoch)}</span></div>
      <div class="bar-wrap">
        <div class="bar" style="width:${pct}%"></div>
        <div class="bar-text">${s.done} / ${s.total} &nbsp;&mdash;&nbsp; ${pct.toFixed(1)}%</div>
      </div>`;
    document.getElementById('log').textContent=(s.log||[]).slice(-80).join('\\n');
    document.getElementById('ts').textContent='Last update: '+new Date().toLocaleTimeString();
  }catch(e){
    document.getElementById('main').innerHTML=
      '<div style="color:#f85149">Cannot reach server (' + e + ') &mdash; retrying&hellip;</div>';
  }
}
tick();
setInterval(tick, 2000);
</script></body></html>"""


# ── HTTP server (reads progress file — no lock, no blocking) ─────────────────
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _HTML)
        elif self.path == "/api/state":
            try:
                body = PROGRESS_FILE.read_bytes()
            except Exception:
                body = b'{"status":"starting","done":0,"total":350,"log":[]}'
            self._send(200, "application/json", body)
        else:
            self.send_response(404); self.end_headers()

    def _send(self, code: int, ct: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _start_server(port: int) -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True, name="http-dashboard")
    t.start()
    _log(f"Dashboard -> http://127.0.0.1:{port}/")


# ── generation ────────────────────────────────────────────────────────────────
def _count_complete() -> int:
    if not OUT_DIR.exists():
        return 0
    return sum(
        1 for p in OUT_DIR.iterdir()
        if p.is_dir() and p.name.startswith("PF_")
        and (p / "metadata.json").exists()
        and (p / "DICOM").is_dir()
        and sum(1 for _ in (p / "DICOM").glob("*.dcm")) > 0
    )


def run(num_patients: int, port: int, no_browser: bool) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    MARKER.write_text(json.dumps({
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": num_patients,
        "pid": os.getpid(),
    }))

    already_done = _count_complete()
    _state.update({
        "started_at": time.time(),
        "total": num_patients,
        "done": already_done,
        "skipped": already_done,
        "status": "running",
    })
    _write_state()

    _start_server(port)

    if not no_browser:
        threading.Timer(1.5, webbrowser.open, args=[f"http://127.0.0.1:{port}/"]).start()

    _log(f"AI generation starting — target {num_patients} patients")
    _log(f"Output: {OUT_DIR}")
    if already_done:
        _log(f"Resuming: {already_done} patients already complete, will skip them")

    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg["generation"]["num_patients"]      = num_patients
    cfg["generation"]["slices_per_volume"] = int(cfg["generation"].get("slices_per_volume", 96))
    cfg["generation"]["sampler_steps"]     = int(cfg["generation"].get("sampler_steps", 50))
    cfg["paths"]["outputs_dir"]            = str(OUT_DIR)
    cfg["generation"]["resume"]            = True

    from src.generate import load_generators, generate_volume, _patient_is_complete
    from src.dicom_builder import write_dicom_series
    from src.exports import write_png_jpg_metadata
    import numpy as np

    _log("Loading AI models...")
    cvae, unet, sched, device = load_generators(cfg)
    _log(f"Models loaded on {device}")

    gcfg = cfg["generation"]
    expected_slices = int(gcfg["slices_per_volume"])
    n = num_patients
    n_plain = int(round(n * float(gcfg["plain_fraction"])))
    plan = [0] * n_plain + [1] * (n - n_plain)
    rng = np.random.default_rng(int(gcfg["seed"]))
    rng.shuffle(plan)

    for i, region_id in enumerate(plan, start=1):
        pid = f"PF_{i:04d}"
        pdir = OUT_DIR / pid

        if _patient_is_complete(pdir, expected_slices):
            continue

        _state["current_patient"] = pid
        _write_state()

        region_name = "plain" if region_id == 0 else "hilly"
        _log(f"[{i}/{n}] Generating {pid} ({region_name})")

        try:
            seed = int(gcfg["seed"]) + i
            vol = generate_volume(cfg, region_id, pid, seed, cvae, unet, sched, device)

            write_dicom_series(vol, pdir / "DICOM", cfg)
            write_png_jpg_metadata(vol, pdir / "PNG", pdir / "JPG", pdir / "metadata.json", cfg)

            _state["done"] += 1
            _update_rate()
            _log(f"  done  ({_state['done']}/{n}  {_state['done']/n*100:.1f}%)")

        except Exception as exc:
            _state["failed"] += 1
            _log(f"  FAILED {pid}: {exc}")
            _write_state()

    done_final = _count_complete()
    summary = {
        "total": num_patients, "done": done_final,
        "failed": _state["failed"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT_DIR / "generation_summary.json").write_text(json.dumps(summary, indent=2))
    MARKER.unlink(missing_ok=True)

    _state.update({"status": "done", "current_patient": "", "done": done_final})
    _log(f"Generation complete: {done_final}/{num_patients} patients.")
    _log("Dashboard stays up for 10 minutes.")
    _write_state()
    time.sleep(600)


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-patients", type=int, default=350)
    ap.add_argument("--port",         type=int, default=8769)
    ap.add_argument("--no-browser",   action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    run(args.num_patients, args.port, args.no_browser)
