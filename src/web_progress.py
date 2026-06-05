"""Web-based progress dashboard for the synthetic pelvic-CT pipeline.

Replaces terminal tqdm bars with a self-contained HTTP dashboard so progress
renders consistently on any platform / shell (the project is run from Windows
PowerShell where tqdm's in-place updates are flaky).

Design:
  * One in-process `ProgressState` singleton holds the live state of all 6
    pipeline stages (build_cache, pseudo_labels, train_cvae, train_diffusion,
    generate, validate) plus a rolling log.
  * A tiny `http.server.ThreadingHTTPServer` runs in a daemon thread serving:
      GET /            -> the embedded HTML/CSS/JS dashboard
      GET /api/state   -> the current ProgressState as JSON
  * Pipeline code calls `set_stage`, `update_stage`, `finish_stage`, `log_msg`
    to publish events. The HTML page polls /api/state every 250 ms via fetch.

No external dependencies. Works offline. Cross-origin not allowed (loopback only).
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass, field, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


# =========================================================================
# State
# =========================================================================

STAGE_ORDER = [
    ("build_cache",       "Build cache"),
    ("pseudo_labels",     "Pseudo-labels"),
    ("train_cvae",        "Train CVAE"),
    ("train_diffusion",   "Train diffusion"),
    ("generate",          "Generate patients"),
    ("validate",          "Validate DICOM"),
    ("anatomy_validate",  "Anatomy validation"),
]


@dataclass
class Stage:
    id: str
    name: str
    status: str = "pending"          # pending | running | done | error
    current: int = 0
    total: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    postfix: str = ""
    error: str = ""

    def elapsed_s(self) -> float:
        if self.started_at == 0:
            return 0.0
        end = self.finished_at if self.finished_at else time.time()
        return end - self.started_at

    def eta_s(self) -> Optional[float]:
        if self.status != "running" or self.current <= 0 or self.total <= 0:
            return None
        rate = self.current / max(self.elapsed_s(), 1e-6)
        remaining = self.total - self.current
        return remaining / max(rate, 1e-6)

    def rate_str(self) -> str:
        if self.status != "running" or self.current <= 0:
            return ""
        rate = self.current / max(self.elapsed_s(), 1e-6)
        return f"{rate:.2f}/s" if rate >= 1 else f"{1/rate:.2f}s/it"


class ProgressState:
    def __init__(self):
        self.lock = threading.RLock()
        self.stages = [Stage(id=sid, name=sname) for sid, sname in STAGE_ORDER]
        self._index = {s.id: s for s in self.stages}
        self.active_id: Optional[str] = None
        self.started_at: float = time.time()
        self.log: list[str] = []
        self.max_log = 250
        self.run_label: str = ""
        # Optional whole-pipeline ETA: seconds the run is expected to take.
        # Set via start_server(expected_total_s=...). The dashboard renders
        # elapsed / expected / remaining in the header.
        self.expected_total_s: Optional[float] = None
        # Config snapshot (paths.checkpoints_dir + training epoch counts) used
        # by _compute_dynamic_remaining_s to give an ETA based on REAL on-disk
        # checkpoint mtimes rather than the preset's static guess. Set via
        # start_server(cfg=...).
        self.cfg: Optional[dict] = None

    def _compute_dynamic_remaining_s(self) -> Optional[float]:
        """ETA based on actual per-epoch time measured from checkpoint mtimes.

        Survives reboots because it reads mtimes of cvae_epochN.pt /
        diffusion_epochN.pt -- not the in-process elapsed counter.
        Returns None if there isn't enough data yet (no completed epoch).
        """
        if not self.cfg:
            return None
        try:
            from pathlib import Path
            ckpt_dir = Path(self.cfg["paths"]["checkpoints_dir"])
            if not ckpt_dir.is_dir():
                return None
            cvae_total = int(self.cfg["training"]["cvae"]["epochs"])
            diff_total = int(self.cfg["training"]["diffusion"]["epochs"])

            cvae_files = sorted(ckpt_dir.glob("cvae_epoch*.pt"),
                                key=lambda p: int(p.stem.replace("cvae_epoch", "")))
            diff_files = sorted(ckpt_dir.glob("diffusion_epoch*.pt"),
                                key=lambda p: int(p.stem.replace("diffusion_epoch", "")))
            cvae_done = len(cvae_files)
            diff_done = len(diff_files)

            def _mean_gap(files):
                if len(files) < 2:
                    return None
                mtimes = sorted(f.stat().st_mtime for f in files)
                gaps = [mtimes[i+1] - mtimes[i] for i in range(len(mtimes)-1)]
                return sum(gaps) / len(gaps)

            cvae_per_epoch_s = _mean_gap(cvae_files)
            diff_per_epoch_s = _mean_gap(diff_files)
            # If only one CVAE epoch is done, use (mtime - started_at) as a proxy.
            if cvae_per_epoch_s is None and cvae_done == 1:
                est = cvae_files[0].stat().st_mtime - self.started_at
                if est > 0:
                    cvae_per_epoch_s = est
            # Fall back: assume diffusion takes as long as CVAE (rough).
            if diff_per_epoch_s is None:
                diff_per_epoch_s = cvae_per_epoch_s

            if cvae_per_epoch_s is None:
                return None

            # Active in-progress fraction
            active = self.active_id
            cvae_remaining_epochs = max(0, cvae_total - cvae_done)
            diff_remaining_epochs = max(0, diff_total - diff_done)
            total_s = 0.0

            if active == "train_cvae":
                stage = self._index["train_cvae"]
                in_prog = (stage.current / stage.total) if stage.total > 0 else 0.0
                total_s += cvae_per_epoch_s * max(0.0, 1.0 - in_prog)
                total_s += cvae_per_epoch_s * max(0, cvae_remaining_epochs - 1)
                total_s += diff_per_epoch_s * diff_remaining_epochs if diff_per_epoch_s else 0.0
            elif active == "train_diffusion":
                stage = self._index["train_diffusion"]
                in_prog = (stage.current / stage.total) if stage.total > 0 else 0.0
                total_s += diff_per_epoch_s * max(0.0, 1.0 - in_prog) if diff_per_epoch_s else 0.0
                total_s += diff_per_epoch_s * max(0, diff_remaining_epochs - 1) if diff_per_epoch_s else 0.0
            else:
                # Outside training (generate/validate/anatomy_validate): tiny
                # compared to training, fall back to the static estimate.
                return None

            return total_s
        except Exception:
            return None

    def snapshot(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.started_at
            static_remaining_s = None
            if self.expected_total_s is not None:
                static_remaining_s = max(0.0, self.expected_total_s - elapsed)
            dynamic_remaining_s = self._compute_dynamic_remaining_s()
            # Prefer the dynamic ETA when it's available -- it knows about
            # epochs trained before this Python process started (which the
            # static elapsed-from-process-start can't see).
            remaining_s = dynamic_remaining_s if dynamic_remaining_s is not None else static_remaining_s
            return {
                "started_at": self.started_at,
                "elapsed_s": elapsed,
                "expected_total_s": self.expected_total_s,
                "static_remaining_s": static_remaining_s,
                "dynamic_remaining_s": dynamic_remaining_s,
                "remaining_s": remaining_s,
                "run_label": self.run_label,
                "active_id": self.active_id,
                "stages": [
                    {
                        **asdict(s),
                        "elapsed_s": round(s.elapsed_s(), 1),
                        "eta_s": (round(s.eta_s(), 1) if s.eta_s() is not None else None),
                        "rate_str": s.rate_str(),
                        "pct": (s.current / s.total * 100.0) if s.total > 0 else None,
                    } for s in self.stages
                ],
                "log": list(self.log),
            }


_state = ProgressState()


# =========================================================================
# Public API (called from pipeline code)
# =========================================================================

def set_run_label(label: str) -> None:
    with _state.lock:
        _state.run_label = label


def set_stages(stage_defs: list[tuple[str, str]]) -> None:
    """Replace the stage list. Call BEFORE start_server. Used by side-pipelines
    (e.g. the PFD POC) that want a parallel dashboard with their own stages
    instead of the main training stages."""
    with _state.lock:
        _state.stages = [Stage(id=sid, name=sname) for sid, sname in stage_defs]
        _state._index = {s.id: s for s in _state.stages}
        _state.active_id = None


def set_stage(stage_id: str, total: int = 0, postfix: str = "") -> None:
    """Mark a stage as running and reset its counters."""
    with _state.lock:
        s = _state._index.get(stage_id)
        if s is None:
            return
        s.status = "running"
        s.current = 0
        s.total = int(total)
        s.postfix = postfix
        s.error = ""
        s.started_at = time.time()
        s.finished_at = 0.0
        _state.active_id = stage_id
    log_msg(f"start: {stage_id}")


def update_stage(current: Optional[int] = None, delta: int = 0,
                 total: Optional[int] = None, postfix: Optional[str] = None) -> None:
    """Update the currently-active stage."""
    with _state.lock:
        if _state.active_id is None:
            return
        s = _state._index[_state.active_id]
        if total is not None:
            s.total = int(total)
        if current is not None:
            s.current = int(current)
        if delta:
            s.current += int(delta)
        if postfix is not None:
            s.postfix = str(postfix)


def finish_stage(status: str = "done", error: str = "") -> None:
    with _state.lock:
        if _state.active_id is None:
            return
        s = _state._index[_state.active_id]
        s.status = status
        s.error = error
        s.finished_at = time.time()
        if s.total > 0 and status == "done":
            s.current = s.total
    log_msg(f"end:   {_state.active_id} ({status})")


def log_msg(msg: str) -> None:
    """Append a short message to the rolling log."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with _state.lock:
        _state.log.append(line)
        if len(_state.log) > _state.max_log:
            del _state.log[: -_state.max_log]


# =========================================================================
# HTTP server
# =========================================================================

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pelvic CT Generator -- Pipeline</title>
<style>
  :root {
    --bg: #0d1117;
    --card: #161b22;
    --card2: #1c2229;
    --border: #2a3038;
    --text: #e6edf3;
    --muted: #8b95a0;
    --accent: #58a6ff;
    --ok: #56d364;
    --warn: #d29922;
    --err: #f85149;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
      Roboto, Oxygen, Ubuntu, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px;
    min-height: 100vh;
  }
  header {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 18px; padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  h1 { font-size: 18px; margin: 0; font-weight: 600; letter-spacing: 0.3px; }
  .meta { color: var(--muted); font-size: 13px; font-variant-numeric: tabular-nums; }
  .meta b { color: var(--text); font-weight: 600; }
  .stages { display: flex; flex-direction: column; gap: 10px; }
  .stage {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 14px 16px; transition: background 0.2s, border-color 0.2s;
  }
  .stage.running { border-color: var(--accent); background: var(--card2); }
  .stage.done    { border-color: var(--ok); opacity: 0.85; }
  .stage.error   { border-color: var(--err); }
  .stage.pending { opacity: 0.55; }
  .row { display: flex; align-items: center; gap: 12px; }
  .icon {
    width: 18px; height: 18px; flex-shrink: 0;
    display: inline-flex; align-items: center; justify-content: center;
  }
  .icon.spin svg { animation: spin 0.9s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .name { font-weight: 600; font-size: 14px; flex: 1; }
  .right { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }
  .bar-wrap {
    margin-top: 10px; height: 8px; background: #0b0f14;
    border-radius: 4px; overflow: hidden; position: relative;
  }
  .bar {
    height: 100%; background: linear-gradient(90deg, var(--accent), #79c0ff);
    border-radius: 4px; transition: width 0.25s ease-out;
    width: 0%;
  }
  .stage.done .bar { background: linear-gradient(90deg, var(--ok), #7ee787); width: 100%; }
  .stage.error .bar { background: var(--err); }
  .bar.indet {
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    background-size: 200% 100%; animation: indet 1.2s linear infinite; width: 100%;
  }
  @keyframes indet {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }
  .postfix {
    margin-top: 6px; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12px; color: var(--muted); min-height: 14px;
    overflow-wrap: anywhere;
  }
  .stage.error .postfix { color: var(--err); }
  .log-panel {
    margin-top: 22px; background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 16px;
  }
  .log-panel h2 {
    font-size: 12px; margin: 0 0 10px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;
  }
  .log {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 11.5px; color: var(--muted); max-height: 220px; overflow-y: auto;
    white-space: pre-wrap; line-height: 1.5;
  }
  .log .ts { color: #4f5965; }
  footer {
    margin-top: 18px; text-align: center; color: #4f5965; font-size: 11px;
  }
</style>
</head>
<body>
  <header>
    <h1 id="title">Pelvic CT Generator</h1>
    <div class="meta">
      <span id="run-label"></span>
      <span style="margin: 0 10px;">&middot;</span>
      <span>elapsed <b id="elapsed">0s</b></span>
      <span id="eta-block" style="display: none;">
        <span style="margin: 0 10px;">&middot;</span>
        <span>est <b id="expected">--</b></span>
        <span style="margin: 0 10px;">&middot;</span>
        <span>remaining <b id="remaining" style="color: var(--accent);">--</b></span>
        <span style="margin: 0 10px;">&middot;</span>
        <span>finishes <b id="eta-time">--</b></span>
      </span>
      <span style="margin: 0 10px;">&middot;</span>
      <span id="conn" style="color: var(--ok);">connected</span>
    </div>
  </header>

  <div id="stages" class="stages"></div>

  <div class="log-panel">
    <h2>Log</h2>
    <div id="log" class="log"></div>
  </div>

  <footer>auto-refresh 250 ms &middot; close this tab when pipeline finishes</footer>

<script>
const SPINNER_SVG = '<svg viewBox="0 0 24 24" width="16" height="16"><circle cx="12" cy="12" r="9" stroke="#58a6ff" stroke-width="2.4" fill="none" stroke-dasharray="14 38" stroke-linecap="round"/></svg>';
const CHECK_SVG   = '<svg viewBox="0 0 24 24" width="16" height="16"><path d="M5 12.5l4 4 10-10" stroke="#56d364" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const DOT_SVG     = '<svg viewBox="0 0 24 24" width="16" height="16"><circle cx="12" cy="12" r="3" fill="#4f5965"/></svg>';
const ERR_SVG     = '<svg viewBox="0 0 24 24" width="16" height="16"><path d="M6 6l12 12M18 6L6 18" stroke="#f85149" stroke-width="2.4" stroke-linecap="round"/></svg>';

function iconFor(status) {
  if (status === "running") return '<div class="icon spin">' + SPINNER_SVG + '</div>';
  if (status === "done")    return '<div class="icon">' + CHECK_SVG + '</div>';
  if (status === "error")   return '<div class="icon">' + ERR_SVG + '</div>';
  return '<div class="icon">' + DOT_SVG + '</div>';
}

function fmtDuration(s) {
  if (!s || s < 1) return "0s";
  s = Math.round(s);
  if (s < 60) return s + "s";
  const m = Math.floor(s / 60), sec = s % 60;
  if (m < 60) return m + "m " + (sec < 10 ? "0" : "") + sec + "s";
  const h = Math.floor(m / 60), mr = m % 60;
  return h + "h " + (mr < 10 ? "0" : "") + mr + "m";
}

function fmtClockTime(epoch_s) {
  const d = new Date(epoch_s * 1000);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  const tomorrow = new Date(today); tomorrow.setDate(today.getDate() + 1);
  const isTomorrow = d.toDateString() === tomorrow.toDateString();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  if (sameDay) return hh + ":" + mm + " today";
  if (isTomorrow) return hh + ":" + mm + " tomorrow";
  const dow = ["Sun","Mon","Tue","Wed","Thu","Fri","Sat"][d.getDay()];
  return hh + ":" + mm + " " + dow + " " + d.getDate() + "/" + (d.getMonth() + 1);
}

function render(state) {
  document.getElementById("elapsed").textContent = fmtDuration(state.elapsed_s);
  document.getElementById("run-label").textContent = state.run_label || "";

  // Pipeline-level ETA. Prefer the dynamic estimate (reads checkpoint mtimes,
  // survives reboots) over the preset's static guess. Tag the source so the
  // user knows whether it's a live measurement or a preset prediction.
  const etaBlock = document.getElementById("eta-block");
  if (state.expected_total_s || state.dynamic_remaining_s !== null) {
    etaBlock.style.display = "inline";
    const remSec = state.remaining_s || 0;
    const sourceTag = (state.dynamic_remaining_s !== null) ? " (live)" : " (preset)";
    document.getElementById("expected").textContent = fmtDuration(state.expected_total_s || 0);
    document.getElementById("remaining").textContent = fmtDuration(remSec) + sourceTag;
    // Finish clock is computed from NOW + remaining, not started_at + expected.
    // That way dynamic ETA's finish time is correct.
    const finishEpoch = (Date.now() / 1000) + remSec;
    document.getElementById("eta-time").textContent = fmtClockTime(finishEpoch);
  } else {
    etaBlock.style.display = "none";
  }

  const container = document.getElementById("stages");
  const existing = new Map(
    [...container.children].map(el => [el.dataset.id, el])
  );

  state.stages.forEach(s => {
    let el = existing.get(s.id);
    if (!el) {
      el = document.createElement("div");
      el.className = "stage";
      el.dataset.id = s.id;
      el.innerHTML = `
        <div class="row">
          <div class="icon-slot"></div>
          <div class="name"></div>
          <div class="right"></div>
        </div>
        <div class="bar-wrap"><div class="bar"></div></div>
        <div class="postfix"></div>
      `;
      container.appendChild(el);
    }

    el.className = "stage " + s.status;
    el.querySelector(".icon-slot").innerHTML = iconFor(s.status);
    el.querySelector(".name").textContent = s.name;

    // Right-aligned: counts + elapsed + eta
    let right = "";
    if (s.status === "running" || s.status === "done") {
      if (s.total > 0) right += s.current + " / " + s.total;
      if (s.rate_str) right += "  &middot;  " + s.rate_str;
      if (s.elapsed_s) right += "  &middot;  elapsed " + fmtDuration(s.elapsed_s);
      if (s.eta_s !== null && s.status === "running")
        right += "  &middot;  eta " + fmtDuration(s.eta_s);
    }
    el.querySelector(".right").innerHTML = right;

    const bar = el.querySelector(".bar");
    if (s.status === "running" && s.total === 0) {
      bar.classList.add("indet");
      bar.style.width = "100%";
    } else {
      bar.classList.remove("indet");
      const pct = s.pct === null || s.pct === undefined ?
        (s.status === "done" ? 100 : 0) : s.pct;
      bar.style.width = Math.min(100, Math.max(0, pct)) + "%";
    }

    el.querySelector(".postfix").textContent =
      s.error ? ("error: " + s.error) : (s.postfix || "");
  });

  const logEl = document.getElementById("log");
  // Render with timestamps grayed out
  logEl.innerHTML = state.log.map(line => {
    const m = line.match(/^\[(.*?)\]\s*(.*)$/);
    if (m) return '<span class="ts">[' + m[1] + ']</span> ' +
      m[2].replace(/&/g,"&amp;").replace(/</g,"&lt;");
    return line.replace(/&/g,"&amp;").replace(/</g,"&lt;");
  }).join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}

let failCount = 0;
async function poll() {
  try {
    const r = await fetch("/api/state", {cache: "no-store"});
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    failCount = 0;
    document.getElementById("conn").textContent = "connected";
    document.getElementById("conn").style.color = "var(--ok)";
    render(data);
  } catch (e) {
    failCount++;
    if (failCount > 8) {
      document.getElementById("conn").textContent = "disconnected";
      document.getElementById("conn").style.color = "var(--err)";
    }
  }
}

setInterval(poll, 250);
poll();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return  # silence default request logging

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/state":
            body = json.dumps(_state.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


_server: Optional[ThreadingHTTPServer] = None


def _find_free_port(preferred: int) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port found near " + str(preferred))


def start_server(port: int = 8765, open_browser: bool = True,
                 run_label: str = "",
                 expected_total_s: Optional[float] = None,
                 cfg: Optional[dict] = None) -> str:
    """Start the dashboard HTTP server on a background thread. Returns the URL."""
    global _server
    if _server is not None:
        return f"http://127.0.0.1:{_server.server_address[1]}/"

    port = _find_free_port(port)
    _server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=_server.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}/"
    _state.started_at = time.time()
    _state.run_label = run_label
    _state.expected_total_s = expected_total_s
    _state.cfg = cfg
    log_msg(f"dashboard up at {url}")
    if expected_total_s:
        hours = expected_total_s / 3600.0
        log_msg(f"estimated total runtime: {hours:.1f}h")
    if open_browser and not os.environ.get("NO_BROWSER"):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return url


def stop_server(grace_s: float = 5.0) -> None:
    """Stop the server after a short grace period (so the user sees the final state)."""
    global _server
    if _server is None:
        return
    log_msg(f"pipeline complete -- dashboard closing in {int(grace_s)}s")
    time.sleep(grace_s)
    try:
        _server.shutdown()
        _server.server_close()
    except Exception:
        pass
    _server = None
