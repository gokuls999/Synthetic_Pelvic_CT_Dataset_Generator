"""Progress bars + spinner utilities (single source of truth for the project).

Two primitives:
  * `pbar(iterable, total, desc, ...)` -- a thin wrapper around tqdm with a
    consistent format used everywhere we have a known-length loop.
  * `Spinner(desc)` -- context manager that animates a loading icon on a
    background thread for one-off operations whose duration is unknown
    (loading checkpoints, indexing a cache, fitting KMeans, etc.).

Both are ASCII-safe so Windows cp1252 consoles don't blow up.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Iterable, Iterator, Optional

from tqdm import tqdm


SPINNER_FRAMES_ASCII = ["|", "/", "-", "\\"]


class Spinner:
    """Animated `loading...` indicator for indeterminate-duration ops.

    Usage:
        with Spinner("Loading CVAE checkpoint"):
            cvae.load_state_dict(torch.load(...))
    """

    def __init__(self, desc: str = "Loading", frames=None,
                 interval: float = 0.12, stream=None):
        self.desc = desc
        self.frames = frames or SPINNER_FRAMES_ASCII
        self.interval = interval
        self.stream = stream or sys.stdout
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def __enter__(self) -> "Spinner":
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        elapsed = time.time() - self._t0
        status = "done " if exc_type is None else "FAIL "
        # Clear current spinner line (overwrite with spaces, then carriage return).
        try:
            line = f"\r[{status}] {self.desc} ({elapsed:.1f}s)"
            pad = max(0, 80 - len(line))
            self.stream.write(line + " " * pad + "\n")
            self.stream.flush()
        except Exception:
            pass

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            frame = self.frames[i % len(self.frames)]
            try:
                self.stream.write(f"\r {frame}  {self.desc}...")
                self.stream.flush()
            except Exception:
                return
            self._stop.wait(self.interval)
            i += 1


def pbar(iterable: Optional[Iterable] = None, total: Optional[int] = None,
         desc: Optional[str] = None, unit: str = "it", leave: bool = True,
         **kwargs) -> tqdm:
    """Standardized tqdm wrapper. Returns the tqdm object."""
    return tqdm(
        iterable, total=total, desc=desc, unit=unit, leave=leave,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        dynamic_ncols=True, mininterval=0.2, **kwargs,
    )


def iter_with_progress(iterable: Iterable, total: Optional[int] = None,
                       desc: str = "Processing", unit: str = "it") -> Iterator:
    """Convenience: yield items with a tqdm bar."""
    bar = pbar(iterable, total=total, desc=desc, unit=unit)
    try:
        for item in bar:
            yield item
    finally:
        bar.close()
