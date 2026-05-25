"""Prevent the OS from sleeping during long pipeline runs.

Context manager — wrap a long operation in `with KeepAwake():` and the
system will not enter standby until the block exits.

Windows: uses SetThreadExecutionState with ES_CONTINUOUS | ES_SYSTEM_REQUIRED
         (ES_DISPLAY_REQUIRED would also keep the monitor on; we omit it so
         the screen can still blank to save the panel).
macOS:   spawns `caffeinate -is` and kills it on exit.
Linux:   no-op (most desktops let the user decide; relying on caller).
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional


_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001
_ES_AWAYMODE_REQUIRED = 0x00000040


class KeepAwake:
    def __init__(self, allow_display_sleep: bool = True):
        self.system = platform.system()
        self.allow_display_sleep = allow_display_sleep
        self._caffeinate: Optional[subprocess.Popen] = None

    def __enter__(self):
        if self.system == "Windows":
            try:
                import ctypes
                flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
                ctypes.windll.kernel32.SetThreadExecutionState(flags)
            except Exception:
                pass
        elif self.system == "Darwin":
            try:
                args = ["caffeinate", "-is"] if self.allow_display_sleep else ["caffeinate", "-d"]
                self._caffeinate = subprocess.Popen(args)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.system == "Windows":
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
            except Exception:
                pass
        elif self.system == "Darwin" and self._caffeinate is not None:
            try:
                self._caffeinate.terminate()
            except Exception:
                pass
        return False
