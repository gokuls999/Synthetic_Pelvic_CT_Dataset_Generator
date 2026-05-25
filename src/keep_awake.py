"""Prevent the OS from sleeping during long pipeline runs.

Context manager -- wrap a long operation in `with KeepAwake():` and the
system will not enter standby until the block exits.

Windows: TWO layers of defense, because SetThreadExecutionState alone is
         insufficient -- it only blocks *idle* sleep, but Windows can still
         sleep if the power scheme's standby timeout fires (we hit this on
         a 1080 Ti rig with a 15-minute timeout: GPU compute apparently
         doesn't count as activity).
         Layer 1: powercfg standby/hibernate timeouts -> 0 (never).
                  Originals are saved in instance state and restored on exit.
         Layer 2: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
                  as a belt-and-braces backup.
macOS:   caffeinate -is subprocess; killed on exit.
Linux:   no-op (most desktops let the user decide).
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional


_ES_CONTINUOUS       = 0x80000000
_ES_SYSTEM_REQUIRED  = 0x00000001


def _powercfg_query_timeout(setting: str) -> Optional[int]:
    """Return the AC or DC standby timeout in seconds, or None if it fails.

    `setting` is "ac" or "dc". Parses `powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE`
    output for the matching Current Power Setting Index hex value.
    """
    try:
        out = subprocess.check_output(
            ["powercfg", "/query", "SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except Exception:
        return None
    needle = "Current AC Power Setting Index:" if setting == "ac" else "Current DC Power Setting Index:"
    for line in out.splitlines():
        if needle in line:
            hex_str = line.split(":")[-1].strip()
            try:
                return int(hex_str, 16)
            except ValueError:
                return None
    return None


def _powercfg_set(setting: str, seconds: int) -> bool:
    flag = "standby-timeout-ac" if setting == "ac" else "standby-timeout-dc"
    try:
        subprocess.run(["powercfg", "/change", flag, str(seconds)],
                       check=True, timeout=5, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


class KeepAwake:
    def __init__(self, allow_display_sleep: bool = True):
        self.system = platform.system()
        self.allow_display_sleep = allow_display_sleep
        self._caffeinate: Optional[subprocess.Popen] = None
        self._orig_ac: Optional[int] = None
        self._orig_dc: Optional[int] = None

    def __enter__(self):
        if self.system == "Windows":
            # Layer 1: change the power scheme so the OS literally cannot sleep.
            self._orig_ac = _powercfg_query_timeout("ac")
            self._orig_dc = _powercfg_query_timeout("dc")
            _powercfg_set("ac", 0)
            _powercfg_set("dc", 0)
            # Layer 2: tell Windows the thread is doing important work.
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(
                    _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
                )
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
            # Restore original sleep timeouts (if we had any).
            if self._orig_ac is not None:
                _powercfg_set("ac", self._orig_ac)
            if self._orig_dc is not None:
                _powercfg_set("dc", self._orig_dc)
        elif self.system == "Darwin" and self._caffeinate is not None:
            try:
                self._caffeinate.terminate()
            except Exception:
                pass
        return False
