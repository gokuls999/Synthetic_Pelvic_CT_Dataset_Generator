"""Progress primitives that publish to the web dashboard (src.web_progress).

Kept the same `pbar(...)` / `Spinner(...)` API so existing call sites don't change:
  * `pbar(iterable, total, desc, ...)`  -- behaves like a tqdm iterator; every
    `__next__` ticks the active stage's `current`. `set_postfix(**kw)` /
    `set_postfix_str(s)` push a one-line summary under the bar.
  * `Spinner(desc)` context manager -- shows an indeterminate (animated) bar
    on the active stage while inside the `with` block.

Outer/inner bar layering: each stage is a single web card. The outermost
pbar() that wraps an iterable controls `current/total`; nested pbars()
(e.g. inner batch loop while outer epoch loop is running) take over the
display for their duration, then the outer call resumes by re-asserting
its own total when it next ticks.

This module never imports tqdm and never writes to stdout, so there is no
terminal-rendering issue to work around.
"""

from __future__ import annotations

import threading
from typing import Iterable, Iterator, Optional

from . import web_progress as wp


class _Bar:
    """Iterator wrapper that publishes progress to the web dashboard."""

    def __init__(self, iterable: Optional[Iterable], total: Optional[int],
                 desc: Optional[str], leave: bool):
        self.iterable = iterable
        self.total = total or (len(iterable) if hasattr(iterable, "__len__") else 0)
        self.desc = desc or ""
        self.leave = leave
        self.n = 0
        self._postfix = ""
        # Reset the active stage's counters for this bar.
        wp.update_stage(current=0, total=self.total)
        if self.desc:
            wp.update_stage(postfix=self.desc)

    def __iter__(self) -> Iterator:
        if self.iterable is None:
            return iter([])
        it = iter(self.iterable)
        while True:
            try:
                item = next(it)
            except StopIteration:
                return
            yield item
            self.n += 1
            # Reassert own total each tick so nested bars can re-claim the card
            # cleanly when control returns to the outer loop.
            wp.update_stage(current=self.n, total=self.total,
                            postfix=self._postfix or self.desc)

    def update(self, n: int = 1) -> None:
        self.n += n
        wp.update_stage(current=self.n, total=self.total,
                        postfix=self._postfix or self.desc)

    def set_postfix(self, **kwargs) -> None:
        self._postfix = " ".join(f"{k}={v}" for k, v in kwargs.items())
        wp.update_stage(postfix=self._postfix, total=self.total)

    def set_postfix_str(self, s: str) -> None:
        self._postfix = str(s)
        wp.update_stage(postfix=self._postfix, total=self.total)

    def close(self) -> None:
        # The owning stage controls finish_stage; bar close is a no-op here.
        pass

    # Support `with pbar(...) as bar:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def pbar(iterable: Optional[Iterable] = None, total: Optional[int] = None,
         desc: Optional[str] = None, unit: str = "it", leave: bool = True,
         **_kwargs) -> _Bar:
    """Drop-in tqdm replacement that publishes to the web dashboard."""
    return _Bar(iterable, total, desc, leave)


def iter_with_progress(iterable: Iterable, total: Optional[int] = None,
                       desc: str = "", unit: str = "it") -> Iterator:
    """Convenience: yield items while ticking the active stage."""
    bar = pbar(iterable, total=total, desc=desc, unit=unit)
    yield from bar
    bar.close()


class Spinner:
    """Indeterminate progress (animated stripe) for unknown-duration ops.

    Usage:
        with Spinner("Loading CVAE checkpoint"):
            cvae.load_state_dict(torch.load(path))
    """

    def __init__(self, desc: str = "Loading", **_kwargs):
        self.desc = desc
        self._prev_total = 0
        self._prev_postfix = ""
        self._prev_current = 0

    def __enter__(self) -> "Spinner":
        # Snapshot whatever the active stage was showing, switch to indeterminate.
        snap = wp._state.snapshot()
        active = next((s for s in snap["stages"] if s["id"] == snap["active_id"]), None)
        if active is not None:
            self._prev_total = active["total"]
            self._prev_postfix = active["postfix"]
            self._prev_current = active["current"]
        wp.update_stage(current=0, total=0, postfix=self.desc)   # total=0 => indeterminate
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous bar state so the owning loop's progress reappears.
        wp.update_stage(
            current=self._prev_current,
            total=self._prev_total,
            postfix=self._prev_postfix,
        )
        return False
