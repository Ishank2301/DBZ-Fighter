"""
FPS counter and frame-timing utilities.
"""

import time
import collections


class FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30):
        self._times: collections.deque = collections.deque(maxlen=window)
        self._last: float = time.perf_counter()

    def tick(self) -> float:
        """Call once per frame. Returns current smoothed FPS."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / sum(self._times)

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / sum(self._times)

    @property
    def ms(self) -> float:
        """Average milliseconds per frame."""
        if not self._times:
            return 0.0
        return (sum(self._times) / len(self._times)) * 1000


class FrameTimer:
    """Simple elapsed-time tracker for frame pacing."""

    def __init__(self, target_fps: float = 30.0):
        self.target_dt = 1.0 / target_fps
        self._last = time.perf_counter()

    def elapsed(self) -> float:
        """Seconds since last call to elapsed()."""
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        return dt

    def sleep_remaining(self):
        """Sleep any leftover time to hit target FPS."""
        now = time.perf_counter()
        wait = self.target_dt - (now - self._last)
        if wait > 0:
            time.sleep(wait)
