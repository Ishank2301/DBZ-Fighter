"""Performance is made to record the Tracks FPS, per-stage latency, and memory usage.
Designed to be lightweight — called every frame."""

import time
import collections
import psutil
import os
from src.core.config import PREF_LOG_INTERVAL


class PerformanceMonitor:
    def __init__(self):
        self.frame_times = collections.deque(maxlen=60)
        self.stage_times = {}
        self.frame_starts = {}
        self.frame_count = 0
        self._process = psutil.Process(os.getpid())

    # Track fps:
    def tick(self):
        now = time.perf_counter()
        self._frame_times.append(now)
        self._frame_count += 1

        if self._frame_count % PREF_LOG_INTERVAL == 0:
            self._log()

    @property
    def fps(self):
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / elapsed if elapsed > 0 else 0.0

    # Staging Latency:
    def start(self, stage: str):
        self._stage_starts[stage] = time.perf_counter()

    def stop(self, stage: str):
        if stage not in self._stage_starts:
            return
        elapsed_ms = (time.perf_counter() - self._stage_starts[stage]) * 1000
        if stage not in self._stage_times:
            self._stage_times[stage] = collections.deque(maxlen=60)
        self._stage_times[stage].append(elapsed_ms)

    def latency(self, stage: str):
        buf = self._stage_times.get(stage)
        return sum(buf) / len(buf) if buf else 0.0

    # Memory:
    @property
    def memory_mb(self):
        return self._process.memory_info().rss / (1024**2)

    # Summary:
    def summary(self):
        return {
            "fps": round(self.fps, 1),
            "memory_mb": round(self.memory_mb, 1),
            **{f"{k}_ms": round(self.latency(k), 2) for k in self._stage_times},
        }

    def _log(self):
        s = self.summary()
        parts = [f"FPS={s['fps']}", f"RAM={s['memory_mb']}MB"]
        parts += [f"{k}={v}" for k, v in s.items() if k.endswith("_ms")]
        print(f"[Perf] {' | '.join(parts)}")
