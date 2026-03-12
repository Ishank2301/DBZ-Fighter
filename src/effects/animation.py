"""
SpriteSheet loader and frame-stepping animator.
Each spritesheet is a horizontal strip:  N frames * frame_width pixels wide.
"""

import os
import cv2
import numpy as np
from src.core.config import SPRITES_DIR


class SpriteSheet:
    """
    Loads a horizontal spritesheet PNG and exposes individual BGRA frames.
    """

    def __init__(self, path: str, n_frames: int, fps: float = 12.0):
        self.path = path
        self.n_frames = n_frames
        self.fps = fps
        self._frames: list[np.ndarray] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            # Return blank placeholder frames so the engine doesn't crash
            print(
                f"[SpriteSheet] WARNING: {self.path} not found — "
                "using blank placeholder."
            )
            self._frames = [
                np.zeros((64, 64, 4), dtype=np.uint8) for _ in range(self.n_frames)
            ]
            return

        img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[SpriteSheet] ERROR reading {self.path}")
            self._frames = [
                np.zeros((64, 64, 4), dtype=np.uint8) for _ in range(self.n_frames)
            ]
            return

        # Ensure 4 channels
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        h, total_w = img.shape[:2]
        fw = total_w // self.n_frames
        for i in range(self.n_frames):
            self._frames.append(img[:, i * fw : (i + 1) * fw].copy())

    def get_frame(self, index: int) -> np.ndarray:
        """Return BGRA frame at given index (wraps around)."""
        return self._frames[index % self.n_frames]

    @property
    def frame_count(self) -> int:
        return len(self._frames)


class Animator:
    """
    Drives a SpriteSheet forward in time.
    Supports one-shot and looping playback.
    """

    def __init__(self, sheet: SpriteSheet, loop: bool = True):
        self.sheet = sheet
        self.loop = loop
        self._time = 0.0
        self.finished = False

    def update(self, dt: float):
        """Advance animation by dt seconds."""
        if self.finished:
            return
        self._time += dt * self.sheet.fps
        if self._time >= self.sheet.n_frames:
            if self.loop:
                self._time %= self.sheet.n_frames
            else:
                self._time = self.sheet.n_frames - 1
                self.finished = True

    def current_frame(self) -> np.ndarray:
        return self.sheet.get_frame(int(self._time))

    def reset(self):
        self._time = 0.0
        self.finished = False

    @property
    def progress(self) -> float:
        """0.0 → 1.0 playback progress."""
        return min(self._time / max(self.sheet.n_frames, 1), 1.0)


# ── Pre-built sheet registry ──────────────────────────────────────────────────
# Maps a short key → (filename, n_frames, fps)
_SHEET_REGISTRY: dict[str, tuple[str, int, float]] = {
    "aura_gold": ("aura_sheet_gold.png", 8, 12.0),
    "aura_blue": ("aura_sheet_blue.png", 8, 12.0),
    "aura_red": ("aura_sheet_red.png", 8, 12.0),
    "aura_purple": ("aura_sheet_purple.png", 8, 12.0),
    "aura_white": ("aura_sheet_white.png", 8, 12.0),
    "beam_blue": ("beam_sheet_blue.png", 8, 18.0),
    "beam_gold": ("beam_sheet_gold.png", 8, 18.0),
    "beam_white": ("beam_sheet_white.png", 8, 18.0),
    "beam_red": ("beam_sheet_red.png", 8, 18.0),
    "explosion_gold": ("explosion_sheet_gold.png", 10, 20.0),
    "explosion_blue": ("explosion_sheet_blue.png", 10, 20.0),
    "explosion_white": ("explosion_sheet_white.png", 10, 20.0),
    "ring_gold": ("ring_sheet_gold.png", 10, 16.0),
    "ring_blue": ("ring_sheet_blue.png", 10, 16.0),
    "ring_purple": ("ring_sheet_purple.png", 10, 16.0),
    "spark_gold": ("spark_sheet_gold.png", 8, 24.0),
    "spark_blue": ("spark_sheet_blue.png", 8, 24.0),
    "spark_white": ("spark_sheet_white.png", 8, 24.0),
    "smoke": ("smoke_sheet.png", 8, 12.0),
    "spirit_bomb": ("spirit_bomb_sheet.png", 10, 14.0),
}

_cache: dict[str, SpriteSheet] = {}


def get_sheet(key: str) -> SpriteSheet:
    """Lazy-load + cache a SpriteSheet by registry key."""
    if key not in _cache:
        if key not in _SHEET_REGISTRY:
            raise KeyError(f"Unknown sprite sheet key: '{key}'")
        fname, n, fps = _SHEET_REGISTRY[key]
        path = os.path.join(SPRITES_DIR, fname)
        _cache[key] = SpriteSheet(path, n, fps)
    return _cache[key]
