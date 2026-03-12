"""
Central effect dispatcher.

Usage:
    engine = EffectEngine(w, h)
    engine.trigger("kamehameha", anchor_x, anchor_y)
    engine.update(dt)
    engine.draw(frame)

Effect lifecycles are managed internally as a list of active Effect instances.
Each Effect has an update(dt) and draw(frame) method.
"""

from __future__ import annotations
import cv2
import numpy as np
import math

from src.effects.animation import get_sheet, Animator
from src.effects.particles import ParticleSystem
from src.core.config import MAX_ACTIVE_EFFECTS, EFFECT_ALPHA_DECAY
from src.utils.math_utils import clamp, ease_out_quad


# ── Base Effect ───────────────────────────────────────────────────────────────


class Effect:
    def __init__(self):
        self.alive = True

    def update(self, dt: float):
        pass

    def draw(self, frame: np.ndarray):
        pass


# ── Sprite Effect (plays a spritesheet at a location) ────────────────────────


class SpriteEffect(Effect):
    def __init__(
        self,
        sheet_key: str,
        x: int,
        y: int,
        scale: float = 1.0,
        loop: bool = False,
    ):
        super().__init__()
        sheet = get_sheet(sheet_key)
        self.animator = Animator(sheet, loop=loop)
        self.x, self.y = x, y
        self.scale = scale

    def update(self, dt: float):
        self.animator.update(dt)
        if self.animator.finished:
            self.alive = False

    def draw(self, frame: np.ndarray):
        img = self.animator.current_frame()
        if img is None:
            return
        h, w = img.shape[:2]
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        if nw != w or nh != h:
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        _blit_bgra(frame, img, self.x - nw // 2, self.y - nh // 2)


# ── Beam Effect (horizontal sweep) ───────────────────────────────────────────


class BeamEffect(Effect):
    def __init__(
        self,
        sheet_key: str,
        x: int,
        y: int,
        direction: float = 0.0,  # degrees
        length: int = 300,
        scale: float = 1.0,
    ):
        super().__init__()
        sheet = get_sheet(sheet_key)
        self.animator = Animator(sheet, loop=True)
        self.x, self.y = x, y
        self.direction = math.radians(direction)
        self.length = length
        self.scale = scale
        self._lifetime = 0.0
        self._max_life = sheet.frame_count / sheet.fps * 2

    def update(self, dt: float):
        self.animator.update(dt)
        self._lifetime += dt
        if self._lifetime >= self._max_life:
            self.alive = False

    def draw(self, frame: np.ndarray):
        img = self.animator.current_frame()
        if img is None:
            return
        h, w = img.shape[:2]
        nw = self.length
        nh = max(1, int(h * self.scale))
        if nw != w or nh != h:
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Rotate if needed
        if abs(self.direction) > 0.01:
            angle_deg = math.degrees(self.direction)
            M = cv2.getRotationMatrix2D((0, nh // 2), -angle_deg, 1.0)
            img = cv2.warpAffine(
                img,
                M,
                (nw + nh, nh + nh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        _blit_bgra(frame, img, self.x, self.y - nh // 2)


# ── Aura Effect (looping aura around body) ───────────────────────────────────


class AuraEffect(Effect):
    def __init__(self, sheet_key: str, x: int, y: int, scale: float = 1.5):
        super().__init__()
        sheet = get_sheet(sheet_key)
        self.animator = Animator(sheet, loop=True)
        self.x, self.y = x, y
        self.scale = scale
        self._duration = 0.0
        self._max_dur = 3.0  # auto-stop after 3 s if not refreshed

    def refresh(self):
        self._duration = 0.0

    def update(self, dt: float):
        self.animator.update(dt)
        self._duration += dt
        if self._duration >= self._max_dur:
            self.alive = False

    def draw(self, frame: np.ndarray):
        img = self.animator.current_frame()
        if img is None:
            return
        h, w = img.shape[:2]
        nw = max(1, int(w * self.scale))
        nh = max(1, int(h * self.scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        _blit_bgra(frame, img, self.x - nw // 2, self.y - nh)


# ── Ring pulse ────────────────────────────────────────────────────────────────


class RingEffect(Effect):
    def __init__(self, sheet_key: str, x: int, y: int, scale: float = 1.0):
        super().__init__()
        sheet = get_sheet(sheet_key)
        self.animator = Animator(sheet, loop=False)
        self.x, self.y = x, y
        self.scale = scale

    def update(self, dt: float):
        self.animator.update(dt)
        if self.animator.finished:
            self.alive = False

    def draw(self, frame: np.ndarray):
        img = self.animator.current_frame()
        if img is None:
            return
        h, w = img.shape[:2]
        nw = max(1, int(w * self.scale))
        nh = max(1, int(h * self.scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        _blit_bgra(frame, img, self.x - nw // 2, self.y - nh // 2)


# ── Effect Engine ─────────────────────────────────────────────────────────────


class EffectEngine:
    def __init__(self, frame_w: int, frame_h: int):
        self.w = frame_w
        self.h = frame_h
        self._effects: list[Effect] = []
        self.particles: ParticleSystem = ParticleSystem()

    def trigger(self, gesture: str, x: int, y: int):
        """
        Spawn the appropriate visual effects for the given gesture.
        x, y — body anchor point in pixel coords (e.g. mid-wrist position).
        """
        if gesture == "charging":
            self._spawn(AuraEffect("aura_gold", x, y, scale=1.2))
            self._spawn(RingEffect("ring_gold", x, y, scale=0.8))
            self.particles.emit(x, y, (30, 180, 255), n=12, speed=60, size=5, alpha=200)

        elif gesture == "firing":
            self._spawn(BeamEffect("beam_blue", x, y, direction=0, length=400))
            self._spawn(SpriteEffect("spark_blue", x, y, scale=0.7, loop=False))
            self.particles.emit(
                x,
                y,
                (255, 130, 30),
                n=10,
                speed=120,
                alpha=220,
                directed=(1, 0),
                spread=0.5,
            )

        elif gesture == "kamehameha":
            self._spawn(BeamEffect("beam_blue", x, y, direction=0, length=self.w))
            self._spawn(AuraEffect("aura_blue", x, y, scale=1.8))
            self._spawn(SpriteEffect("explosion_blue", x, y, scale=1.2))
            self.particles.emit(x, y, (255, 210, 40), n=30, speed=200, alpha=240)

        elif gesture == "spirit_bomb":
            self._spawn(SpriteEffect("spirit_bomb", x, y - 80, scale=2.0))
            self._spawn(AuraEffect("aura_blue", x, y, scale=2.2))
            self._spawn(RingEffect("ring_blue", x, y, scale=1.5))
            self.particles.emit(x, y - 80, (255, 200, 100), n=40, speed=80, alpha=200)

        elif gesture == "teleport":
            self._spawn(RingEffect("ring_gold", x, y, scale=2.0))
            self._spawn(SpriteEffect("spark_white", x, y, scale=1.5))
            self.particles.emit(x, y, (160, 255, 100), n=20, speed=300, alpha=255)

        elif gesture == "power_up":
            self._spawn(AuraEffect("aura_gold", x, y, scale=2.5))
            self._spawn(RingEffect("ring_gold", x, y, scale=1.8))
            self._spawn(SpriteEffect("explosion_gold", x, y, scale=1.5))
            self.particles.emit(
                x, y, (30, 180, 255), n=50, speed=150, size=8, alpha=230
            )

        elif gesture == "block":
            self._spawn(RingEffect("ring_purple", x, y, scale=1.2))
            self._spawn(SpriteEffect("spark_white", x, y, scale=1.0))
            self.particles.emit(x, y, (255, 60, 200), n=16, speed=100, alpha=210)

    def update(self, dt: float):
        self.particles.update(dt)
        for eff in self._effects:
            eff.update(dt)
        # Prune dead effects
        self._effects = [e for e in self._effects if e.alive]

    def draw(self, frame: np.ndarray):
        self.particles.draw(frame)
        for eff in self._effects:
            eff.draw(frame)

    def clear(self):
        self._effects.clear()

    def _spawn(self, eff: Effect):
        if len(self._effects) < MAX_ACTIVE_EFFECTS:
            self._effects.append(eff)


# ── Blitter helper ────────────────────────────────────────────────────────────


def _blit_bgra(dst: np.ndarray, src: np.ndarray, x: int, y: int):
    """
    Alpha-composite a BGRA src image onto a BGR dst image at position (x, y).
    Handles partial overlap / out-of-bounds gracefully.
    """
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]

    # Clip source to destination bounds
    sx1 = max(0, -x)
    sy1 = max(0, -y)
    sx2 = min(sw, dw - x)
    sy2 = min(sh, dh - y)
    dx1 = max(0, x)
    dy1 = max(0, y)
    dx2 = dx1 + (sx2 - sx1)
    dy2 = dy1 + (sy2 - sy1)

    if sx2 <= sx1 or sy2 <= sy1:
        return

    src_crop = src[sy1:sy2, sx1:sx2]
    dst_crop = dst[dy1:dy2, dx1:dx2]

    alpha = src_crop[:, :, 3:4].astype(np.float32) / 255.0
    bgr = src_crop[:, :, :3].astype(np.float32)

    blended = dst_crop.astype(np.float32) * (1 - alpha) + bgr * alpha
    dst[dy1:dy2, dx1:dx2] = np.clip(blended, 0, 255).astype(np.uint8)
