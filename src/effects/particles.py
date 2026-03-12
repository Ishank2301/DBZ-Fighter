"""
Object-pooled particle system.
Particles are simple quads drawn with additive blending.
"""

import numpy as np
import cv2
from src.core.config import PARTICLE_POOL_SIZE
from src.utils.math_utils import clamp


class Particle:
    __slots__ = [
        "x",
        "y",
        "vx",
        "vy",
        "size",
        "alpha",
        "alpha_decay",
        "color_bgr",
        "active",
    ]

    def __init__(self):
        self.active = False

    def init(self, x, y, vx, vy, size, alpha, alpha_decay, color_bgr):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.color_bgr = color_bgr
        self.active = True

    def update(self, dt: float):
        if not self.active:
            return
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.alpha -= self.alpha_decay * dt * 60  # decay per second
        self.size *= max(0.0, 1.0 - 0.5 * dt)  # gentle shrink
        if self.alpha <= 0 or self.size < 0.5:
            self.active = False


class ParticleSystem:
    """
    Fixed-size object pool.  Emit particles by calling emit().
    Draw by calling draw(frame).
    """

    def __init__(self, pool_size: int = PARTICLE_POOL_SIZE):
        self._pool = [Particle() for _ in range(pool_size)]
        self._rng = np.random.default_rng(42)

    def _acquire(self) -> Particle | None:
        for p in self._pool:
            if not p.active:
                return p
        return None

    def emit(
        self,
        x: float,
        y: float,
        color_bgr: tuple,
        n: int = 8,
        speed: float = 80.0,
        size: float = 6.0,
        alpha: float = 200.0,
        alpha_decay: float = 120.0,
        directed: tuple[float, float] | None = None,
        spread: float = 3.14159,
    ):
        """
        Spawn n particles at (x, y).

        directed : (dx, dy) unit vector for directed emission.
                   If None, emission is omnidirectional.
        spread   : angular spread in radians around directed direction.
        """
        for _ in range(n):
            p = self._acquire()
            if p is None:
                break

            if directed is not None:
                base_angle = np.arctan2(directed[1], directed[0])
                angle = base_angle + self._rng.uniform(-spread / 2, spread / 2)
            else:
                angle = self._rng.uniform(0, 2 * 3.14159)

            s = speed * self._rng.uniform(0.5, 1.5)
            vx = np.cos(angle) * s
            vy = np.sin(angle) * s

            p.init(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                size=size * self._rng.uniform(0.5, 1.5),
                alpha=alpha,
                alpha_decay=alpha_decay,
                color_bgr=color_bgr,
            )

    def update(self, dt: float):
        for p in self._pool:
            if p.active:
                p.update(dt)

    def draw(self, frame: np.ndarray):
        """Draw all active particles onto frame (additive blend, in-place)."""
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame, dtype=np.float32)

        for p in self._pool:
            if not p.active:
                continue
            xi, yi = int(p.x), int(p.y)
            r = max(1, int(p.size))
            a = p.alpha / 255.0

            if xi - r >= w or xi + r < 0 or yi - r >= h or yi + r < 0:
                continue

            x1, y1 = max(0, xi - r), max(0, yi - r)
            x2, y2 = min(w - 1, xi + r), min(h - 1, yi + r)
            b, g, c_r = p.color_bgr

            overlay[y1:y2, x1:x2, 0] += b * a
            overlay[y1:y2, x1:x2, 1] += g * a
            overlay[y1:y2, x1:x2, 2] += c_r * a

        np.clip(frame.astype(np.float32) + overlay, 0, 255, out=overlay)
        frame[:] = overlay.astype(np.uint8)

    def active_count(self) -> int:
        return sum(1 for p in self._pool if p.active)
