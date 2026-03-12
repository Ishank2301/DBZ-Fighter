"""
Full-screen overlay utilities:
  - Teleport flash (white/green screen flash)
  - Power-up screen shake simulation (frame border glow)
  - Spirit-bomb dimming
  - General semi-transparent tint helper
"""

import cv2
import numpy as np
from src.utils.math_utils import clamp


class OverlayManager:
    """
    Manages transient full-frame overlay effects.
    Call trigger(gesture) on activation, update(dt) each frame, draw(frame).
    """

    def __init__(self):
        self._effects: list[dict] = []  # list of active overlay descriptors

    def trigger(self, gesture: str):
        if gesture == "teleport":
            # White-green flash, fades quickly
            self._effects.append(
                {
                    "type": "flash",
                    "color": (150, 255, 100),  # BGR light green
                    "alpha": 0.75,
                    "decay": 4.0,
                }
            )
        elif gesture == "power_up":
            # Golden border glow
            self._effects.append(
                {
                    "type": "border",
                    "color": (30, 200, 255),  # BGR gold
                    "alpha": 0.6,
                    "decay": 1.5,
                }
            )
        elif gesture == "kamehameha":
            # Cyan border flash
            self._effects.append(
                {
                    "type": "border",
                    "color": (255, 220, 80),  # BGR cyan
                    "alpha": 0.5,
                    "decay": 2.0,
                }
            )
        elif gesture == "spirit_bomb":
            # Blue overlay dimming
            self._effects.append(
                {
                    "type": "tint",
                    "color": (120, 60, 20),  # BGR dark-blue tint
                    "alpha": 0.35,
                    "decay": 0.8,
                }
            )
        elif gesture == "firing":
            self._effects.append(
                {
                    "type": "flash",
                    "color": (200, 140, 60),  # BGR light blue
                    "alpha": 0.30,
                    "decay": 6.0,
                }
            )

    def update(self, dt: float):
        for eff in self._effects:
            eff["alpha"] -= eff["decay"] * dt
        self._effects = [e for e in self._effects if e["alpha"] > 0]

    def draw(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        for eff in self._effects:
            a = clamp(eff["alpha"], 0.0, 1.0)
            col = eff["color"]

            if eff["type"] == "flash":
                overlay = np.full_like(frame, col, dtype=np.uint8)
                cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)

            elif eff["type"] == "border":
                # Gradient border (bright edges, transparent centre)
                thickness = max(8, int(min(w, h) * 0.06))
                for i in range(thickness):
                    t = 1.0 - i / thickness
                    alf = a * t * 0.9
                    cv2.rectangle(frame, (i, i), (w - i - 1, h - i - 1), col, 1)
                # Additive tint only on border region
                border_mask = np.zeros((h, w), dtype=np.float32)
                border_mask[:thickness, :] = 1
                border_mask[-thickness:, :] = 1
                border_mask[:, :thickness] = 1
                border_mask[:, -thickness:] = 1
                for c_idx, c_val in enumerate(col):
                    frame[:, :, c_idx] = np.clip(
                        frame[:, :, c_idx].astype(np.float32)
                        + border_mask * c_val * a * 0.5,
                        0,
                        255,
                    ).astype(np.uint8)

            elif eff["type"] == "tint":
                overlay = np.full_like(frame, col, dtype=np.uint8)
                cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)

    @property
    def is_active(self) -> bool:
        return len(self._effects) > 0
