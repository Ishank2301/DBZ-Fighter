"""
Heads-up display renderer.
Draws the energy bar, gesture icon, gesture label, cooldown ring,
FPS counter, and confidence indicator onto the frame.
"""

from __future__ import annotations
import os
import cv2
import numpy as np

from src.core.config import (
    MAX_ENERGY,
    HUD_ENERGY_BAR_POS,
    HUD_GESTURE_ICON_POS,
    HUD_ICON_SIZE,
    HUD_GESTURE_LABEL_Y,
    HUD_COOLDOWN_POS,
    HUD_FONT_LARGE,
    HUD_FONT_SMALL,
    ICONS_DIR,
    HUD_DIR,
)
from src.gestures.state_machine import GestureState
from src.utils.math_utils import clamp

# DBZ colour palette (BGR for OpenCV)
_GOLD = (30, 200, 255)
_BLUE = (255, 160, 60)
_CYAN = (255, 220, 80)
_GREEN = (150, 255, 80)
_PURPLE = (255, 60, 200)
_WHITE = (255, 255, 255)
_DARK = (20, 20, 40)

_GESTURE_COLORS: dict[str, tuple] = {
    "idle": _WHITE,
    "charging": _GOLD,
    "firing": _BLUE,
    "kamehameha": _CYAN,
    "spirit_bomb": _BLUE,
    "teleport": _GREEN,
    "power_up": _GOLD,
    "block": _PURPLE,
}

_GESTURE_LABELS: dict[str, str] = {
    "idle": "IDLE",
    "charging": "CHARGING",
    "firing": "FIRING",
    "kamehameha": "KAMEHAMEHA",
    "spirit_bomb": "SPIRIT BOMB",
    "teleport": "TELEPORT",
    "power_up": "POWER UP",
    "block": "BLOCK",
}


class HUD:
    """
    Stateful HUD renderer.
    Call update(state, energy) each frame, then draw(frame).
    """

    def __init__(self):
        self._icons: dict[str, np.ndarray | None] = {}
        self._hud_imgs: dict[str, np.ndarray | None] = {}
        self._energy: float = MAX_ENERGY
        self._state: GestureState = GestureState()
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._mono_font = cv2.FONT_HERSHEY_DUPLEX
        self._load_assets()

    # ── Asset loading ─────────────────────────────────────────────────────────

    def _load_assets(self):
        """Load gesture icons and HUD image assets.  Missing = None (draws fallback)."""
        from src.core.config import GESTURES

        for g in GESTURES:
            path = os.path.join(ICONS_DIR, f"gesture_{g}.png")
            self._icons[g] = self._load_png(path)

        for key in (
            "energy_bar_bg",
            "energy_bar_fill",
            "energy_bar_frame",
            "cooldown_ring",
            "crosshair",
        ):
            path = os.path.join(HUD_DIR, f"{key}.png")
            self._hud_imgs[key] = self._load_png(path)

    @staticmethod
    def _load_png(path: str) -> np.ndarray | None:
        if not os.path.exists(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, state: GestureState, energy: float):
        self._state = state
        self._energy = clamp(energy, 0.0, MAX_ENERGY)

    def draw(self, frame: np.ndarray):
        self._draw_energy_bar(frame)
        self._draw_gesture_icon(frame)
        self._draw_gesture_label(frame)
        self._draw_hold_indicator(frame)

    def draw_fps(self, frame: np.ndarray, fps: float):
        cv2.putText(
            frame,
            f"FPS: {fps:.0f}",
            (frame.shape[1] - 120, 30),
            self._font,
            0.7,
            _GREEN,
            2,
            cv2.LINE_AA,
        )

    def draw_debug(self, frame: np.ndarray, probabilities: dict[str, float]):
        """Optional: show all gesture probabilities as a small bar chart."""
        x0, y0 = frame.shape[1] - 200, 60
        for i, (g, p) in enumerate(probabilities.items()):
            bar_w = int(p * 120)
            col = _GESTURE_COLORS.get(g, _WHITE)
            cv2.rectangle(
                frame, (x0, y0 + i * 20), (x0 + bar_w, y0 + i * 20 + 14), col, -1
            )
            cv2.putText(
                frame,
                f"{g[:8]} {p:.2f}",
                (x0 + 125, y0 + i * 20 + 12),
                self._font,
                0.38,
                col,
                1,
                cv2.LINE_AA,
            )

    # ── Private drawing helpers ───────────────────────────────────────────────

    def _draw_energy_bar(self, frame: np.ndarray):
        bx, by = HUD_ENERGY_BAR_POS
        bar_w = 200
        bar_h = 18
        fill_pct = self._energy / MAX_ENERGY

        # Background
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (30, 30, 50), -1)
        # Fill — colour shifts: gold (full) → blue (half) → red (low)
        if fill_pct > 0.5:
            col = _GOLD
        elif fill_pct > 0.25:
            col = _BLUE
        else:
            col = (50, 50, 255)  # BGR red

        fill_px = int(bar_w * fill_pct)
        if fill_px > 0:
            cv2.rectangle(frame, (bx, by), (bx + fill_px, by + bar_h), col, -1)

        # Border
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), _WHITE, 1)

        # Label
        cv2.putText(
            frame,
            f"KI: {int(self._energy)}",
            (bx, by - 5),
            self._font,
            0.55,
            _GOLD,
            1,
            cv2.LINE_AA,
        )

    def _draw_gesture_icon(self, frame: np.ndarray):
        g = self._state.gesture
        icon = self._icons.get(g)
        ix, iy = HUD_GESTURE_ICON_POS
        size = HUD_ICON_SIZE

        if icon is not None:
            # Scale and alpha-composite
            scaled = cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)
            self._blit(frame, scaled, ix, iy)
        else:
            # Fallback coloured square
            col = _GESTURE_COLORS.get(g, _WHITE)
            cv2.rectangle(frame, (ix, iy), (ix + size, iy + size), col, 2)
            cv2.putText(
                frame,
                g[:3].upper(),
                (ix + 4, iy + size - 6),
                self._font,
                0.6,
                col,
                2,
                cv2.LINE_AA,
            )

    def _draw_gesture_label(self, frame: np.ndarray):
        g = self._state.gesture
        col = _GESTURE_COLORS.get(g, _WHITE)
        lbl = _GESTURE_LABELS.get(g, g.upper())

        # Shadow
        cv2.putText(
            frame,
            lbl,
            (HUD_GESTURE_ICON_POS[0] + 2, HUD_GESTURE_LABEL_Y + 2),
            self._mono_font,
            0.75,
            _DARK,
            3,
            cv2.LINE_AA,
        )
        # Text
        cv2.putText(
            frame,
            lbl,
            (HUD_GESTURE_ICON_POS[0], HUD_GESTURE_LABEL_Y),
            self._mono_font,
            0.75,
            col,
            2,
            cv2.LINE_AA,
        )

        # Confidence bar below label
        conf_w = int(self._state.confidence * 80)
        bx, by = HUD_GESTURE_ICON_POS[0], HUD_GESTURE_LABEL_Y + 8
        cv2.rectangle(frame, (bx, by), (bx + 80, by + 5), (40, 40, 60), -1)
        if conf_w > 0:
            cv2.rectangle(frame, (bx, by), (bx + conf_w, by + 5), col, -1)

    def _draw_hold_indicator(self, frame: np.ndarray):
        """Small arc showing hold progress toward gesture activation."""
        if self._state.hold_pct <= 0:
            return
        cx, cy = HUD_COOLDOWN_POS
        r = 14
        start = -90
        end = int(-90 + 360 * self._state.hold_pct)
        g = self._state.gesture
        col = _GESTURE_COLORS.get(g, _WHITE)

        cv2.ellipse(frame, (cx + r, cy + r), (r, r), 0, start, end, col, 3, cv2.LINE_AA)
        cv2.ellipse(
            frame, (cx + r, cy + r), (r, r), 0, 0, 360, (60, 60, 80), 1, cv2.LINE_AA
        )

    @staticmethod
    def _blit(dst: np.ndarray, src_bgra: np.ndarray, x: int, y: int):
        dh, dw = dst.shape[:2]
        sh, sw = src_bgra.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(dw, x + sw), min(dh, y + sh)
        if x2 <= x1 or y2 <= y1:
            return
        sx1 = x1 - x
        sy1 = y1 - y
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)
        src = src_bgra[sy1:sy2, sx1:sx2]
        a = src[:, :, 3:4].astype(np.float32) / 255.0
        roi = dst[y1:y2, x1:x2].astype(np.float32)
        blended = roi * (1 - a) + src[:, :, :3].astype(np.float32) * a
        dst[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
