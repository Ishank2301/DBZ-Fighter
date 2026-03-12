"""
Hold-gating + cooldown state machine.

Pipeline:
    raw_prediction (per frame)
    → vote_buffer (5-frame majority)
    → hold_gate   (must hold N frames)
    → cooldown    (lockout after firing)
    → ACTIVE GESTURE

Output: GestureState dataclass
"""

from __future__ import annotations
import time
import collections
from dataclasses import dataclass, field

from src.core.config import (
    VOTE_BUFFER_SIZE,
    VOTE_THRESHOLD,
    HOLD_FRAMES,
    COOLDOWN_FRAMES,
    GESTURES,
)


@dataclass
class GestureState:
    gesture: str = "idle"
    confidence: float = 0.0
    is_active: bool = False  # True on the first confirmed activation frame
    hold_pct: float = 0.0  # 0→1 how far through hold gate we are
    cooldown_pct: float = 0.0  # 0→1 how far through cooldown we are
    frame: int = 0  # running frame counter


class StateMachine:
    """
    Single-gesture state machine.
    Call update(prediction, confidence) once per frame.
    """

    def __init__(self):
        # Vote buffer
        self._vote_buf: collections.deque[str] = collections.deque(
            maxlen=VOTE_BUFFER_SIZE
        )

        # Hold gate
        self._candidate: str = "idle"
        self._hold_count: int = 0

        # Confirmed state
        self._current: str = "idle"
        self._conf: float = 0.0
        self._just_active: bool = False

        # Cooldowns  {gesture: remaining_frames}
        self._cooldowns: dict[str, int] = {g: 0 for g in GESTURES}

        self._frame: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, prediction: str, confidence: float) -> GestureState:
        self._frame += 1

        # 1. Vote buffer majority
        self._vote_buf.append(prediction)
        voted = self._majority_vote()

        # 2. Tick cooldowns
        for g in GESTURES:
            if self._cooldowns[g] > 0:
                self._cooldowns[g] -= 1

        # 3. Hold gate
        self._just_active = False

        if voted != self._candidate:
            self._candidate = voted
            self._hold_count = 0
        else:
            self._hold_count += 1

        required_hold = HOLD_FRAMES.get(self._candidate, 4)

        if self._hold_count >= required_hold:
            # Check cooldown
            if self._cooldowns[self._candidate] == 0:
                if self._candidate != self._current:
                    self._just_active = True
                    # Trigger cooldown for the *previous* gesture
                    self._cooldowns[self._current] = COOLDOWN_FRAMES.get(
                        self._current, 0
                    )
                self._current = self._candidate
                self._conf = confidence

        # 4. Build state
        cd_total = COOLDOWN_FRAMES.get(self._current, 1)
        cd_remain = self._cooldowns.get(self._current, 0)
        cd_pct = 1.0 - (cd_remain / cd_total) if cd_total > 0 else 1.0

        hd_total = max(HOLD_FRAMES.get(self._candidate, 1), 1)
        hd_pct = min(self._hold_count / hd_total, 1.0)

        return GestureState(
            gesture=self._current,
            confidence=self._conf,
            is_active=self._just_active,
            hold_pct=hd_pct,
            cooldown_pct=cd_pct,
            frame=self._frame,
        )

    def reset(self):
        self._vote_buf.clear()
        self._candidate = "idle"
        self._hold_count = 0
        self._current = "idle"
        self._conf = 0.0
        self._just_active = False
        self._cooldowns = {g: 0 for g in GESTURES}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _majority_vote(self) -> str:
        if not self._vote_buf:
            return "idle"
        counts: dict[str, int] = {}
        for g in self._vote_buf:
            counts[g] = counts.get(g, 0) + 1
        best_g = max(counts, key=counts.__getitem__)
        if counts[best_g] >= VOTE_THRESHOLD:
            return best_g
        # No clear majority — keep current
        return self._current
