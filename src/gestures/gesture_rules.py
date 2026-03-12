"""
Rule-based gesture detector used as a fallback when no ML model is loaded,
or as a confidence booster when ML is uncertain.

Each rule function takes a smoothed landmark list and returns a confidence
float in [0.0, 1.0], or 0.0 if the gesture is clearly not present.
"""

import math
from src.vision import landmark_utils as lmu
from src.core.config import GESTURES


def _safe_angle(lms, a, b, c) -> float:
    try:
        return lmu.angle_3pts(lms, a, b, c)
    except Exception:
        return 90.0


def _wrist_height_norm(lms, wrist_idx, shoulder_idx, hip_idx) -> float:
    """
    Returns how high the wrist is relative to torso height.
    0 = at hip level, 1 = at shoulder level, >1 = above shoulders.
    """
    wy = lms[wrist_idx]["y"]
    sy = lms[shoulder_idx]["y"]
    hy = lms[hip_idx]["y"]
    torso = abs(sy - hy) + 1e-6
    # In image coords y increases downward, so:
    # wrist above shoulder → wy < sy → large positive value
    return (hy - wy) / torso


class RulesClassifier:
    """
    Simple heuristic classifier.
    Returns (gesture_name, confidence) same interface as GestureClassifier.
    """

    def predict(self, landmarks: list[dict] | None) -> tuple[str, float]:
        if landmarks is None:
            return "idle", 0.0

        scores = {g: 0.0 for g in GESTURES}

        # ── CHARGING ──────────────────────────────────────────────────────────
        # Both fists pulled to sides at waist level, roughly symmetric
        lw_h = _wrist_height_norm(
            lms=landmarks,
            wrist_idx=lmu.LEFT_WRIST,
            shoulder_idx=lmu.LEFT_SHOULDER,
            hip_idx=lmu.LEFT_HIP,
        )
        rw_h = _wrist_height_norm(
            lms=landmarks,
            wrist_idx=lmu.RIGHT_WRIST,
            shoulder_idx=lmu.RIGHT_SHOULDER,
            hip_idx=lmu.RIGHT_HIP,
        )
        wrist_sep = lmu.distance_2d(landmarks, lmu.LEFT_WRIST, lmu.RIGHT_WRIST)
        shoulder_w = lmu.distance_2d(landmarks, lmu.LEFT_SHOULDER, lmu.RIGHT_SHOULDER)
        # Charging: both wrists near waist, widely separated
        if 0.2 < lw_h < 0.7 and 0.2 < rw_h < 0.7:
            if wrist_sep > shoulder_w * 0.9:
                scores["charging"] = min(1.0, wrist_sep / (shoulder_w * 1.5))

        # FIRING
        # One arm extended forward (elbow nearly straight), other bent or down
        l_elbow_ang = _safe_angle(
            landmarks, lmu.LEFT_SHOULDER, lmu.LEFT_ELBOW, lmu.LEFT_WRIST
        )
        r_elbow_ang = _safe_angle(
            landmarks, lmu.RIGHT_SHOULDER, lmu.RIGHT_ELBOW, lmu.RIGHT_WRIST
        )
        if r_elbow_ang > 145:  # right arm extended
            scores["firing"] = (r_elbow_ang - 145) / 35
        elif l_elbow_ang > 145:
            scores["firing"] = (l_elbow_ang - 145) / 35

        # KAMEHAMEHA
        # Both arms extended forward, wrists close together, at chest height
        both_extended = l_elbow_ang > 140 and r_elbow_ang > 140
        wrists_close = wrist_sep < shoulder_w * 0.5
        lw_mid = 0.5 < lw_h < 1.0
        rw_mid = 0.5 < rw_h < 1.0
        if both_extended and wrists_close and lw_mid and rw_mid:
            scores["kamehameha"] = min(1.0, (l_elbow_ang + r_elbow_ang - 280) / 80)

        # SPIRIT BOMB
        # At least one wrist well above head
        if lw_h > 1.5 or rw_h > 1.5:
            above = max(lw_h, rw_h)
            scores["spirit_bomb"] = min(1.0, (above - 1.5) / 0.8)

        # TELEPORT
        # Right index finger near nose/forehead
        right_index_nose_d = lmu.distance_2d(landmarks, lmu.RIGHT_INDEX, lmu.NOSE)
        head_size = lmu.distance_2d(landmarks, lmu.NOSE, lmu.LEFT_EAR) + 1e-6
        if right_index_nose_d < head_size * 0.8:
            scores["teleport"] = 1.0 - right_index_nose_d / (head_size * 0.8)

        # POWER UP
        # Both arms spread wide and slightly above hip, wide stance
        if lw_h > 0.4 and rw_h > 0.4 and wrist_sep > shoulder_w * 2.0:
            scores["power_up"] = min(1.0, wrist_sep / (shoulder_w * 3.0))

        # BLOCK
        # Arms crossed: left wrist over to right side, right wrist to left side
        lx = landmarks[lmu.LEFT_WRIST]["x"]
        rx = landmarks[lmu.RIGHT_WRIST]["x"]
        ls = landmarks[lmu.LEFT_SHOULDER]["x"]
        rs = landmarks[lmu.RIGHT_SHOULDER]["x"]
        mid_x = (ls + rs) / 2
        # If left wrist is to the right of midline, arms are crossed
        if lx > mid_x and rx < mid_x:
            cross_amount = abs(lx - rx) / (abs(ls - rs) + 1e-6)
            scores["block"] = min(1.0, cross_amount)

        #  Pick best
        best_g = max(scores, key=scores.__getitem__)
        best_conf = scores[best_g]

        if best_conf < 0.25:
            return "idle", 1.0 - best_conf

        return best_g, best_conf
