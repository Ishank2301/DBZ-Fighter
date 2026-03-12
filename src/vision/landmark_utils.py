"""
Utility functions for working with MediaPipe landmark dicts.
Named landmark indices from mediapipe.solutions.pose.PoseLandmark.
"""

from typing import Optional

# ── Named landmark indices ────────────────────────────────────────────────────
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# ── Helpers ───────────────────────────────────────────────────────────────────


def get_xy(landmarks: list[dict], idx: int) -> tuple[float, float]:
    lm = landmarks[idx]
    return lm["x"], lm["y"]


def get_xyz(landmarks: list[dict], idx: int) -> tuple[float, float, float]:
    lm = landmarks[idx]
    return lm["x"], lm["y"], lm["z"]


def visibility(landmarks: list[dict], idx: int) -> float:
    return landmarks[idx]["visibility"]


def midpoint(landmarks: list[dict], a: int, b: int) -> tuple[float, float]:
    ax, ay = get_xy(landmarks, a)
    bx, by = get_xy(landmarks, b)
    return (ax + bx) / 2, (ay + by) / 2


def distance_2d(landmarks: list[dict], a: int, b: int) -> float:
    ax, ay = get_xy(landmarks, a)
    bx, by = get_xy(landmarks, b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def angle_3pts(
    landmarks: list[dict],
    a: int,
    b: int,
    c: int,
    use_z: bool = False,
) -> float:
    """
    Angle at joint b formed by segments a-b and b-c.
    Returns degrees [0, 180].
    """
    import math

    if use_z:
        ax, ay, az = get_xyz(landmarks, a)
        bx, by, bz = get_xyz(landmarks, b)
        cx, cy, cz = get_xyz(landmarks, c)
        v1 = (ax - bx, ay - by, az - bz)
        v2 = (cx - bx, cy - by, cz - bz)
        dot = sum(x * y for x, y in zip(v1, v2))
        mag1 = sum(x**2 for x in v1) ** 0.5
        mag2 = sum(x**2 for x in v2) ** 0.5
    else:
        ax, ay = get_xy(landmarks, a)
        bx, by = get_xy(landmarks, b)
        cx, cy = get_xy(landmarks, c)
        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        mag2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

    if mag1 * mag2 < 1e-9:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def both_visible(landmarks: list[dict], *indices, threshold: float = 0.5) -> bool:
    return all(visibility(landmarks, i) >= threshold for i in indices)
