"""
Converts a smoothed list of 33 MediaPipe landmarks into a fixed-length
212-dimensional feature vector suitable for the ML classifier.

Feature breakdown (212 total):
  [0:99]   — normalised (x, y, z) for all 33 landmarks         = 99 dims
             (normalised relative to shoulder-midpoint & torso scale)
  [99:165] — pairwise distances for 66 key joint pairs         = 66 dims
  [165:195]— joint angles for 30 important body angles         = 30 dims
  [195:212]— velocity features (Δ per frame) for 17 key joints = 17 dims
"""

import numpy as np
from src.vision import landmark_utils as lmu


# ── Key joint pairs for pairwise distance features (66 pairs) ────────────────
_DIST_PAIRS = [
    # wrist–shoulder
    (lmu.LEFT_WRIST, lmu.LEFT_SHOULDER),
    (lmu.RIGHT_WRIST, lmu.RIGHT_SHOULDER),
    # wrist–elbow
    (lmu.LEFT_WRIST, lmu.LEFT_ELBOW),
    (lmu.RIGHT_WRIST, lmu.RIGHT_ELBOW),
    # elbow–shoulder
    (lmu.LEFT_ELBOW, lmu.LEFT_SHOULDER),
    (lmu.RIGHT_ELBOW, lmu.RIGHT_SHOULDER),
    # wrist–hip
    (lmu.LEFT_WRIST, lmu.LEFT_HIP),
    (lmu.RIGHT_WRIST, lmu.RIGHT_HIP),
    # wrist–wrist (key for charging / kamehameha)
    (lmu.LEFT_WRIST, lmu.RIGHT_WRIST),
    # wrist–nose (spirit bomb / teleport)
    (lmu.LEFT_WRIST, lmu.NOSE),
    (lmu.RIGHT_WRIST, lmu.NOSE),
    # wrist–knee (power-up wide stance)
    (lmu.LEFT_WRIST, lmu.LEFT_KNEE),
    (lmu.RIGHT_WRIST, lmu.RIGHT_KNEE),
    # shoulder–shoulder
    (lmu.LEFT_SHOULDER, lmu.RIGHT_SHOULDER),
    # hip–shoulder
    (lmu.LEFT_HIP, lmu.LEFT_SHOULDER),
    (lmu.RIGHT_HIP, lmu.RIGHT_SHOULDER),
    # hand–forehead (teleport: index to nose area)
    (lmu.RIGHT_INDEX, lmu.NOSE),
    (lmu.LEFT_INDEX, lmu.NOSE),
    # elbows cross (block X)
    (lmu.LEFT_ELBOW, lmu.RIGHT_WRIST),
    (lmu.RIGHT_ELBOW, lmu.LEFT_WRIST),
    # knee–ankle
    (lmu.LEFT_KNEE, lmu.LEFT_ANKLE),
    (lmu.RIGHT_KNEE, lmu.RIGHT_ANKLE),
    # hip–knee
    (lmu.LEFT_HIP, lmu.LEFT_KNEE),
    (lmu.RIGHT_HIP, lmu.RIGHT_KNEE),
    # shoulder-midpoint to wrists (vertical raise)
    (lmu.LEFT_WRIST, lmu.RIGHT_HIP),
    (lmu.RIGHT_WRIST, lmu.LEFT_HIP),
    # extra
    (lmu.LEFT_PINKY, lmu.RIGHT_PINKY),
    (lmu.LEFT_THUMB, lmu.RIGHT_THUMB),
    (lmu.NOSE, lmu.LEFT_HIP),
    (lmu.NOSE, lmu.RIGHT_HIP),
    (lmu.LEFT_SHOULDER, lmu.RIGHT_HIP),
    (lmu.RIGHT_SHOULDER, lmu.LEFT_HIP),
    (lmu.LEFT_WRIST, lmu.RIGHT_SHOULDER),
    (lmu.RIGHT_WRIST, lmu.LEFT_SHOULDER),
    (lmu.LEFT_ELBOW, lmu.RIGHT_ELBOW),
    (lmu.LEFT_ANKLE, lmu.RIGHT_ANKLE),
    (lmu.LEFT_HEEL, lmu.RIGHT_HEEL),
    (lmu.LEFT_WRIST, lmu.LEFT_KNEE),
    (lmu.RIGHT_WRIST, lmu.RIGHT_KNEE),
    (lmu.LEFT_ELBOW, lmu.LEFT_HIP),
    (lmu.RIGHT_ELBOW, lmu.RIGHT_HIP),
    # thumbs / index fingers to shoulders
    (lmu.LEFT_THUMB, lmu.LEFT_SHOULDER),
    (lmu.RIGHT_THUMB, lmu.RIGHT_SHOULDER),
    (lmu.LEFT_INDEX, lmu.LEFT_SHOULDER),
    (lmu.RIGHT_INDEX, lmu.RIGHT_SHOULDER),
    # cross-body
    (lmu.LEFT_INDEX, lmu.RIGHT_SHOULDER),
    (lmu.RIGHT_INDEX, lmu.LEFT_SHOULDER),
    (lmu.LEFT_PINKY, lmu.LEFT_HIP),
    (lmu.RIGHT_PINKY, lmu.RIGHT_HIP),
    # eye-level hands
    (lmu.LEFT_WRIST, lmu.LEFT_EYE),
    (lmu.RIGHT_WRIST, lmu.RIGHT_EYE),
    # foot-level hands
    (lmu.LEFT_WRIST, lmu.LEFT_FOOT_INDEX),
    (lmu.RIGHT_WRIST, lmu.RIGHT_FOOT_INDEX),
    # nose to shoulders
    (lmu.NOSE, lmu.LEFT_SHOULDER),
    (lmu.NOSE, lmu.RIGHT_SHOULDER),
    # ear–shoulder
    (lmu.LEFT_EAR, lmu.LEFT_SHOULDER),
    (lmu.RIGHT_EAR, lmu.RIGHT_SHOULDER),
    # more wrist combos
    (lmu.LEFT_WRIST, lmu.RIGHT_ELBOW),
    (lmu.RIGHT_WRIST, lmu.LEFT_ELBOW),
    (lmu.LEFT_WRIST, lmu.LEFT_ANKLE),
    (lmu.RIGHT_WRIST, lmu.RIGHT_ANKLE),
    (lmu.LEFT_PINKY, lmu.RIGHT_WRIST),
    (lmu.RIGHT_PINKY, lmu.LEFT_WRIST),
    (lmu.MOUTH_LEFT, lmu.LEFT_WRIST),
    (lmu.MOUTH_RIGHT, lmu.RIGHT_WRIST),
]
assert len(_DIST_PAIRS) == 66, f"Expected 66 pairs, got {len(_DIST_PAIRS)}"

# Key angle triplets (30 angles)
_ANGLES = [
    # elbow angles
    (lmu.LEFT_SHOULDER, lmu.LEFT_ELBOW, lmu.LEFT_WRIST),
    (lmu.RIGHT_SHOULDER, lmu.RIGHT_ELBOW, lmu.RIGHT_WRIST),
    # shoulder angles
    (lmu.LEFT_ELBOW, lmu.LEFT_SHOULDER, lmu.LEFT_HIP),
    (lmu.RIGHT_ELBOW, lmu.RIGHT_SHOULDER, lmu.RIGHT_HIP),
    # hip–shoulder–elbow
    (lmu.LEFT_HIP, lmu.LEFT_SHOULDER, lmu.LEFT_ELBOW),
    (lmu.RIGHT_HIP, lmu.RIGHT_SHOULDER, lmu.RIGHT_ELBOW),
    # knee
    (lmu.LEFT_HIP, lmu.LEFT_KNEE, lmu.LEFT_ANKLE),
    (lmu.RIGHT_HIP, lmu.RIGHT_KNEE, lmu.RIGHT_ANKLE),
    # hip tilt
    (lmu.LEFT_SHOULDER, lmu.LEFT_HIP, lmu.LEFT_KNEE),
    (lmu.RIGHT_SHOULDER, lmu.RIGHT_HIP, lmu.RIGHT_KNEE),
    # wrist–elbow–shoulder angle (arm direction)
    (lmu.LEFT_WRIST, lmu.LEFT_ELBOW, lmu.LEFT_SHOULDER),
    (lmu.RIGHT_WRIST, lmu.RIGHT_ELBOW, lmu.RIGHT_SHOULDER),
    # shoulder span angle
    (lmu.LEFT_ELBOW, lmu.LEFT_SHOULDER, lmu.RIGHT_SHOULDER),
    (lmu.RIGHT_ELBOW, lmu.RIGHT_SHOULDER, lmu.LEFT_SHOULDER),
    # cross-body wrist–shoulder angles (block X detection)
    (lmu.LEFT_WRIST, lmu.LEFT_SHOULDER, lmu.RIGHT_SHOULDER),
    (lmu.RIGHT_WRIST, lmu.RIGHT_SHOULDER, lmu.LEFT_SHOULDER),
    # nose–shoulder angle (head tilt)
    (lmu.NOSE, lmu.LEFT_SHOULDER, lmu.RIGHT_SHOULDER),
    # wrist vertical angle (above/below shoulder)
    (lmu.LEFT_WRIST, lmu.LEFT_SHOULDER, lmu.LEFT_HIP),
    (lmu.RIGHT_WRIST, lmu.RIGHT_SHOULDER, lmu.RIGHT_HIP),
    # full arm vs body
    (lmu.LEFT_WRIST, lmu.LEFT_HIP, lmu.LEFT_KNEE),
    (lmu.RIGHT_WRIST, lmu.RIGHT_HIP, lmu.RIGHT_KNEE),
    # hip angle (wide stance)
    (lmu.LEFT_KNEE, lmu.LEFT_HIP, lmu.RIGHT_HIP),
    (lmu.RIGHT_KNEE, lmu.RIGHT_HIP, lmu.LEFT_HIP),
    # ankle–knee–hip
    (lmu.LEFT_ANKLE, lmu.LEFT_KNEE, lmu.LEFT_HIP),
    (lmu.RIGHT_ANKLE, lmu.RIGHT_KNEE, lmu.RIGHT_HIP),
    # shoulder–elbow–hip (charging detection)
    (lmu.LEFT_SHOULDER, lmu.LEFT_ELBOW, lmu.LEFT_HIP),
    (lmu.RIGHT_SHOULDER, lmu.RIGHT_ELBOW, lmu.RIGHT_HIP),
    # index finger–wrist–elbow
    (lmu.LEFT_INDEX, lmu.LEFT_WRIST, lmu.LEFT_ELBOW),
    (lmu.RIGHT_INDEX, lmu.RIGHT_WRIST, lmu.RIGHT_ELBOW),
    # nose–hip–knee (lean)
    (lmu.NOSE, lmu.LEFT_HIP, lmu.LEFT_KNEE),
]
assert len(_ANGLES) == 30, f"Expected 30 angles, got {len(_ANGLES)}"

# Velocity joints (17)
_VEL_JOINTS = [
    lmu.LEFT_WRIST,
    lmu.RIGHT_WRIST,
    lmu.LEFT_ELBOW,
    lmu.RIGHT_ELBOW,
    lmu.LEFT_SHOULDER,
    lmu.RIGHT_SHOULDER,
    lmu.LEFT_HIP,
    lmu.RIGHT_HIP,
    lmu.NOSE,
    lmu.LEFT_PINKY,
    lmu.RIGHT_PINKY,
    lmu.LEFT_INDEX,
    lmu.RIGHT_INDEX,
    lmu.LEFT_KNEE,
    lmu.RIGHT_KNEE,
    lmu.LEFT_ANKLE,
    lmu.RIGHT_ANKLE,
]
assert len(_VEL_JOINTS) == 17


class FeatureExtractor:
    """
    Stateful: remembers the previous landmark frame to compute velocities.
    Call extract(landmarks) each frame.
    """

    def __init__(self):
        self._prev: list[dict] | None = None

    def extract(self, landmarks: list[dict]) -> np.ndarray:
        """
        Returns float32 ndarray of shape (212,).
        If landmarks is None, returns zero vector.
        """
        if landmarks is None:
            self._prev = None
            return np.zeros(212, dtype=np.float32)

        vec = np.zeros(212, dtype=np.float32)

        # 1. Normalised positions (0:99)
        # Reference: mid-shoulder as origin, torso height as scale
        sx = (
            landmarks[lmu.LEFT_SHOULDER]["x"] + landmarks[lmu.RIGHT_SHOULDER]["x"]
        ) / 2
        sy = (
            landmarks[lmu.LEFT_SHOULDER]["y"] + landmarks[lmu.RIGHT_SHOULDER]["y"]
        ) / 2
        hx = (landmarks[lmu.LEFT_HIP]["x"] + landmarks[lmu.RIGHT_HIP]["x"]) / 2
        hy = (landmarks[lmu.LEFT_HIP]["y"] + landmarks[lmu.RIGHT_HIP]["y"]) / 2
        scale = max(abs(sy - hy), 1e-5)

        for i, lm in enumerate(landmarks):
            vec[i * 3 + 0] = (lm["x"] - sx) / scale
            vec[i * 3 + 1] = (lm["y"] - sy) / scale
            vec[i * 3 + 2] = lm["z"] / scale
        # 33*3 = 99 dims used

        #  2. Pairwise distances (99:165)
        for j, (a, b) in enumerate(_DIST_PAIRS):
            ax, ay = lmu.get_xy(landmarks, a)
            bx, by = lmu.get_xy(landmarks, b)
            d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            vec[99 + j] = d / scale

        #  3. Joint angles (165:195)
        for k, (a, b, c) in enumerate(_ANGLES):
            ang = lmu.angle_3pts(landmarks, a, b, c)
            vec[165 + k] = ang / 180.0  # normalise to [0, 1]

        #  4. Velocities (195:212)
        if self._prev is not None:
            for m, idx in enumerate(_VEL_JOINTS):
                px, py = lmu.get_xy(self._prev, idx)
                cx, cy = lmu.get_xy(landmarks, idx)
                vel = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                vec[195 + m] = vel / scale

        self._prev = landmarks
        return vec
