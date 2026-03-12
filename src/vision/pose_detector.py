"""
Wraps MediaPipe Pose to detect body landmarks from a BGR frame.
Returns normalised landmark data ready for feature extraction.
"""

import cv2
import mediapipe as mp
import numpy as np
from scr.core.config import (
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
)


class PoseDetector:
    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_style
        self.pose = self._mp_pose.Pose(
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
            model_complexity=MP_MODEL_COMPLEXITY,
        )
        self.results = None
        print("[PoseDetector] Initialized")

    def detect(self, frame: np.ndarray):
        """
        Run pose detection on a BGR frame.
        Returns (annotated_frame, landmarks_or_None).
        `landmarks` is a list of 33 dicts: {x, y, z, visibility}.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.pose.process(rgb)
        rgb.flags.writeable = True

        landmarks = None
        if self.results.pose_landmarks:
            landmarks = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                }
                for lm in self.results.pose_landmarks.landmark
            ]

        return frame, landmarks

    def draw_landmarks(self, frame: np.ndarray):
        if self.results and self.results.pose_landmarks:
            self._mp_draw.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
            )
            return frame

    def get_landmark_array(self, landmarks: list) -> np.ndarray:
        """
        Flatten landmarks into a 1D array: [x0,y0,z0, x1,y1,z1, ...]
        Shape: (99,) — 33 landmarks × 3 values (x, y, z).
        Visibility is excluded from the feature vector.
        """
        if landmarks is None:
            return np.zeros(33 * 3)
        return np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks]).flatten()

    def close(self):
        self.pose.close()

    # MediaPipe landmark index constants
    # Convenient aliases so other modules don't need to import mediapipe
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
