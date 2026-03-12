"""
MediaPipe Pose wrapper. Converts raw BGR frames → normalised landmark lists.
"""

import cv2
import mediapipe as mp
import numpy as np
from src.core.config import (
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
    N_LANDMARKS,
)


class PoseDetector:
    """
    Thin wrapper around mediapipe.solutions.pose.Pose.
    Exposes a single process() call that returns a list of dicts:
        [{'x': float, 'y': float, 'z': float, 'visibility': float}, ...]
    or None if no person is detected.
    """

    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            model_complexity=MP_MODEL_COMPLEXITY,
            min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MP_MIN_TRACKING_CONFIDENCE,
            enable_segmentation=False,
        )
        self._draw = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def process(self, frame_bgr: np.ndarray):
        """
        Parameters
        ----------
        frame_bgr : H×W×3 BGR uint8 ndarray

        Returns
        -------
        landmarks : list[dict] | None
            33 normalised landmarks in image coords [0, 1],
            or None if no pose detected.
        results : mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList
            Raw mediapipe results (for drawing).
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        if results.pose_landmarks is None:
            return None, results

        lms = results.pose_landmarks.landmark
        landmarks = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }
            for lm in lms
        ]
        assert len(landmarks) == N_LANDMARKS
        return landmarks, results

    def draw_landmarks(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """Draw pose skeleton on frame (in-place, returns frame)."""
        if results.pose_landmarks:
            self._draw.draw_landmarks(
                frame_bgr,
                results.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._draw.DrawingSpec(
                    color=(0, 220, 255), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=self._draw.DrawingSpec(
                    color=(180, 60, 255), thickness=2
                ),
            )
        return frame_bgr

    def close(self):
        self._pose.close()
