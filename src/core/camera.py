"""
Webcam management with warm-up, resolution forcing, and graceful release.
"""

import cv2
import time
from src.core.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, TARGET_FPS


class Camera:
    """Wraps cv2.VideoCapture with resolution enforcement and warmup."""

    def __init__(self):
        self.cap = None
        self.width = CAMERA_WIDTH
        self.height = CAMERA_HEIGHT

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            print(f"[Camera] ERROR: Cannot open camera index {CAMERA_INDEX}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        # Warm up — discard first few frames (exposure settling)
        for _ in range(5):
            self.cap.read()
            time.sleep(0.05)

        # Read actual resolution (driver may override)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"[Camera] Opened at {self.width}x{self.height} "
            f"@ {int(self.cap.get(cv2.CAP_PROP_FPS))} fps"
        )
        return True

    def read(self):
        """Return (ok, frame) where frame is BGR uint8 ndarray."""
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
