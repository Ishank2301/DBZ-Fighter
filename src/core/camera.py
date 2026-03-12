import cv2
from src.core.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS


class Camera:
    def __init__(self, index: int = CAMERA_INDEX):
        self.index = index
        self.cap = None
        self.open()

    def open(self):
        self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"[Camera] Could not open the camera at index {self.index}"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        print(f"[Camera] Opened at {FRAME_HEIGHT}*{FRAME_HEIGHT} @ {TARGET_FPS}fps")

    def read(self):
        """Capture a single frame.
        Returns (success: bool, frame: np.ndarray).
        Frame is flipped horizontally (mirror mode).
        """
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)
        return success, frame

    def release(self):
        if self.cap:
            self.cap.release()
            print("[Camera] Released.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.released()
