"""
Landmark smoothing using Exponential Moving Average (EMA)
to reduce jitter from MediaPipe pose detection.
"""


class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        """
        alpha : smoothing factor
        higher = more responsive
        lower = smoother but slower
        """
        self.alpha = alpha
        self.prev = None

    def smooth(self, landmarks):
        """
        landmarks: list of dicts [{x,y,z,visibility}, ...]
        returns smoothed landmarks
        """

        if landmarks is None:
            self.prev = None
            return None

        if self.prev is None:
            self.prev = landmarks
            return landmarks

        smoothed = []

        for i in range(len(landmarks)):
            curr = landmarks[i]
            prev = self.prev[i]

            smoothed.append(
                {
                    "x": prev["x"] * (1 - self.alpha) + curr["x"] * self.alpha,
                    "y": prev["y"] * (1 - self.alpha) + curr["y"] * self.alpha,
                    "z": prev["z"] * (1 - self.alpha) + curr["z"] * self.alpha,
                    "visibility": curr.get("visibility", 1.0),
                }
            )

        self.prev = smoothed
        return smoothed
