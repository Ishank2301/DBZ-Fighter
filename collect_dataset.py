"""
collect_dataset.py
------------------
Interactive dataset collection tool.
Saves feature vectors (212-dim) + labels for each gesture to disk.
After collecting samples for all gestures, trains and saves the ML model.

Usage:
    python collect_dataset.py

Controls (during collection):
    SPACE  — start/stop recording for current gesture
    N      — next gesture
    T      — train model now (using collected data)
    Q/Esc  — quit
"""

import os
import sys
import cv2
import numpy as np
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.camera import Camera
from src.core.config import GESTURES, DATASET_DIR, TARGET_FPS
from src.vision.pose_detector import PoseDetector
from src.vision.feature_extractor import FeatureExtractor
from src.utils.smoothing import LandmarkSmoother

os.makedirs(DATASET_DIR, exist_ok=True)

SAMPLES_PER_GESTURE = 300  # target samples per class


def save_samples(gesture: str, X: list, append: bool = True):
    path = os.path.join(DATASET_DIR, f"{gesture}.npy")
    arr = np.array(X, dtype=np.float32)
    if append and os.path.exists(path):
        existing = np.load(path)
        arr = np.vstack([existing, arr])
    np.save(path, arr)
    print(f"[Dataset] Saved {len(arr)} samples for '{gesture}' → {path}")


def load_all() -> tuple[np.ndarray, np.ndarray]:
    """Load all gesture data. Returns (X, y) arrays."""
    X_all, y_all = [], []
    for label_idx, g in enumerate(GESTURES):
        path = os.path.join(DATASET_DIR, f"{g}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            X_all.append(arr)
            y_all.extend([label_idx] * len(arr))
            print(f"  {g:15s}: {len(arr)} samples")
    if not X_all:
        return np.empty((0, 212)), np.empty((0,), dtype=int)
    return np.vstack(X_all), np.array(y_all, dtype=int)


def main():
    camera = Camera()
    if not camera.open():
        print("ERROR: Could not open camera.")
        return

    detector = PoseDetector()
    smoother = LandmarkSmoother()
    extractor = FeatureExtractor()

    gesture_idx = 0
    recording = False
    buffer: list[np.ndarray] = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    W, H = camera.width, camera.height

    print("\n" + "=" * 55)
    print("  Gesture Dataset Collector")
    print("  SPACE=record  N=next  T=train  Q=quit")
    print("=" * 55)

    while gesture_idx < len(GESTURES):
        gesture = GESTURES[gesture_idx]
        ok, frame = camera.read()
        if not ok:
            continue

        landmarks_raw, _ = detector.process(frame)
        landmarks = smoother.smooth(landmarks_raw)
        features = extractor.extract(landmarks)

        if recording and landmarks is not None:
            buffer.append(features.copy())

        # ── UI ────────────────────────────────────────────────────────────────
        status = "● REC" if recording else "○ PAUSED"
        g_color = (0, 255, 80) if recording else (200, 200, 200)

        cv2.rectangle(frame, (0, 0), (W, 70), (20, 20, 40), -1)
        cv2.putText(
            frame,
            f"Gesture: {gesture.upper()}",
            (20, 35),
            font,
            1.0,
            (255, 210, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{status}  {len(buffer)}/{SAMPLES_PER_GESTURE}",
            (20, 62),
            font,
            0.7,
            g_color,
            2,
            cv2.LINE_AA,
        )

        prog = int((len(buffer) / SAMPLES_PER_GESTURE) * (W - 40))
        cv2.rectangle(frame, (20, H - 20), (20 + prog, H - 8), (30, 200, 255), -1)
        cv2.rectangle(frame, (20, H - 20), (W - 20, H - 8), (80, 80, 100), 1)

        cv2.imshow("Dataset Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            recording = not recording
            if not recording and buffer:
                save_samples(gesture, buffer)
                buffer = []
        elif key == ord("n"):
            if buffer:
                save_samples(gesture, buffer)
                buffer = []
            recording = False
            gesture_idx += 1
            smoother.reset()
            extractor._prev = None
            if gesture_idx < len(GESTURES):
                print(f"\n→ Next: {GESTURES[gesture_idx]}")
        elif key == ord("t"):
            if buffer:
                save_samples(gesture, buffer, append=True)
                buffer = []
            print("\n[Training] Loading all data …")
            X, y = load_all()
            if len(X) > 0:
                from src.gestures.gesture_classifier import GestureClassifier

                clf = GestureClassifier()
                clf.train(X, y)
            else:
                print("[Training] No data found!")

        # Auto-advance when target reached
        if recording and len(buffer) >= SAMPLES_PER_GESTURE:
            save_samples(gesture, buffer)
            buffer = []
            recording = False
            print(f"  Target reached for '{gesture}'. Press N for next gesture.")

    camera.release()
    detector.close()
    cv2.destroyAllWindows()

    # Final train if data was collected
    print("\n[Dataset] Collection complete. Training final model …")
    X, y = load_all()
    if len(X) > 0:
        from src.gestures.gesture_classifier import GestureClassifier

        clf = GestureClassifier()
        clf.train(X, y)
    print("Done.")


if __name__ == "__main__":
    main()
