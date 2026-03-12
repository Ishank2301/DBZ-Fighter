"""
Gesture Vision Engine — main application loop.

Pipeline (per frame):
    WEBCAM
    → pose_detector  (MediaPipe → 33 landmarks)
    → smoother       (EMA)
    → feature_extractor (212-dim vector)
    → classifier     (ML model) or rules fallback
    → vote_buffer + state_machine (hold-gate + cooldown)
    → effect_engine  (spawn effects on activation)
    → hud            (draw KI bar, icon, label)
    → overlay        (full-frame flash effects)
    → cv2.imshow()

Controls:
    Q / Esc  — quit
    D        — toggle debug probability bars
    S        — save screenshot
    R        — reset state machine
"""

import os
import sys
import cv2
import numpy as np
import time

# ── Make sure project root is on the path ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.camera import Camera
from src.core.performance import FPSCounter
from src.core.config import (
    TARGET_FPS,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    ENERGY_COSTS,
    MAX_ENERGY,
    ENERGY_REGEN,
    GESTURES,
)
from src.vision.pose_detector import PoseDetector
from src.vision.feature_extractor import FeatureExtractor
from src.utils.smoothing import LandmarkSmoother
from src.gestures.gesture_classifier import GestureClassifier
from src.gestures.gesture_rules import RulesClassifier
from src.gestures.state_machine import StateMachine
from src.effects.effect_engine import EffectEngine
from src.ui.hud import HUD
from src.ui.overlay import OverlayManager
import src.vision.landmark_utils as lmu


# ── Sound (optional, graceful degradation) ───────────────────────────────────
try:
    import pygame

    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    _SOUND_OK = True
except Exception:
    _SOUND_OK = False


def _load_sounds() -> dict:
    """Load WAV files from assets/sounds/.  Missing files are skipped silently."""
    if not _SOUND_OK:
        return {}
    from src.core.config import SOUNDS_DIR

    mapping = {
        "charging": os.path.join(SOUNDS_DIR, "gestures", "charging_start.wav"),
        "firing": os.path.join(SOUNDS_DIR, "gestures", "firing.wav"),
        "kamehameha": os.path.join(SOUNDS_DIR, "gestures", "kamehameha_start.wav"),
        "spirit_bomb": os.path.join(SOUNDS_DIR, "gestures", "spirit_bomb.wav"),
        "teleport": os.path.join(SOUNDS_DIR, "gestures", "teleport.wav"),
        "power_up": os.path.join(SOUNDS_DIR, "gestures", "power_up.wav"),
        "block": os.path.join(SOUNDS_DIR, "gestures", "block.wav"),
        "cooldown_ready": os.path.join(SOUNDS_DIR, "ui", "cooldown_ready.wav"),
    }
    sounds = {}
    for key, path in mapping.items():
        if os.path.exists(path):
            try:
                sounds[key] = pygame.mixer.Sound(path)
            except Exception as e:
                print(f"[Sound] Could not load {path}: {e}")
    return sounds


def _play(sounds: dict, key: str):
    if key in sounds:
        try:
            sounds[key].play()
        except Exception:
            pass


# ── Body anchor helper ────────────────────────────────────────────────────────


def _body_anchor(landmarks, frame_w: int, frame_h: int) -> tuple[int, int]:
    """
    Mid-wrist position in pixel coords — used as effect anchor.
    Falls back to frame center if landmarks unavailable.
    """
    if landmarks is None:
        return frame_w // 2, frame_h // 2
    lx, ly = lmu.get_xy(landmarks, lmu.LEFT_WRIST)
    rx, ry = lmu.get_xy(landmarks, lmu.RIGHT_WRIST)
    mx = (lx + rx) / 2
    my = (ly + ry) / 2
    return int(mx * frame_w), int(my * frame_h)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    # ── Init subsystems ───────────────────────────────────────────────────────
    camera = Camera()
    if not camera.open():
        print("ERROR: Could not open camera. Exiting.")
        return

    W, H = camera.width, camera.height

    detector = PoseDetector()
    smoother = LandmarkSmoother()
    extractor = FeatureExtractor()
    classifier = GestureClassifier()
    rules = RulesClassifier()
    fsm = StateMachine()
    fx_engine = EffectEngine(W, H)
    hud = HUD()
    overlay = OverlayManager()
    fps_ctr = FPSCounter()
    sounds = _load_sounds()

    energy = float(MAX_ENERGY)
    debug_mode = False
    screenshot_n = 0

    print("\n" + "=" * 55)
    print("  Gesture Vision Engine — Running")
    print("  Q/Esc=quit  D=debug  S=screenshot  R=reset")
    print("=" * 55 + "\n")

    dt = 1.0 / TARGET_FPS  # initial estimate, refined each frame

    prev_time = time.perf_counter()

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            print("[Main] Camera read failed — retrying …")
            time.sleep(0.05)
            continue

        # ── 1. Pose detection ─────────────────────────────────────────────────
        landmarks_raw, mp_results = detector.process(frame)
        landmarks = smoother.smooth(landmarks_raw)

        # ── 2. Feature extraction ─────────────────────────────────────────────
        features = extractor.extract(landmarks)

        # ── 3. Classification ─────────────────────────────────────────────────
        if classifier.is_trained:
            pred, conf = classifier.predict(features)
            probas = classifier.predict_proba(features)
        else:
            # Fallback to rule-based (uses raw landmarks, not features)
            pred, conf = rules.predict(landmarks)
            probas = {g: (conf if g == pred else 0.0) for g in GESTURES}

        # ── 4. State machine ──────────────────────────────────────────────────
        state = fsm.update(pred, conf)

        # ── 5. Effects + sounds on activation ────────────────────────────────
        if state.is_active and state.gesture != "idle":
            ax, ay = _body_anchor(landmarks, W, H)
            fx_engine.trigger(state.gesture, ax, ay)
            overlay.trigger(state.gesture)
            _play(sounds, state.gesture)

        # ── 6. Energy system ──────────────────────────────────────────────────
        cost = ENERGY_COSTS.get(state.gesture, 0.0)
        energy -= cost * dt * 30  # normalise to per-second
        energy += ENERGY_REGEN * dt * 30
        energy = max(0.0, min(MAX_ENERGY, energy))

        # ── 7. Update systems ─────────────────────────────────────────────────
        fx_engine.update(dt)
        overlay.update(dt)
        hud.update(state, energy)

        # ── 8. Render ─────────────────────────────────────────────────────────
        # Optional: draw skeleton
        # detector.draw_landmarks(frame, mp_results)

        # Effects (particles + sprites)
        fx_engine.draw(frame)

        # Overlay (full-frame flashes)
        overlay.draw(frame)

        # HUD
        hud.draw(frame)
        hud.draw_fps(frame, fps_ctr.fps)

        if debug_mode:
            hud.draw_debug(frame, probas)

        # ── 9. Display ────────────────────────────────────────────────────────
        cv2.imshow("Gesture Vision Engine", frame)

        # ── 10. Input ─────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            debug_mode = not debug_mode
            print(f"[Main] Debug mode: {debug_mode}")
        elif key == ord("s"):
            fname = f"screenshot_{screenshot_n:04d}.png"
            cv2.imwrite(fname, frame)
            print(f"[Main] Screenshot saved → {fname}")
            screenshot_n += 1
        elif key == ord("r"):
            fsm.reset()
            smoother.reset()
            extractor._prev = None
            print("[Main] State machine reset.")

        # ── 11. Frame timing ──────────────────────────────────────────────────
        now = time.perf_counter()
        dt = now - prev_time
        dt = max(0.001, min(dt, 0.1))  # clamp to [1ms, 100ms]
        prev_time = now
        fps_ctr.tick()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    camera.release()
    detector.close()
    cv2.destroyAllWindows()
    if _SOUND_OK:
        pygame.mixer.quit()
    print("[Main] Exited cleanly.")


if __name__ == "__main__":
    main()
