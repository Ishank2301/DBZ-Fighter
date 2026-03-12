"""
Central configuration for the Gesture Vision Engine.
All tunable constants live here — no magic numbers scattered through modules.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Asset Paths:
SPRITES_DIR = os.path.join(ROOT, "assets", "effects", "sprites")
TEXTURES_DIR = os.path.join(ROOT, "assets", "effects", "textures")
SOUNDS_DIR = os.path.join(ROOT, "assets", "sounds")
ICONS_DIR = os.path.join(ROOT, "assets", "ui", "icons")
HUD_DIR = os.path.join(ROOT, "assets", "ui", "hud")
FONTS_DIR = os.path.join(ROOT, "assets", "ui", "fonts")
DATASET_DIR = os.path.join(ROOT, "data", "gesture_dataset")
MODEL_PATH = os.path.join(ROOT, "data", "gesture_model.pkl")

# Camera:
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 30

# MediaPipe Pose
MP_MIN_DETECTION_CONFIDENCE = 0.6
MP_MIN_TRACKING_CONFIDENCE = 0.5
MP_MODEL_COMPLEXITY = 1  # 0=fast, 1=balanced, 2=accurate

# Feature extraction
N_LANDMARKS = 33  # MediaPipe full-body landmarks
FEATURE_DIM = 212  # total feature vector length

# Gesture classes (order = label encoding)
GESTURES = [
    "idle",
    "charging",
    "firing",
    "kamehameha",
    "spirit_bomb",
    "teleport",
    "power_up",
    "block",
]
N_GESTURES = len(GESTURES)

# Smoothing / voting
EMA_ALPHA = 0.35  # exponential moving average for landmark smoothing
VOTE_BUFFER_SIZE = 5  # majority vote window (frames)
VOTE_THRESHOLD = 3  # minimum votes needed to confirm a gesture

# State machine
HOLD_FRAMES: dict[str, int] = {
    # gesture: minimum consecutive frames required before activation
    "idle": 1,
    "charging": 6,
    "firing": 4,
    "kamehameha": 8,
    "spirit_bomb": 10,
    "teleport": 5,
    "power_up": 6,
    "block": 4,
}
COOLDOWN_FRAMES: dict[str, int] = {
    # gesture: frames to wait before same gesture can fire again
    "idle": 0,
    "charging": 0,
    "firing": 18,
    "kamehameha": 45,
    "spirit_bomb": 60,
    "teleport": 30,
    "power_up": 40,
    "block": 15,
}

#  Effect system
MAX_ACTIVE_EFFECTS = 12
PARTICLE_POOL_SIZE = 256
EFFECT_ALPHA_DECAY = 6  # alpha reduction per frame for fading effects

# Spritesheet animation frame rates (fps) per effect type
AURA_FPS = 12
BEAM_FPS = 18
EXPLOSION_FPS = 20
RING_FPS = 16
SPARK_FPS = 24
SMOKE_FPS = 12
SPIRIT_BOMB_FPS = 14

# HUD layout
HUD_ENERGY_BAR_POS = (20, 30)  # top-left corner of energy bar
HUD_GESTURE_ICON_POS = (20, 70)  # top-left of active gesture icon
HUD_ICON_SIZE = 48  # displayed size (icons are 128px, scaled)
HUD_GESTURE_LABEL_Y = 130  # y-position of gesture name text
HUD_COOLDOWN_POS = (20, 140)  # cooldown ring position
HUD_FONT_LARGE = 28  # pt
HUD_FONT_SMALL = 18  # pt
HUD_FONT_NAME = "ShareTechMono-Regular.ttf"  # falls back to cv2 default

#  Energy system
MAX_ENERGY = 100.0
ENERGY_REGEN = 0.4  # units per frame at idle
ENERGY_COSTS: dict[str, float] = {
    "idle": 0.0,
    "charging": -2.0,  # negative = gain energy
    "firing": 8.0,
    "kamehameha": 25.0,
    "spirit_bomb": 35.0,
    "teleport": 15.0,
    "power_up": 10.0,
    "block": 0.5,
}

# Sound
AUDIO_ENABLED = True
MASTER_VOLUME = 0.8
MUSIC_VOLUME = 0.3
SFX_VOLUME = 0.9

#  Performance
SKIP_FRAMES = 0  # 0 = process every frame, 1 = every other, etc.
RENDER_SCALE = 1.0  # 0.5 = half-res rendering then upscale
