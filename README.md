# ⚡ Gesture Vision Engine

<div align="center">

```
 ██████╗ ███████╗███████╗████████╗██╗   ██╗██████╗ ███████╗
██╔════╝ ██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗██╔════╝
██║  ███╗█████╗  ███████╗   ██║   ██║   ██║██████╔╝█████╗  
██║   ██║██╔══╝  ╚════██║   ██║   ██║   ██║██╔══██╗██╔══╝  
╚██████╔╝███████╗███████║   ██║   ╚██████╔╝██║  ██║███████╗
 ╚═════╝ ╚══════╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
          ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗
          ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║
          ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║
          ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║
           ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║
            ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                    ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗
                    ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝
                    █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  
                    ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  
                    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗
                    ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝
```

**Real-time pose-based gesture recognition with ML classification and visual effects**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-red?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

</div>

---

## 🎮 What It Does

```
      📷 WEBCAM
          │
          ▼
   ┌─────────────┐        ┌──────────────────────────────────────────┐
   │  MediaPipe  │───────▶│  33 Body Landmarks  [x, y, z, visibility]│
   │  Pose Model │        └──────────────────────────────────────────┘
   └─────────────┘                          │
                                            ▼
                              ┌─────────────────────────┐
                              │   Feature Extractor      │
                              │  • Raw xyz (99 values)   │
                              │  • Joint angles (8)      │
                              │  • Key distances (6)     │
                              │  • Velocity delta (99)   │
                              └────────────┬────────────┘
                                           │
                         ┌─────────────────┴──────────────────┐
                         │                                     │
                    ┌────▼────┐                         ┌──────▼──────┐
                    │ ML Mode │                         │  Rule Mode  │
                    │Random   │                         │  Angle /    │
                    │Forest   │                         │  Distance   │
                    └────┬────┘                         └──────┬──────┘
                         └─────────────────┬──────────────────┘
                                           ▼
                               ┌───────────────────────┐
                               │    State Machine       │
                               │  idle → charging       │
                               │  charging → firing     │
                               │  firing → cooldown     │
                               └───────────┬───────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                       ▼
             ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐
             │  Particles  │    │   Aura / Flash   │    │   HUD Overlay    │
             │  Blast FX   │    │   Screen Shake   │    │  Energy / FPS    │
             └─────────────┘    └──────────────────┘    └──────────────────┘
```

---

## 🥋 Supported Gestures

| Gesture | Trigger Pose | Effect |
|---|---|---|
| ⚡ **Charging** | Arms bent inward at chest | Gold aura pulse |
| 🔥 **Firing** | One arm extended forward | Directional blast particles |
| 💥 **Kamehameha** | Both hands together, pushed forward | Wide energy beam |
| 🌐 **Spirit Bomb** | Both arms raised high above head | Expanding ring burst |
| ✨ **Power Up** | Arms spread wide | Ring + floating sparkles |
| 💨 **Teleport** | Body leaned sharply to one side | Flash + fade effect |
| 🛡 **Block** | Arms raised and crossed | Shield shimmer |
| ⬜ **Idle** | Natural standing | Nothing |

---

## 🏗️ System Architecture

```
gesture-vision-engine/
│
├── 📁 assets/
│   ├── effects/          ← Future: sprite sheets, effect textures
│   ├── sounds/           ← Future: audio triggers per gesture
│   └── ui/               ← Future: custom HUD skins
│
├── 📁 data/
│   └── gesture_dataset/  ← .npy files per gesture label (collected by you)
│
├── 📁 src/
│   │
│   ├── 🧠 core/
│   │   ├── camera.py         ← Webcam wrapper (OpenCV)
│   │   ├── config.py         ← ALL parameters in one place
│   │   └── performance.py    ← FPS, latency, memory tracking
│   │
│   ├── 👁️ vision/
│   │   ├── pose_detector.py      ← MediaPipe Pose wrapper
│   │   ├── landmark_utils.py     ← Angles, distances, geometry
│   │   └── feature_extractor.py  ← Builds ML feature vector
│   │
│   ├── 🤜 gestures/
│   │   ├── gesture_classifier.py ← Random Forest / GB ML model
│   │   ├── gesture_rules.py      ← Rule-based fallback detector
│   │   └── state_machine.py      ← Hold gating + cooldown FSM
│   │
│   ├── ✨ effects/
│   │   ├── effect_engine.py  ← Orchestrates all visual effects
│   │   ├── animation.py      ← Flash, shake, aura pulse
│   │   └── particles.py      ← GPU-style particle system
│   │
│   ├── 🖥️ ui/
│   │   ├── hud.py     ← Energy bar, gesture label, charge meter
│   │   └── overlay.py ← Debug panel (toggle with D key)
│   │
│   └── 🔧 utils/
│       ├── math_utils.py  ← lerp, clamp, pixel convert, angle 2D
│       └── smoothing.py   ← EMA landmark smoother, vote buffer
│
├── 🚀 main.py                 ← Full engine entry point
├── 📊 collect_dataset.py      ← Record gesture training data
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/your-username/gesture-vision-engine.git
cd gesture-vision-engine

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Immediately (Rule-Based Mode)

No training needed. Starts with rule-based gesture detection:

```bash
python main.py
```

### 3. Collect Your Own Gesture Data

```bash
# Collect 200 frames for each gesture
python collect_dataset.py --gesture charging    --frames 200
python collect_dataset.py --gesture firing      --frames 200
python collect_dataset.py --gesture kamehameha  --frames 200
python collect_dataset.py --gesture spirit_bomb --frames 200
python collect_dataset.py --gesture idle        --frames 200
```

> Stand in front of your webcam, hold the pose, and press **SPACE** to capture each frame.

### 4. Train the ML Model

```bash
# Press T inside the running engine, or run directly:
python -c "
from src.gestures.gesture_classifier import GestureClassifier
clf = GestureClassifier()
clf.train(model_type='random_forest')
"
```

### 5. Run with ML Mode

```bash
python main.py
# Engine auto-loads the model from models/gesture_classifier.pkl
```

---

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `D` | Toggle debug overlay |
| `R` | Reset state machine & effects |
| `T` | Train ML model from collected dataset |
| `C` | Reminder to use collect_dataset.py |

---

## 📈 ML Pipeline Detail

```
Raw landmarks (33 × 3 = 99 values)
         +
Joint angles (8 values, normalised 0–1)
  - Right elbow, Left elbow
  - Right shoulder, Left shoulder  
  - Right knee, Left knee
  - Cross-body shoulder angles
         +
Key distances (6 values)
  - Wrist-to-wrist
  - Each wrist to hip
  - Cross-body wrist-to-shoulder
  - Wrist distance / shoulder width ratio
         +
Velocity (99 values — frame delta of raw xyz)
         │
         ▼
  StandardScaler → RandomForestClassifier(200 trees)
         │
         ▼
  Confidence threshold filter (0.65)
         │
         ▼
  GestureVoteBuffer (majority vote over 5 frames)
         │
         ▼
  GestureStateMachine (hold gating + cooldown)
```

---

## ⚙️ Configuration

All parameters are in **`src/core/config.py`** — no magic numbers elsewhere:

```python
# Tune gesture sensitivity
CONFIDENCE_THRESHOLD = 0.65   # lower = more sensitive
CHARGE_HOLD_FRAMES   = 15     # frames to hold before triggering
COOLDOWN_FRAMES      = 30     # frames between triggers

# Tune effects
PARTICLE_COUNT  = 60
BLAST_SPEED     = 18

# Tune energy system
ENERGY_MAX           = 100
ENERGY_REGEN_RATE    = 0.4
ENERGY_COST_FIRE     = 30
ENERGY_COST_KAMEHAMEHA = 60
```

---

## 🛣️ Roadmap

- [x] Rule-based gesture detection
- [x] ML classifier (Random Forest)
- [x] Particle effects system
- [x] HUD with energy bar
- [x] State machine with hold gating
- [ ] LSTM sequence model for temporal gestures
- [ ] Sound effects per gesture
- [ ] Enemy AI + gesture-controlled game mode
- [ ] Multiplayer gesture battle mode
- [ ] Export as standalone `.exe`

---

## 🙏 Acknowledgements

- [MediaPipe](https://mediapipe.dev/) — real-time pose estimation
- [OpenCV](https://opencv.org/) — frame capture and rendering
- [scikit-learn](https://scikit-learn.org/) — ML classification pipeline
- Dragon Ball Z — for the gesture inspiration 🐉

---

<div align="center">

```
⚡ "It's not just a gesture — it's a technique." ⚡
```

**Star ⭐ the repo if you went Super Saiyan with it**

</div>