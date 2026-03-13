import joblib
import numpy as np
from pathlib import Path


class GestureClassifier:

    def __init__(self, model_path="models/gesture_model.pkl"):

        self.model = None
        self.labels = []
        self.is_trained = False

        path = Path(model_path)

        if path.exists():

            data = joblib.load(path)

            self.model = data["model"]
            self.labels = data["labels"]

            self.is_trained = True

            print("[GestureClassifier] Model loaded")

        else:

            print("[GestureClassifier] No trained model found")
            print("[GestureClassifier] Falling back to rule-based gestures")

    def predict(self, features: np.ndarray):

        if not self.is_trained:
            return None, 0.0

        features = features.reshape(1, -1)

        probs = self.model.predict_proba(features)[0]
        idx = np.argmax(probs)

        label = self.labels[idx]
        confidence = float(probs[idx])

        return label, confidence
