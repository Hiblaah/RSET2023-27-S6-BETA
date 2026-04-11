"""
ml_detector.py
Loads trained ML model and predicts webcam behaviour
"""

import joblib
import numpy as np
import os
try:
    from mlmodel.popup_alert import alert_if_suspicious
except ImportError:
    from popup_alert import alert_if_suspicious
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model", "isolation_forest.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "model", "features.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "model", "threshold.pkl"))


def predict(features_dict):

    X = np.array([[

        features_dict["is_known_app"],
        features_dict["is_foreground"],
        features_dict["user_active"],
        features_dict["is_night"],
        features_dict["has_network_connection"],
        features_dict["network_connection_count"],
        features_dict["duration_minutes"]

    ]])

    X_scaled = scaler.transform(X)

    score = model.decision_function(X_scaled)[0]

    prediction = model.predict(X_scaled)[0]

    result = "Normal" if prediction == 1 else "Suspicious"
    alert_if_suspicious(prediction, score, features=features_dict)
    return result, score