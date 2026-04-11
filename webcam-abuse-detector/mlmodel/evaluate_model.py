import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = [
    'is_known_app',
    'is_foreground',
    'user_active',
    'is_night',
    'has_network_connection',
    'network_connection_count',
    'duration_minutes'
]

def evaluate():

    base_dir = os.path.dirname(__file__)

    print("\n📂 Loading model...")

    model = joblib.load(os.path.join(base_dir, "model/isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "model/scaler.pkl"))

    print("✓ Model loaded")

    print("\n📂 Loading dataset...")

    data = pd.read_excel(os.path.join(base_dir, "labeled_output.xlsx"))

    if "label" not in data.columns:
        raise ValueError("Dataset must contain 'label' column")

    X = data[FEATURES]
    y = data["label"]

    print(f"Dataset size: {len(data)} samples")

    # scale features
    X_scaled = scaler.transform(X)

    print("\n🔍 Predicting anomalies...")

    preds = model.predict(X_scaled)

    # convert Isolation Forest output
    preds = np.where(preds == -1, 1, 0)

    print("\n📊 Evaluation Results")
    print("------------------------")

    accuracy = accuracy_score(y, preds)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y, preds))

    print("\nClassification Report")
    print(classification_report(y, preds))


if __name__ == "__main__":

    print("\n🚀 Evaluating model\n")

    evaluate()