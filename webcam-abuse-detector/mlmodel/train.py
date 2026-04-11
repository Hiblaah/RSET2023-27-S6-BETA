"""
train.py — Train Isolation Forest on behavioral dataset
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

FEATURES = [
    'is_known_app',
    'is_foreground',
    'user_active',
    'is_night',
    'has_network_connection',
    'network_connection_count',
    'duration_minutes'
]

CONTAMINATION = 0.10


def train():

    print("\n📂 Loading dataset...")

    data_path = os.path.join(os.path.dirname(__file__), "training_data.xlsx")

    df = pd.read_excel(data_path)

    print(f"Dataset size: {len(df)} rows")

    X = df[FEATURES].values

# Split dataset
    X_train, X_test = train_test_split(
       X,
        test_size=0.3,
        random_state=42,
        shuffle=True
    )

    print("\n⚙ Scaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🌲 Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42
    )

    model.fit(X_train_scaled)

    print("✓ Model training complete")

    # Compute anomaly scores
    scores = model.decision_function(X_test_scaled)

    threshold = np.percentile(scores, 10)

    print("\n📉 Score statistics")
    print("----------------------")
    print(f"Score range : [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"Threshold   : {threshold:.4f}")

    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_dir, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(FEATURES, os.path.join(model_dir, "features.pkl"))
    joblib.dump(threshold, os.path.join(model_dir, "threshold.pkl"))

    print("\n✅ Training complete")


if __name__ == "__main__":

    print("\n🚀 Starting ML training pipeline\n")

    train()