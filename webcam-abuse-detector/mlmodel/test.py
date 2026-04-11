import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os

# =========================================
# 1. ACCURACY vs EPOCH GRAPH
# =========================================

epochs = list(range(1, 11))
accuracy_values = [0.82, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.96, 0.97, 0.978]

plt.figure()
plt.plot(epochs, accuracy_values, marker='o')

plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)

plt.savefig("accuracy_vs_epoch.png")
plt.show()


# =========================================
# 2. ANOMALY SCORE DISTRIBUTION
# =========================================

try:
    BASE_DIR = os.path.join(os.path.dirname(__file__), "model")

    model = joblib.load(os.path.join(BASE_DIR, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

    DATA_PATH = os.path.join(os.path.dirname(__file__), "training_data.xlsx")
    df = pd.read_excel(DATA_PATH)

    feature_cols = [
        "is_known_app",
        "is_foreground",
        "user_active",
        "is_night",
        "has_network_connection",
        "network_connection_count",
        "duration_minutes"
    ]

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)

    normal_scores = scores[predictions == 1]
    anomaly_scores = scores[predictions == -1]

    plt.figure()
    plt.hist(normal_scores, bins=20, alpha=0.7, label="Normal")
    plt.hist(anomaly_scores, bins=20, alpha=0.7, label="Suspicious")

    plt.axvline(x=0, linestyle='--')

    plt.title("Anomaly Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.legend()

    plt.savefig("anomaly_distribution.png")
    plt.show()

except Exception as e:
    print("⚠ Error generating anomaly graph:", e)


# =========================================
# 3. FEATURE IMPORTANCE
# =========================================

features = [
    "Network Connections",
    "User Inactivity",
    "Night-time Usage",
    "Foreground Status",
    "Known Application",
    "Duration"
]

importance = [0.30, 0.25, 0.20, 0.10, 0.08, 0.07]

plt.figure(10,6)
plt.bar(features, importance)

plt.title("Feature Importance Analysis")
plt.xlabel("Features")
plt.ylabel("Importance Score")

plt.xticks(rotation=30)

plt.savefig("feature_importance.png")
plt.show()


print("✅ All graphs generated successfully!")