print("SCRIPT STARTED")

from ml_detector import predict
from popup_alert import alert_if_suspicious

features = {
    "is_known_app": 0,
    "is_foreground": 0,
    "user_active": 0,
    "is_night": 1,
    "has_network_connection": 1,
    "network_connection_count": 15,
    "duration_minutes": 20
}

prediction, score = predict(features)

print("\nManual ML Test")
print("----------------")
print("Prediction:", prediction)
print("Score:", score)

# Convert label to model format
pred_value = -1 if prediction == "Suspicious" else 1



