from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

DATASET_PATH = "../mlmodel/training_data.xlsx"
import json
from pathlib import Path

ALERT_FILE = Path("alerts/dashboard_alerts.json")

@app.route("/alerts")
def get_alerts():
    if ALERT_FILE.exists():
        with open(ALERT_FILE) as f:
            return json.load(f)
    return []
def load_data():

    if not os.path.exists(DATASET_PATH):
        return []

    df = pd.read_excel(DATASET_PATH)

    events = []

    for _, row in df.iterrows():

        status = "Suspicious" if row["duration_minutes"] > 2 else "Normal"

        events.append({
            "application": "Unknown Process",
            "date": "Today",
            "time": "Now",
            "duration": f"{round(row['duration_minutes'],2)} min",
            "status": status
        })

    return events


@app.route("/")
def dashboard():

    events = load_data()

    total = len(events)

    suspicious = len([e for e in events if e["status"] == "Suspicious"])

    normal = total - suspicious

    return render_template(
        "index.html",
        events=events,
        total=total,
        normal=normal,
        suspicious=suspicious
    )


if __name__ == "__main__":
    print("Starting Webcam Dashboard...")
    app.run(host="127.0.0.1", port=5000, debug=True)