# process_detector.py
"""
Detects suspicious processes ONLY when the webcam is actively in use.
Fires popup alert + logs to dashboard + appends to dataset.
"""

import psutil
import time
import datetime
import os
import json
import pandas as pd
from pathlib import Path
from mlmodel.popup_alert import alert_if_suspicious

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent          # project root
ALERT_LOG    = BASE_DIR / "alerts" / "alerts_log.txt"
DATASET_PATH = Path(__file__).parent / "training_data.xlsx"
ALERT_LOG.parent.mkdir(exist_ok=True)

# ── Apps that USE the webcam (safe, expected) ─────────────────────────────────
WEBCAM_APPS = {
    "zoom.exe",
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "teams.exe",
    "skype.exe",
    "obs64.exe",
    "webex.exe",
    "meet.exe",
}

# ── ONLY these specific tools are truly suspicious ────────────────────────────
# python.exe, cmd.exe, powershell.exe removed — these are normal system tools
SUSPICIOUS_APPS = {
    "zoom.exe",
    "nmap.exe",
    "wireshark.exe",
    "netcat.exe",
    "nc.exe",
    "mshta.exe",
    "wscript.exe",
    "cscript.exe",
    "regsvr32.exe",
    "rundll32.exe",
    "mimikatz.exe",
    "procdump.exe",
    "pwdump.exe",
    "cobaltstrike.exe",
    "metasploit.exe",
    "psexec.exe",
}


# ── Step 1: Check if webcam is actively in use ────────────────────────────────
def is_webcam_active() -> tuple:
    """
    Returns (True, app_name) if a known webcam app is currently running.
    Returns (False, '') if no webcam app is detected.
    """
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name']
            if name and name.lower() in WEBCAM_APPS:
                return True, name.lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False, ''

def get_camera_app():
    """
    Returns the name of the webcam application currently running.
    Used by process_identifier.py
    """
    webcam_on, camera_app = is_webcam_active()

    if webcam_on:
        return camera_app

    return "unknown"
    
# ── Step 2: Scan for suspicious processes ─────────────────────────────────────
def get_suspicious_processes() -> list:
    """Returns list of suspicious process names currently running."""
    found = []
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name']
            if name and name.lower() in SUSPICIOUS_APPS:
                found.append(name.lower())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


# ── Step 3: Log to alerts_log.txt ─────────────────────────────────────────────
def log_to_alerts(timestamp, camera_app, suspicious_proc, score):
    entry = (
        f"{timestamp} | CRITICAL | "
        f"camera={camera_app} | "
        f"suspicious_process={suspicious_proc} | "
        f"score={score:.4f}\n"
    )
    with open(ALERT_LOG, "a") as f:
        f.write(entry)
    print(f"   📝 Logged to alerts_log.txt")


# ── Step 4: Log to dashboard ──────────────────────────────────────────────────
DASHBOARD_LOG = BASE_DIR / "alerts" / "dashboard_alerts.json"

def log_to_dashboard(timestamp, camera_app, suspicious_proc, score):
    alert = {
        "timestamp":          timestamp,
        "label":              "Suspicious",
        "severity":           "CRITICAL",
        "score":              round(score, 4),
        "camera_app":         camera_app,
        "suspicious_process": suspicious_proc,
    }

    existing = []
    if DASHBOARD_LOG.exists():
        try:
            with open(DASHBOARD_LOG) as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.insert(0, alert)
    existing = existing[:100]

    with open(DASHBOARD_LOG, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"   📊 Dashboard alert saved")


# ── Step 5: Append row to training dataset ────────────────────────────────────
def log_to_dataset(camera_app, suspicious_proc):
    now    = datetime.datetime.now()
    is_night = 1 if now.hour >= 22 or now.hour < 6 else 0

    try:
        connections    = psutil.net_connections(kind='inet')
        net_conn_count = len([c for c in connections if c.status == 'ESTABLISHED'])
    except Exception:
        net_conn_count = 1

    new_row = {
        "timestamp":                now.strftime("%Y-%m-%d %H:%M:%S"),
        "is_known_app":             1,
        "is_foreground":            1,
        "user_active":              1,
        "is_night":                 is_night,
        "has_network_connection":   1 if net_conn_count > 0 else 0,
        "network_connection_count": min(net_conn_count, 12),
        "duration_minutes":         0.1,
        "camera_app":               camera_app,
        "suspicious_process":       suspicious_proc,
        "label":                    "Suspicious",
    }

    if DATASET_PATH.exists():
        df = pd.read_excel(DATASET_PATH)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(DATASET_PATH, index=False)
    print(f"   💾 Dataset updated: training_data.xlsx")


# ── Step 6: Master check — ONLY fires if webcam is active ─────────────────────
def check_suspicious_processes():
    """
    ONLY triggers alert if:
      1. A webcam app is currently running   ← GATE
      2. AND a suspicious process is running
    """
    webcam_on, camera_app = is_webcam_active()

    # GATE: do nothing if webcam is not active
    if not webcam_on:
        return False, None

    suspicious_procs = get_suspicious_processes()

    if not suspicious_procs:
        return False, None

    # Both conditions met — fire everything
    proc      = suspicious_procs[0]
    score     = -0.30
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n🚨 ALERT [{timestamp}]")
    print(f"   Camera app        : {camera_app}")
    print(f"   Suspicious process: {proc}")

    # 1 — Popup
    alert_if_suspicious(
         score=score,
         label="Suspicious",
         reason="Suspicious process detected while webcam is active",
         extra_info={
            "suspicious_process": proc,
            "camera_app": camera_app,
        }
    )

    # 2 — alerts_log.txt
    log_to_alerts(timestamp, camera_app, proc, score)

    # 3 — dashboard_alerts.json (read by app.py)
    log_to_dashboard(timestamp, camera_app, proc, score)

    # 4 — training_data.xlsx
    log_to_dataset(camera_app, proc)

    return True, proc


# ── Continuous monitor loop ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🟢 Process monitor started (webcam-gated). Press Ctrl+C to stop.\n")

    while True:
        webcam_on, camera_app = is_webcam_active()

        if not webcam_on:
            print(f"📷 [{time.strftime('%H:%M:%S')}] Webcam not active — skipping scan")
        else:
            print(f"📷 [{time.strftime('%H:%M:%S')}] Webcam ON ({camera_app}) — scanning...")
            is_susp, proc = check_suspicious_processes()
            if not is_susp:
                print(f"   ✅ No suspicious processes found")

        time.sleep(10)