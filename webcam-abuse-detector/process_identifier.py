"""
process_identification.py
Identifies processes currently accessing the webcam on Windows
and extracts ML features.
"""
import pandas as pd
import os
import csv
import json
import time
import ctypes
import ctypes.wintypes
import datetime

import psutil
import winreg
from mlmodel.ml_detector import predict
from mlmodel.process_detector import get_camera_app 


LOG_FILE = os.path.join(os.getcwd(), "logs", "webcam_logs.csv")

def raise_alert(alert_data, score):

    alert_message = f"""
    ⚠ SUSPICIOUS WEBCAM ACTIVITY DETECTED ⚠

    Application: {alert_data["application"]}
    Time: {alert_data["time"]}
    Duration: {alert_data["duration"]} min
    Score: {score}

----------------------------------------
"""

    print(alert_message)

    os.makedirs("alerts", exist_ok=True)

    with open("alerts/alerts_log.txt", "a") as f:
        f.write(alert_message)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

OUTPUT_JSON = "process_features.json"
OUTPUT_CSV = "process_features.csv"
DATASET_PATH = "mlmodel/training_data.xlsx"
DASHBOARD_LOG = "../logs/webcam_logs.csv"

KNOWN_APPS = {
    "zoom", "ms-teams", "teams", "skype", "webex", "discord", "slack",
    "chrome", "firefox", "msedge", "opera", "brave",
    "obs64", "obs32", "vlc", "camera", "windowscamera",
    "facetime", "meet", "gotomeeting", "ringcentral",
}

FEATURE_KEYS = [
    "is_known_app",
    "is_foreground",
    "user_active",
    "is_night",
    "has_network_connection",
    "network_connection_count",
    "duration_minutes",
]

# Track session start time
_session_start = {}

# Windows APIs
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


# ─────────────────────────────────────────────
# STEP 1 — REGISTRY DETECTION
# ─────────────────────────────────────────────

def _get_pids_via_registry():

    pids = {}

    registry_paths = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam",
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam\NonPackaged",
    ]

    for reg_path in registry_paths:

        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path)
            i = 0

            while True:

                try:
                    subkey_name = winreg.EnumKey(key, i)
                    i += 1

                    subkey = winreg.OpenKey(key, subkey_name)

                    try:
                        start_time, _ = winreg.QueryValueEx(subkey, "LastUsedTimeStart")
                        stop_time, _ = winreg.QueryValueEx(subkey, "LastUsedTimeStop")

                        # Active webcam usage
                        if start_time > stop_time:

                            exe_name = subkey_name.replace("#", "\\").split("\\")[-1].lower()

                            for proc in psutil.process_iter(['pid','name','create_time','ppid']):

                                try:
                                    if proc.info['name'].lower() == exe_name:

                                        proc_obj = psutil.Process(proc.info['pid'])

                                        # Remove browser child processes
                                        parent = proc_obj.parent()
                                        if parent and parent.name().lower() == exe_name:
                                            continue

                                        # Ensure network activity
                                        try:
                                            connections = proc_obj.net_connections()
                                        except AttributeError:
                                            connections = proc_obj.connections()

                                        if len(connections) == 0:
                                            continue

                                        pids[proc.info['pid']] = proc.info['create_time']

                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    continue

                    except FileNotFoundError:
                        pass

                    winreg.CloseKey(subkey)

                except OSError:
                    break

            winreg.CloseKey(key)

        except OSError:
            continue

    return pids


# ─────────────────────────────────────────────
# STEP 2 — FALLBACK DETECTION
# ─────────────────────────────────────────────

def _get_pids_via_name_fallback():

    pids = {}

    for proc in psutil.process_iter(['pid','name','create_time','ppid']):

        try:
            name = proc.info['name'].lower().replace(".exe","")

            if not any(k in name for k in KNOWN_APPS):
                continue

            proc_obj = psutil.Process(proc.info['pid'])

            parent = proc_obj.parent()

            if parent and parent.name().lower().replace(".exe","") == name:
                continue

            try:
                connections = proc_obj.net_connections()
            except AttributeError:
                connections = proc_obj.connections()

            if len(connections) == 0:
                continue

            pids[proc.info['pid']] = proc.info['create_time']

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return pids


def identify_webcam_processes():

    pids = _get_pids_via_registry()

    if pids:
        print(f"[process_identification] Registry detected {len(pids)} webcam process(es).")

    else:
        print("[process_identification] Registry returned no active webcam processes.")
        print("[process_identification] Falling back to known-app name detection.")
        pids = _get_pids_via_name_fallback()

    return pids


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def _get_foreground_pid():

    hwnd = user32.GetForegroundWindow()

    pid = ctypes.wintypes.DWORD()

    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

    return pid.value


def _seconds_since_last_input():

    lii = LASTINPUTINFO()

    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)

    user32.GetLastInputInfo(ctypes.byref(lii))

    return (kernel32.GetTickCount() - lii.dwTime) / 1000.0


def extract_features(pid, proc_start, duration_minutes):

    now = time.time()
    dt = datetime.datetime.now()

    try:
        proc = psutil.Process(pid)
        proc_name = proc.name().lower().replace(".exe","")
    except:
        proc = None
        proc_name = "unknown"

    is_known_app = int(any(k in proc_name for k in KNOWN_APPS))

    try:
        is_foreground = int(_get_foreground_pid() == pid)
    except:
        is_foreground = 0

    user_active = int(_seconds_since_last_input() < 60)

    is_night = int(dt.hour < 6 or dt.hour >= 22)

    net_count = 0
    has_network = 0

    if proc:

        try:
            try:
                conns = proc.net_connections()
            except AttributeError:
                conns = proc.connections()

            net_count = len(conns)
            has_network = int(net_count > 0)

        except:
            pass 
    

    return {
        "is_known_app": is_known_app,
        "is_foreground": is_foreground,
        "user_active": user_active,
        "is_night": is_night,
        "has_network_connection": has_network,
        "network_connection_count": net_count,
        "duration_minutes": round(duration_minutes, 3)
    }


# ─────────────────────────────────────────────
# OUTPUT WRITING
# ─────────────────────────────────────────────

def write_json(features_list):

    with open(OUTPUT_JSON,"w") as f:
        json.dump(features_list,f,indent=2)

    print(f"[process_identification] JSON written → {OUTPUT_JSON} ({len(features_list)} process(es))")


def write_csv(features_list):

    if not features_list:
        return

    with open(OUTPUT_CSV,"w",newline="") as f:

        writer = csv.DictWriter(f,fieldnames=FEATURE_KEYS)

        writer.writeheader()

        writer.writerows(features_list)

    print(f"[process_identification] CSV written → {OUTPUT_CSV} ({len(features_list)} process(es))")

def append_to_dataset(features):

    df = pd.DataFrame([features])

    if os.path.exists(DATASET_PATH):

        existing = pd.read_excel(DATASET_PATH)

        updated = pd.concat([existing, df], ignore_index=True)

        updated.to_excel(DATASET_PATH, index=False)

    else:

        df.to_excel(DATASET_PATH, index=False)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def write_dashboard_log(app_name, duration_minutes, status):

    now = datetime.datetime.now()

    row = {
        "application": app_name,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "duration": f"{round(duration_minutes,2)} min",
        "status": status
    }

    df = pd.DataFrame([row])

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

    print("✔ Dashboard log written:", row)

def run(output_format="json", duration_minutes=0, save_to_dataset=True):

    print("[process_identification] Scanning for active webcam processes...")

    webcam_pids = identify_webcam_processes()

    features_list = []

    if not webcam_pids:
        print("[process_identification] No webcam processes found.")

    else:

        for pid,start_time in webcam_pids.items(): 
            features = extract_features(pid, start_time, duration_minutes)

            result = "Normal"
            score = 0

            # Run ML only after webcam session ends
            if duration_minutes > 0:
                result, score = predict(features)
                print(f"ML Prediction: {result} (score={score:.4f})")

            try:
                proc = psutil.Process(pid)
                app_name = proc.name()
            except:
                app_name = "unknown"

            

            if save_to_dataset and duration_minutes > 0 and not features_list:
              append_to_dataset(features)

            # Write row to dashboard CSV
            if duration_minutes > 0:
                write_dashboard_log(app_name, duration_minutes, result)
            

            if result == "Suspicious":

                alert_data = {
                "application": app_name,
                "duration": duration_minutes,
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "score": score
                }
                raise_alert(alert_data, score)
            

            features_list.append(features)

            print(f"PID {pid}: {features}")
            break

    if output_format in ("json","both"):
        write_json(features_list)

    if output_format in ("csv","both"):
        write_csv(features_list)

    return features_list


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--format",
        choices=["json","csv","both"],
        default="json"
    )

    args = parser.parse_args()

    while True:
      run(args.format)
      time.sleep(30)