"""
popup_alert.py — Desktop popup alert system for webcam-abuse-detector.

Drop this file into your project root or mlmodel/ folder.
Call raise_popup_alert() from ml_detector.py or process_detector.py
whenever a suspicious event is detected.

Dependencies:
    pip install plyer tkinter  (tkinter is built-in with Python)

Cross-platform support:
    Windows  → tkinter MessageBox (always works, no extra install)
    macOS    → osascript notification + tkinter fallback
    Linux    → libnotify (notify-send) + tkinter fallback
"""

import threading
import platform
import subprocess
import os
import datetime
import logging
from pathlib import Path
import json
import csv
DASHBOARD_LOG = Path(__file__).parent.parent / "alerts" / "dashboard_alerts.json"

# ── Log setup ────────────────────────────────────────────────────────────────
LOG_PATH = Path(__file__).parent / "alerts" / "alerts_log.txt"
LOG_PATH.parent.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ── Severity thresholds ───────────────────────────────────────────────────────
def get_severity(score: float) -> str:
    if score < -0.20:
        return "SUSPICIOUS"
    elif score < -0.10:
        return "SUSPICIOUS"
    elif score < 0.0:
        return "SUSPICIOUS"
    return "LOW"


def get_severity_color(severity: str) -> str:
    return {
        "SUSPICIOUS": "#c0392b",
    }.get(severity, "#2c3e50")


# ── Core popup function ───────────────────────────────────────────────────────
def raise_popup_alert(
    score: float,
    label: str = "Suspicious",
    reason: str = "",
    extra_info: dict = None,
    blocking: bool = False,
):
    if label != "Suspicious":
        return

    severity = get_severity(score)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    extra_str = ""
    if extra_info:
        extra_str = "\n".join(f"  {k}: {v}" for k, v in extra_info.items())

    log_msg = (
        f"[{severity}] Score={score:.4f} | {reason} | {extra_str.replace(chr(10), ' | ')}"
    )

    logging.warning(log_msg)

    # ── Save suspicious alert to webcam_logs.csv for dashboard ──
    try:
        csv_path = Path(__file__).parent.parent / "logs" / "webcam_logs.csv"
        csv_path.parent.mkdir(exist_ok=True)

        row = {
            "process_name": "ML Detector",
            "date": timestamp.split()[0],
            "time": timestamp.split()[1],
            "duration": "0.00 min",
            "status": severity
        }

        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["process_name", "date", "time", "duration", "status"]
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

        print("Alert written to CSV:", row)

    except Exception as e:
        print("[CSVLog Error]", e)
    # ── Save alert to dashboard history ──
    try:
        alert_entry = {
            "timestamp": timestamp,
            "severity": severity,
            "score": round(score, 4),
            "reason": reason,
        }

        history = []

        if DASHBOARD_LOG.exists():
            with open(DASHBOARD_LOG, "r") as f:
                history = json.load(f)

        history.insert(0, alert_entry)
        history = history[:100]

        with open(DASHBOARD_LOG, "w") as f:
            json.dump(history, f, indent=2)

    except Exception as e:
        print("[DashboardLog] Error:", e)

    print(f"🚨 [{timestamp}] {severity} ALERT → {reason} | Score: {score:.4f}")

    t = threading.Thread(
        target=_show_popup,
        args=(severity, score, reason, extra_str, timestamp),
        daemon=True,
    )
    t.start()

    if blocking:
        t.join()


# ── Platform dispatcher ───────────────────────────────────────────────────────
def _show_popup(severity, score, reason, extra_str, timestamp):
    system = platform.system()
    try:
        if system == "Windows":
            _popup_windows(severity, score, reason, extra_str, timestamp)
        elif system == "Darwin":
            _popup_macos(severity, score, reason, extra_str, timestamp)
        else:
            _popup_linux(severity, score, reason, extra_str, timestamp)
    except Exception as e:
        print(f"[PopupAlert] Primary popup failed ({e}), using tkinter fallback...")
        _popup_tkinter(severity, score, reason, extra_str, timestamp)


# ── Windows popup ─────────────────────────────────────────────────────────────
def _popup_windows(severity, score, reason, extra_str, timestamp):
    _popup_tkinter(severity, score, reason, extra_str, timestamp)


# ── macOS popup ───────────────────────────────────────────────────────────────
def _popup_macos(severity, score, reason, extra_str, timestamp):
    msg = f"{severity} | Score: {score:.4f} | {reason}"

    subprocess.run(
        [
            "osascript",
            "-e",
            f'display notification "{msg}" with title "⚠️ Webcam Abuse Detector" '
            f'subtitle "{timestamp}" sound name "Basso"',
        ],
        timeout=5,
    )

    _popup_tkinter(severity, score, reason, extra_str, timestamp)


# ── Linux popup ───────────────────────────────────────────────────────────────
def _popup_linux(severity, score, reason, extra_str, timestamp):

    msg = f"{severity} | Score: {score:.4f}\n{reason}"

    try:
        subprocess.run(
            [
                "notify-send",
                "-u",
                "critical",
                "-t",
                "8000",
                "⚠️ Webcam Abuse Detector",
                msg,
            ],
            timeout=5,
        )
    except FileNotFoundError:
        pass

    _popup_tkinter(severity, score, reason, extra_str, timestamp)


# ── Tkinter popup ─────────────────────────────────────────────────────────────
def _popup_tkinter(severity, score, reason, extra_str, timestamp):

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        win = tk.Toplevel(root)

        win.title("⚠️ Behavioral Anomaly Detected")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.attributes("-alpha", 0.97)

        BG = "#1a1a2e"
        CARD_BG = "#16213e"
        FG = "#e0e0e0"
        ACCENT = get_severity_color(severity)
        BTN_BG = "#0f3460"
        BTN_FG = "#e0e0e0"

        win.configure(bg=BG)
        win.geometry("420x320")

        bar = tk.Frame(win, bg=ACCENT, height=6)
        bar.pack(fill="x")

        header = tk.Frame(win, bg=CARD_BG, pady=12)
        header.pack(fill="x")

        icon_label = tk.Label(
            header, text="🚨", font=("Segoe UI Emoji", 28), bg=CARD_BG, fg=ACCENT
        )
        icon_label.pack(side="left", padx=(18, 0))

        title_frame = tk.Frame(header, bg=CARD_BG)
        title_frame.pack(side="left", padx=12)

        tk.Label(
            title_frame,
            text=f"{severity} ANOMALY DETECTED",
            font=("Segoe UI", 13, "bold"),
            bg=CARD_BG,
            fg=ACCENT,
        ).pack(anchor="w")

        tk.Label(
            title_frame,
            text=f"Webcam Abuse Detector  ·  {timestamp}",
            font=("Segoe UI", 8),
            bg=CARD_BG,
            fg="#888",
        ).pack(anchor="w")

        body = tk.Frame(win, bg=BG, padx=18, pady=10)
        body.pack(fill="both", expand=True)

        def row(label, value, value_color=FG):
            f = tk.Frame(body, bg=BG)
            f.pack(fill="x", pady=3)

            tk.Label(
                f,
                text=label,
                width=20,
                anchor="w",
                font=("Segoe UI", 9),
                bg=BG,
                fg="#888",
            ).pack(side="left")

            tk.Label(
                f,
                text=value,
                anchor="w",
                font=("Segoe UI", 9, "bold"),
                bg=BG,
                fg=value_color,
            ).pack(side="left")

        score_color = ACCENT if score < 0 else "#27ae60"

        row("Severity:", severity, ACCENT)
        row("Anomaly Score:", f"{score:.4f}", score_color)
        row("Reason:", reason or "—")

        btn_frame = tk.Frame(win, bg=BG, pady=10)
        btn_frame.pack(fill="x", padx=18)

        def dismiss():
            win.destroy()
            root.destroy()

        tk.Button(
            btn_frame,
            text="Dismiss",
            command=dismiss,
            bg=BTN_BG,
            fg=BTN_FG,
            relief="flat",
            font=("Segoe UI", 9, "bold"),
            padx=18,
            pady=6,
        ).pack(side="right", padx=(6, 0))

        win.after(15000, lambda: (win.destroy(), root.destroy()))

        win.update_idletasks()

        w, h = win.winfo_width(), win.winfo_height()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()

        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        win.deiconify()
        root.mainloop()

    except Exception as e:
        print(f"[PopupAlert] tkinter popup failed: {e}")


# ── Integration helpers ───────────────────────────────────────────────────────
def alert_if_suspicious(prediction: int, score: float, features: dict = None):

    if prediction != -1:
        return

    severity = get_severity(score)
    reasons = []
    extra = {}

    if features:

        if features.get("suspicious_process_detected"):
            reasons.append("Suspicious process detected")

        if features.get("unknown_process_count", 0) > 20:
            reasons.append(
                f"High unknown process count ({features['unknown_process_count']})"
            )

        if features.get("high_cpu_process_count", 0) > 3:
            reasons.append(
                f"High CPU usage ({features['high_cpu_process_count']} processes)"
            )

        if features.get("is_night") and not features.get("user_active"):
            reasons.append("Night-time activity with inactive user")

        extra = {
            k: v
            for k, v in features.items()
            if k not in ("process_names_sample",) and v is not None
        }

    reason = "; ".join(reasons) if reasons else "Anomalous behavioral pattern"

    raise_popup_alert(
    score=score,
    label="Suspicious",
    reason=reason,
    extra_info=extra,
    blocking=True
)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("Testing CRITICAL alert popup...")

    raise_popup_alert(
        score=-0.25,
        label="Suspicious",
        reason="Unknown process spike detected",
        extra_info={
            "process": "unknown_miner.exe",
            "cpu_usage": "94%",
            "unknown_processes": 28,
            "session_duration": "42 min",
            "network_conns": 7,
        },
        blocking=True,
    )

    print("✅ Popup test complete. Check alerts/alerts_log.txt for the log entry.")