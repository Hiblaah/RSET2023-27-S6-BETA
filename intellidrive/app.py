import cv2
import time
import pygame
from camera import DrownessDetectionCamera
from model import TeachableMachineModel


class DriverDrowsinessSystem:

    def __init__(self):

        self.camera = DrownessDetectionCamera(0)

        self.tm = TeachableMachineModel(
            "keras_model.h5",
            "labels.txt"
        )

        self.blink_count = 0
        self.prev_eye = "OPEN"

        self.eye_closed_start = None
        self.eye_closed_duration = 0

        # -------- CALIBRATION SETTINGS --------
        self.calibration_time = 30
        self.start_calibration()

        # -------- AUDIO --------
        pygame.mixer.init()
        pygame.mixer.music.load("alarm.wav")
        self.alarm_playing = False

    # ---------------- CALIBRATION ----------------
    def start_calibration(self):
        self.calibration_start = time.time()
        self.calibration_mode = True
        self.blink_durations = []
        self.blink_count = 0
        self.personalized_threshold = 3
        print("Calibration Started...")

    # ---------------- ALARM ----------------
    def start_alarm(self):
        if not self.alarm_playing:
            pygame.mixer.music.play(-1)
            self.alarm_playing = True

    def stop_alarm(self):
        if self.alarm_playing:
            pygame.mixer.music.stop()
            self.alarm_playing = False

    # ---------------- UI ----------------
    def draw_ui(self, frame, face, eye, blink, drowsy,
                conf, closed_time, blink_rate):

        cv2.rectangle(frame, (0, 0), (550, 270), (40, 40, 40), -1)

        cv2.putText(frame, f"Face: {face}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Eye: {eye}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"Blinks: {blink}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"Blink Rate: {blink_rate:.2f}/min", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Closed Time: {closed_time:.2f}s", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Status: {drowsy}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if drowsy == "DROWSY" else (0, 255, 0), 2)

        cv2.putText(frame,
                    f"Mode: {'CALIBRATION' if self.calibration_mode else 'MONITORING'}",
                    (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255) if self.calibration_mode else (0, 255, 0),
                    2)

        cv2.putText(frame,
                    f"Threshold: {self.personalized_threshold:.2f}s",
                    (250, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        cv2.putText(frame,
                    "Press R to Recalibrate",
                    (250, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1)

        if drowsy == "DROWSY":
            cv2.putText(frame, "WAKE UP!", (300, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ---------------- MAIN LOOP ----------------
    def start(self):

        while True:

            frame, face_detected, landmarks = self.camera.capture_and_detect()

            if frame is None:
                break

            face_roi = self.camera.last_face_region

            # Eye state
            if landmarks and landmarks["eye_count"] >= 2:
                eye_state = "OPEN"
            else:
                eye_state = "CLOSED"

            current_time = time.time()

            # Blink detection
            if self.prev_eye == "CLOSED" and eye_state == "OPEN":
                self.blink_count += 1

            self.prev_eye = eye_state

            # Eye closure timing
            if eye_state == "CLOSED":
                if self.eye_closed_start is None:
                    self.eye_closed_start = current_time

                self.eye_closed_duration = current_time - self.eye_closed_start

            else:
                if self.eye_closed_start is not None:
                    duration = current_time - self.eye_closed_start

                    if self.calibration_mode:
                        self.blink_durations.append(duration)

                self.eye_closed_start = None
                self.eye_closed_duration = 0

            # -------- CALIBRATION CHECK --------
            if self.calibration_mode:
                elapsed = current_time - self.calibration_start

                if elapsed >= self.calibration_time:
                    if len(self.blink_durations) > 0:
                        avg_blink = sum(self.blink_durations) / len(self.blink_durations)
                        self.personalized_threshold = 6 * avg_blink

                        print("Calibration Complete")
                        print("Avg Blink:", avg_blink)
                        print("Threshold:", self.personalized_threshold)

                    self.calibration_mode = False

            # Blink rate calculation
            minutes_elapsed = (current_time - self.calibration_start) / 60
            blink_rate = self.blink_count / minutes_elapsed if minutes_elapsed > 0 else 0

            # Model prediction
            if face_roi is not None:
                label, confidence = self.tm.predict(face_roi)
            else:
                label, confidence = None, 0

            drowsy_status = "AWAKE"

            if not self.calibration_mode and \
               self.eye_closed_duration >= self.personalized_threshold:
                drowsy_status = "DROWSY"

            elif label == "DROWSY":
                drowsy_status = "DROWSY"

            if drowsy_status == "DROWSY" and not self.calibration_mode:
                self.start_alarm()
            else:
                self.stop_alarm()

            self.draw_ui(frame,
                         "YES" if face_detected else "NO",
                         eye_state,
                         self.blink_count,
                         drowsy_status,
                         confidence,
                         self.eye_closed_duration,
                         blink_rate)

            cv2.imshow("Driver Drowsiness Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Press R to recalibrate
            if key == ord("r"):
                self.start_calibration()

            if key == ord("q"):
                break

        self.stop_alarm()
        self.camera.release()


if __name__ == "__main__":
    DriverDrowsinessSystem().start()
