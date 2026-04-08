import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:

    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")

        if self.face_cascade.empty():
            raise RuntimeError("Face cascade not loaded")

    def detect_face(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        detected = False
        bbox = None

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            bbox = faces[0]
            detected = True

            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return frame, bbox, detected


class DrownessDetectionCamera:

    def __init__(self, camera_index=0):

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")

        self.face_detector = FaceDetector()

        self.last_face_region = None

    def capture_and_detect(self):

        ret, frame = self.cap.read()

        if not ret:
            return None, False, None

        frame, bbox, detected = self.face_detector.detect_face(frame)

        eyes_count = 0

        if detected:
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]
            self.last_face_region = face_roi

            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes = self.face_detector.eye_cascade.detectMultiScale(gray, 1.1, 4)

            for ex, ey, ew, eh in eyes:
                cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

            eyes_count = len(eyes)
            landmarks = {"eye_count": eyes_count}

        else:
            self.last_face_region = None
            landmarks = None

        return frame, detected, landmarks

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()