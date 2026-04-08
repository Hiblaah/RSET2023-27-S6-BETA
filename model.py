import cv2
import numpy as np
from tensorflow.keras.models import load_model


class TeachableMachineModel:

    def __init__(self, model_path="keras_model.h5", labels_path="labels.txt"):

        self.model = load_model(model_path, compile=False)

        with open(labels_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def predict(self, face_roi):

        if face_roi is None:
            return None, 0

        img = cv2.resize(face_roi, (224, 224))
        img = np.asarray(img, dtype=np.float32)
        img = (img / 127.5) - 1
        img = np.expand_dims(img, axis=0)

        prediction = self.model.predict(img, verbose=0)

        index = np.argmax(prediction)
        confidence = float(prediction[0][index])

        # Remove index number (0 or 1) from label
        label = self.labels[index].split(" ", 1)[1]

        return label, confidence