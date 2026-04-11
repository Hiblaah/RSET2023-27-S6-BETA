import os
import numpy as np
import librosa
import opensmile
import nolds
from scipy.stats import entropy
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
import joblib

###############################################
# LOAD TRAINED MODEL
###############################################

model = joblib.load("voice_knn.pkl")

###############################################
# OPENSMILE FEATURE EXTRACTOR
###############################################

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

###############################################
# NONLINEAR FEATURES
###############################################

def compute_rpde(signal):
    hist, _ = np.histogram(signal, bins=100)
    return entropy(hist)

def compute_dfa(signal):
    return nolds.dfa(signal)

def compute_d2(signal):
    try:
        return nolds.corr_dim(signal, emb_dim=10)
    except:
        return 0

def compute_ppe(signal):
    phase = np.angle(np.fft.fft(signal))
    hist, _ = np.histogram(phase, bins=50)
    return entropy(hist)

###############################################
# RQA FEATURES (spread1 / spread2)
###############################################

def compute_rqa(signal):

    ts = TimeSeries(signal, embedding_dimension=2, time_delay=2)

    settings = Settings(
        ts,
        analysis_type=Classic,
        neighbourhood=FixedRadius(0.5),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1,
    )

    computation = RQAComputation.create(settings)
    result = computation.run()

    spread1 = result.determinism
    spread2 = result.laminarity

    return spread1, spread2

###############################################
# FEATURE EXTRACTION
###############################################

def extract_features(audio_path):

    signal, sr = librosa.load(audio_path, sr=None)

    smile_features = smile.process_file(audio_path)
    smile_features = smile_features.values.flatten()

    rpde = compute_rpde(signal)
    dfa = compute_dfa(signal)
    d2 = compute_d2(signal)
    ppe = compute_ppe(signal)

    spread1, spread2 = compute_rqa(signal)

    features = np.concatenate([
        smile_features,
        [rpde, dfa, spread1, spread2, d2, ppe]
    ])

    return features.reshape(1, -1)

###############################################
# FOLDER WITH VOICE FILES
###############################################

audio_folder = r"C:\Users\fbash\OneDrive\Desktop\RSET\thrid year\s6\mini project\MAIN PROJECT\project\voice\voice_recordings"

###############################################
# PREDICT EACH FILE
###############################################

for file in os.listdir(audio_folder):

    if file.endswith(".wav"):

        path = os.path.join(audio_folder, file)

        try:
            features = extract_features(path)

            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]

            label = "Parkinson's" if prediction == 1 else "Healthy"

            print(f"{file} -> {label} ({probability*100:.2f}% confidence)")

        except Exception as e:
            print(f"{file} -> Error processing file")