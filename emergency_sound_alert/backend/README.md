# Backend - Emergency Sound Alert System

## Overview

This backend processes audio data and trains a machine learning model to classify emergency and non-emergency sounds.

---

## Folder Structure

```
backend/
├── data/
│   ├── raw/            # Original audio datasets
│   └── processed/      # Preprocessed features (.npz, metadata)
├── models/             # Trained models and evaluation results
├── src/                # Scripts for preprocessing, training, testing
```

---

## Features

* Audio preprocessing and feature extraction
* Multi-class sound classification (siren, explosion, etc.)
* Model training and evaluation
* Conversion to TFLite for mobile use

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess data

```bash
python src/preprocess_audio_multiclass.py
```

### 3. Train model

```bash
python src/train_model_multiclass.py
```

### 4. Test model

```bash
python src/test_predictions_multiclass.py
```

---

## Dataset

* Includes emergency (siren, explosion, glass breaking, etc.) and non-emergency sounds
* Sources:

  * ESC-50 dataset
  * Custom collected audio samples
* Dataset is **not included** in this repository

---

## Notes

* Large files like `.npz`, model weights (`.h5`), and logs are excluded
* They will be generated after running the scripts

---

## Tech Stack

* Python
* NumPy, Librosa
* TensorFlow / Keras

---

## Output

* Trained model (`.h5` / `.tflite`)
* Evaluation metrics and plots
* Prediction scripts for testing
