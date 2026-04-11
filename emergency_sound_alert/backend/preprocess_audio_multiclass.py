import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import pandas as pd
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# Android-matching constants (DO NOT MODIFY - must match Android inference)
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE = 22050
DURATION    = 5
N_MFCC      = 40
N_FFT       = 512
HOP_LENGTH  = 256
NUM_FRAMES  = 216
TARGET_LEN  = SAMPLE_RATE * DURATION


def extract_android_features(audio: np.ndarray) -> np.ndarray:
    """
    Replicates MFCCExtractor.extractMFCC() from Android exactly.
    
    DO NOT MODIFY THIS FUNCTION - it must match Android implementation.
    
    Input : float32 audio array, already padded/trimmed to TARGET_LEN samples
    Output: ndarray shape (N_MFCC, NUM_FRAMES, 1) — matches model input (40, 216, 1)
    """

    # ── 1. Normalize amplitude (mirrors Android's maxVal normalisation) ────────
    max_val = np.max(np.abs(audio))
    if max_val > 0:#to check if its not silent audio, if its silent then we skip normalisation to avoid division by zero
        audio = audio / (max_val + 0.01)#normalizes amplitude (0.01-prevent large scaling)

    # ── 2. Frame-level spectral energy ────────────────────────────────────────
    num_frames = min(NUM_FRAMES, (len(audio) - N_FFT) // HOP_LENGTH) # calculates how many frames exracted
    mfcc = np.full((N_MFCC, NUM_FRAMES), -23.0, dtype=np.float32)  # padded default, CREATES MFCC MATRIX(40x216)

    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N_FFT) / (N_FFT - 1)) 

    for frame in range(num_frames):
        start = frame * HOP_LENGTH
        frame_data = audio[start:start + N_FFT] * hamming  # windowed frame

        for band in range(N_MFCC):
            # Quadratic mel-like band boundaries — exact match to Android
            freq_start = int((band       / N_MFCC) ** 2 * N_FFT / 2)
            freq_end   = int(((band + 1) / N_MFCC) ** 2 * N_FFT / 2)
            freq_end   = min(freq_end, N_FFT)

            energy = np.sum(frame_data[freq_start:freq_end] ** 2)
            mfcc[band, frame] = np.log(energy + 1e-10)  # log compression

    # ── 3. Per-band normalisation (only over valid frames) ─────────────────────
    for band in range(N_MFCC):
        valid = mfcc[band, :num_frames]
        mean  = np.mean(valid)
        std   = np.std(valid)

        if std > 0.01:
            mfcc[band, :num_frames] = (valid - mean) / std
            mfcc[band, num_frames:] = 0.0   # zero-pad after normalise
        else:
            mfcc[band, :] = 0.0

    return mfcc[..., np.newaxis]  # (40, 216, 1)


def load_and_normalize_audio(file_path: str):
    """
    Load audio file and normalize to target length.
    
    Returns normalized audio array or None on error.
    """
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or trim to exact target length
        if len(audio) < TARGET_LEN:
            audio = np.pad(audio, (0, TARGET_LEN - len(audio)))
        else:
            audio = audio[:TARGET_LEN]
            
        return audio
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {e}")
        return None


def augment_audio(audio: np.ndarray, augmentation_factor: int = 1):
    """
    Apply audio augmentations before MFCC extraction.
    
    Args:
        audio: Input audio array (TARGET_LEN samples)
        augmentation_factor: How many augmented versions to create
        
    Returns:
        List of augmented audio arrays (includes original)
    """
    samples = [audio.copy()]  # Always include original
    
    for _ in range(augmentation_factor):
        # Random augmentation selection
        aug_type = np.random.choice(['noise', 'time_shift', 'pitch_shift', 'time_stretch'])
        
        if aug_type == 'noise':
            # Add Gaussian noise
            noise_level = np.random.uniform(0.003, 0.008)
            augmented = audio + np.random.randn(len(audio)) * noise_level
            
        elif aug_type == 'time_shift':
            # Random time shift
            shift = np.random.randint(-SAMPLE_RATE // 2, SAMPLE_RATE // 2)
            augmented = np.roll(audio, shift)
            
        elif aug_type == 'pitch_shift':
            # Random pitch shift
            try:
                n_steps = np.random.choice([-2, -1, 1, 2])
                augmented = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
                # Ensure correct length
                if len(augmented) < TARGET_LEN:
                    augmented = np.pad(augmented, (0, TARGET_LEN - len(augmented)))
                else:
                    augmented = augmented[:TARGET_LEN]
            except Exception:
                augmented = audio.copy()
                
        elif aug_type == 'time_stretch':
            # Random time stretch
            try:
                rate = np.random.uniform(0.9, 1.1)
                stretched = librosa.effects.time_stretch(audio, rate=rate)
                # Ensure correct length
                if len(stretched) < TARGET_LEN:
                    stretched = np.pad(stretched, (0, TARGET_LEN - len(stretched)))
                else:
                    stretched = stretched[:TARGET_LEN]
                augmented = stretched
            except Exception:
                augmented = audio.copy()
        
        samples.append(augmented)
    
    return samples


def load_custom_emergency_dataset(custom_dir):

    class_mapping = {
        'glass_breaking': 1,
        'siren': 2,
        'crying_baby': 3,
        'car_horn': 4,
        'explosion': 5
    }

    dataset = defaultdict(list)

    for class_name, label in class_mapping.items():
        class_dir = os.path.join(custom_dir, class_name)

        if not os.path.exists(class_dir):
            continue

        audio_files = librosa.util.find_files(class_dir, ext=['wav','mp3','flac'])

        dataset[label] = audio_files
        print(f"{class_name}: {len(audio_files)} files")

    return dataset


def load_esc50_non_emergency(esc50_audio_dir, esc50_metadata_path):
    """
    Load non-emergency sounds from ESC-50 dataset.
    
    Returns:
        list: Audio file paths for non-emergency sounds
    """
    print("\n📂 Loading ESC-50 non-emergency sounds...")
    
    df = pd.read_csv(esc50_metadata_path)
    
    # Define non-emergency categories (excluding emergency-like sounds)
    non_emergency_categories = {
        'dog', 'rain', 'sea_waves', 'thunderstorm', 'wind',
        'footsteps', 'breathing', 'coughing', 'sneezing', 'clapping',
        'keyboard_typing', 'door_wood_knock', 'mouse_click',
        'engine', 'train', 'airplane',
        'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick',
        'water_drops', 'crickets', 'insects', 'frog',
        'crow', 'hen', 'rooster', 'cat', 'sheep', 'cow', 'pig',
        'hand_saw', 'chainsaw', 'can_opening',
        'church_bells', 'helicopter',
        'laughing', 'pouring_water', 'toilet_flush',
        'drinking_sipping', 'brushing_teeth', 'snoring'
    }
    
    # Filter for non-emergency sounds
    non_emergency_files = []
    for _, row in df.iterrows():
        if row['category'] in non_emergency_categories:
            filepath = os.path.join(esc50_audio_dir, row['filename'])
            if os.path.exists(filepath):
                non_emergency_files.append(filepath)
    
    print(f"   Found {len(non_emergency_files)} non-emergency samples from ESC-50")
    
    return non_emergency_files


def create_balanced_dataset(custom_dataset, esc50_files, target_samples_per_class=None):

    # Determine target samples per class
    all_counts = {0: len(esc50_files)}
    for label, files in custom_dataset.items():
        all_counts[label] = len(files)

    if target_samples_per_class is None:
        target_samples_per_class = max(all_counts.values())
        print(f"\n🎯 Target samples per class: {target_samples_per_class}")

    # Safe augmentation calculation
    augmentation_plan = {}

    for label, count in all_counts.items():

        # Prevent division by zero
        if count == 0:
            print(f"⚠️ Class {label} has 0 samples — skipping augmentation")
            augmentation_plan[label] = 0
            continue

        if count < target_samples_per_class:
            aug_factor = max(0, (target_samples_per_class // count) - 1)
            augmentation_plan[label] = aug_factor
        else:
            augmentation_plan[label] = 0

    print(f"\n📊 Augmentation Plan:")

    class_names = [
        'Non-Emergency',
        'Glass Breaking',
        'Siren',
        'Baby Crying',
        'Car Horn',
        'Explosion'
    ]

    for label in range(6):
        if label in all_counts:
            current = all_counts[label]
            aug_factor = augmentation_plan.get(label, 0)
            estimated = current * (aug_factor + 1)

            print(
                f"   {class_names[label]:20s}: "
                f"{current:4d} → {estimated:4d} samples "
                f"(aug_factor={aug_factor})"
            )

    features = []
    labels = []
    final_counts = defaultdict(int)

    print(f"\n🔄 Processing and augmenting audio files...")

    # Process Non-Emergency
    print(f"\n   Processing class 0 (Non-Emergency)...")

    for filepath in tqdm(esc50_files, desc="   Non-Emergency"):

        audio = load_and_normalize_audio(filepath)
        if audio is None:
            continue

        aug_factor = augmentation_plan.get(0, 0)
        augmented_samples = augment_audio(audio, aug_factor)

        for aug_audio in augmented_samples:
            feat = extract_android_features(aug_audio)

            features.append(feat)
            labels.append(0)
            final_counts[0] += 1

    # Process emergency classes
    for label, filepaths in custom_dataset.items():

        if len(filepaths) == 0:
            print(f"⚠️ Skipping class {label} (no files)")
            continue

        class_name = class_names[label]
        print(f"\n   Processing class {label} ({class_name})...")

        for filepath in tqdm(filepaths, desc=f"   {class_name}"):

            audio = load_and_normalize_audio(filepath)
            if audio is None:
                continue

            aug_factor = augmentation_plan.get(label, 0)
            augmented_samples = augment_audio(audio, aug_factor)

            for aug_audio in augmented_samples:
                feat = extract_android_features(aug_audio)

                features.append(feat)
                labels.append(label)
                final_counts[label] += 1

    features = np.array(features)
    labels = np.array(labels)

    return features, labels, dict(final_counts)

def main():
    """Main preprocessing pipeline"""
    
    print("=" * 80)
    print("MULTI-CLASS EMERGENCY SOUND DETECTION - DATA PREPROCESSING")
    print("=" * 80)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Configuration
    # ══════════════════════════════════════════════════════════════════════════
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # ESC-50 paths
    esc50_audio_dir     = os.path.join(base_dir, "data", "raw", "ESC-50", "audio")
    esc50_metadata_path = os.path.join(base_dir, "data", "raw", "ESC-50", "meta", "esc50.csv")
    
    # Custom emergency dataset path
    custom_emergency_dir = os.path.join(base_dir, "data", "raw", "custom", "emergency")
    
    # Output path
    output_path = os.path.join(base_dir, "data", "processed", "preprocessed_data_multiclass.npz")
    
    print(f"\n📂 Dataset Paths:")
    print(f"   ESC-50 audio:        {esc50_audio_dir}")
    print(f"   ESC-50 metadata:     {esc50_metadata_path}")
    print(f"   Custom emergency:    {custom_emergency_dir}")
    print(f"   Output:              {output_path}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Validate paths
    # ══════════════════════════════════════════════════════════════════════════
    if not os.path.exists(esc50_audio_dir):
        raise FileNotFoundError(f"❌ ESC-50 audio directory not found: {esc50_audio_dir}")
    if not os.path.exists(esc50_metadata_path):
        raise FileNotFoundError(f"❌ ESC-50 metadata not found: {esc50_metadata_path}")
    if not os.path.exists(custom_emergency_dir):
        raise FileNotFoundError(f"❌ Custom emergency directory not found: {custom_emergency_dir}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Load datasets
    # ══════════════════════════════════════════════════════════════════════════
    
    # Load custom emergency sounds (classes 1-5)
    custom_dataset = load_custom_emergency_dataset(custom_emergency_dir)
    
    # Load ESC-50 non-emergency sounds (class 0)
    esc50_files = load_esc50_non_emergency(esc50_audio_dir, esc50_metadata_path)
    
    # Check if we have data for all classes
    print(f"\n✅ Dataset Loading Summary:")
    print(f"   Non-Emergency (ESC-50):  {len(esc50_files)} files")
    for label in range(1, 6):
        count = len(custom_dataset.get(label, []))
        print(f"   Emergency Class {label}:         {count} files")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Create balanced dataset with augmentation
    # ══════════════════════════════════════════════════════════════════════════
    
    features, labels, class_counts = create_balanced_dataset(
        custom_dataset,
        esc50_files,
        target_samples_per_class=None  # Auto-calculate from largest class
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # Dataset Summary
    # ══════════════════════════════════════════════════════════════════════════
    
    CLASS_NAMES = [
        'Non-Emergency',
        'Glass Breaking',
        'Siren',
        'Baby Crying',
        'Car Horn',
        'Explosion'
    ]
    
    print(f"\n{'='*80}")
    print(f"FINAL DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples:  {len(features)}")
    print(f"Feature shape:  {features.shape}")
    print(f"\n📊 Class Distribution (After Balancing):")
    for cls, name in enumerate(CLASS_NAMES):
        count = class_counts.get(cls, 0)
        percentage = (count / len(features)) * 100
        print(f"   {cls}: {name:20s} - {count:5d} samples ({percentage:5.1f}%)")
    
    # Check for missing classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 6:
        print(f"\n⚠️  WARNING: Only {len(unique_labels)} classes found!")
        print(f"   Expected 6 classes (0-5)")
        print(f"   Missing classes: {set(range(6)) - set(unique_labels)}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Train / Validation / Test Split (Stratified)
    # ══════════════════════════════════════════════════════════════════════════
    
    print(f"\n🔀 Splitting dataset (70% train / 15% val / 15% test)...")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )
    
    print(f"   Training:   {len(X_train):5d} samples")
    print(f"   Validation: {len(X_val):5d} samples")
    print(f"   Testing:    {len(X_test):5d} samples")
    
    # Verify stratification
    print(f"\n📊 Class Distribution per Split:")
    for cls, name in enumerate(CLASS_NAMES):
        train_count = np.sum(y_train == cls)
        val_count = np.sum(y_val == cls)
        test_count = np.sum(y_test == cls)
        print(f"   {name:20s}: Train={train_count:4d}, Val={val_count:4d}, Test={test_count:4d}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Save Dataset
    # ══════════════════════════════════════════════════════════════════════════
    
    print(f"\n💾 Saving dataset...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.savez_compressed(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_names=np.array(CLASS_NAMES)
    )
    
    print(f"✅ Dataset saved to: {output_path}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Save Metadata
    # ══════════════════════════════════════════════════════════════════════════
    
    metadata = {
        'preprocessing_version': '2.0',
        'dataset_sources': {
            'non_emergency': 'ESC-50',
            'emergency': 'Custom Dataset'
        },
        'audio_parameters': {
            'sample_rate': SAMPLE_RATE,
            'duration': DURATION,
            'target_length': TARGET_LEN
        },
        'feature_extraction': {
            'method': 'android_matching_fast_mfcc',
            'n_mfcc': N_MFCC,
            'n_fft': N_FFT,
            'hop_length': HOP_LENGTH,
            'num_frames': NUM_FRAMES,
            'output_shape': list(features.shape[1:])
        },
        'classes': {
            'num_classes': 6,
            'class_names': CLASS_NAMES,
            'class_mapping': {
                '0': 'non_emergency',
                '1': 'glass_breaking',
                '2': 'siren',
                '3': 'baby_crying',
                '4': 'car_horn',
                '5': 'explosion'
            }
        },
        'dataset_statistics': {
            'total_samples': len(features),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'class_distribution': class_counts,
            'augmentation_applied': True
        },
        'split_ratios': {
            'train': 0.7,
            'validation': 0.15,
            'test': 0.15
        }
    }
    
    metadata_path = output_path.replace('.npz', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {metadata_path}")
    
    print(f"\n{'='*80}")
    print(f"✅ PREPROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📍 Next step: python src/train_model_multiclass.py")


if __name__ == "__main__":
    main()