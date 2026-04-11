import tensorflow as tf
import numpy as np
import json
import os
from datetime import datetime


def convert_to_tflite(model_path, output_path, quantize=True):
    """
    Convert Keras model to TensorFlow Lite format.
    
    Args:
        model_path: Path to .keras model file
        output_path: Path to save .tflite file
        quantize: Whether to apply float16 quantization
    """
    
    print(f"\n{'='*80}")
    print(f"CONVERTING MODEL TO TFLITE")
    print(f"{'='*80}")
    print(f"\n📂 Input model:  {model_path}")
    print(f"📂 Output file:  {output_path}")
    print(f"⚙️  Quantization: {'float16' if quantize else 'none'}")
    
    # Load the Keras model
    print(f"\n📥 Loading Keras model...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"✅ Model loaded successfully!")
    print(f"\n📋 Model Summary:")
    model.summary()
    
    # Convert to TFLite
    print(f"\n🔄 Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print(f"   Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    print(f"\n💾 Saving TFLite model...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Get file sizes
    keras_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    reduction = ((keras_size - tflite_size) / keras_size) * 100
    
    print(f"✅ Conversion complete!")
    print(f"\n📊 Model Sizes:")
    print(f"   Keras model:  {keras_size:.2f} MB")
    print(f"   TFLite model: {tflite_size:.2f} MB")
    print(f"   Reduction:    {reduction:.1f}%")
    
    return tflite_model, tflite_size


def test_tflite_model(tflite_path, test_data_path):
    """
    Test TFLite model to verify it works correctly.
    
    Args:
        tflite_path: Path to .tflite file
        test_data_path: Path to preprocessed test data (.npz)
    """
    
    print(f"\n{'='*80}")
    print(f"TESTING TFLITE MODEL")
    print(f"{'='*80}")
    
    # Load test data
    print(f"\n📂 Loading test data...")
    data = np.load(test_data_path)
    X_test = data['X_test']
    y_test = data['y_test']
    class_names = data.get('class_names', [
        'Non-Emergency', 'Glass Breaking', 'Siren',
        'Baby Crying', 'Car Horn', 'Explosion'
    ])
    
    print(f"✅ Test data loaded: {len(X_test)} samples")
    
    # Load TFLite model
    print(f"\n📥 Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✅ TFLite model loaded!")
    print(f"\n📋 Model Details:")
    print(f"   Input shape:  {input_details[0]['shape']}")
    print(f"   Input dtype:  {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")
    
    # Test predictions
    print(f"\n🧪 Running test predictions...")
    predictions = []
    
    for i, sample in enumerate(X_test):
        # Prepare input
        input_data = np.expand_dims(sample, axis=0).astype(input_details[0]['dtype'])
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data[0])
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i+1}/{len(X_test)} samples...")
    
    predictions = np.array(predictions)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test) * 100
    
    print(f"\n📊 TFLite Model Performance:")
    print(f"   Test Accuracy: {accuracy:.2f}%")
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    for i, name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == i) * 100
            print(f"   {name:20s}: {class_acc:6.2f}%")
    
    # Show sample predictions
    print(f"\n🔍 Sample Predictions:")
    for i in range(min(10, len(X_test))):
        true_label = class_names[y_test[i]]
        pred_label = class_names[y_pred[i]]
        confidence = predictions[i][y_pred[i]] * 100
        symbol = "✅" if y_test[i] == y_pred[i] else "❌"
        print(f"   {symbol} True: {true_label:20s} | Pred: {pred_label:20s} ({confidence:5.1f}%)")
    
    return accuracy


def create_model_metadata(tflite_path, keras_model_path, training_info_path):
    """
    Create metadata JSON file for the TFLite model.
    """
    
    print(f"\n📝 Creating model metadata...")
    
    # Load training info if available
    training_info = {}
    if os.path.exists(training_info_path):
        with open(training_info_path, 'r') as f:
            training_info = json.load(f)
    
    # Get file size
    tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    
    metadata = {
        'model_info': {
            'name': 'Emergency Sound Detection Model',
            'version': '2.0',
            'type': '6-class multi-class classification',
            'framework': 'TensorFlow Lite',
            'quantization': 'float16',
            'size_mb': round(tflite_size, 2),
            'creation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'input': {
            'shape': [1, 40, 216, 1],
            'dtype': 'float32',
            'description': 'MFCC features extracted from 5-second audio clips'
        },
        'output': {
            'shape': [1, 6],
            'dtype': 'float32',
            'description': 'Probability distribution over 6 classes'
        },
        'classes': {
            '0': 'Non-Emergency',
            '1': 'Glass Breaking',
            '2': 'Siren',
            '3': 'Baby Crying',
            '4': 'Car Horn',
            '5': 'Explosion'
        },
        'audio_requirements': {
            'sample_rate': 22050,
            'duration_seconds': 5,
            'num_samples': 110250,
            'format': 'PCM 16-bit mono'
        },
        'feature_extraction': {
            'method': 'Android-matching MFCC',
            'n_mfcc': 40,
            'n_fft': 512,
            'hop_length': 256,
            'num_frames': 216
        },
        'performance': training_info.get('test_accuracy', 'N/A'),
        'usage': {
            'confidence_threshold': 0.7,
            'emergency_classes': [1, 2, 3, 4, 5],
            'non_emergency_class': 0
        }
    }
    
    metadata_path = tflite_path.replace('.tflite', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {metadata_path}")
    
    return metadata_path


def main():
    """Main conversion pipeline"""
    
    print("="*80)
    print("KERAS TO TFLITE CONVERSION - 6-CLASS EMERGENCY SOUND DETECTION")
    print("="*80)
    
    # Paths
    keras_model_path = 'models/best_model_multiclass.keras'
    tflite_output_path = 'models/emergency_sound_model_multiclass_quantized.tflite'
    test_data_path = 'data/processed/preprocessed_data_multiclass.npz'
    training_info_path = 'models/training_info_multiclass.json'
    
    # Check if Keras model exists
    if not os.path.exists(keras_model_path):
        print(f"\n❌ Error: Keras model not found at {keras_model_path}")
        print(f"   Please run train_model_multiclass.py first!")
        return
    
    # Convert to TFLite
    tflite_model, tflite_size = convert_to_tflite(
        keras_model_path,
        tflite_output_path,
        quantize=True
    )
    
    # Test TFLite model
    if os.path.exists(test_data_path):
        test_accuracy = test_tflite_model(tflite_output_path, test_data_path)
    else:
        print(f"\n⚠️  Warning: Test data not found, skipping accuracy test")
        test_accuracy = None
    
    # Create metadata
    metadata_path = create_model_metadata(
        tflite_output_path,
        keras_model_path,
        training_info_path
    )
    
    # Copy class_names.json for Android
    class_names_src = 'models/class_names.json'
    class_names_dst = 'models/class_names_android.json'
    if os.path.exists(class_names_src):
        import shutil
        shutil.copy(class_names_src, class_names_dst)
        print(f"\n✅ Class names copied for Android: {class_names_dst}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"✅ CONVERSION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Generated Files:")
    print(f"   1. {tflite_output_path}")
    print(f"   2. {metadata_path}")
    print(f"   3. {class_names_dst}")
    
    print(f"\n📊 Summary:")
    print(f"   Model size:      {tflite_size:.2f} MB")
    if test_accuracy:
        print(f"   Test accuracy:   {test_accuracy:.2f}%")
    
    print(f"\n📱 Next Steps for Android:")
    print(f"   1. Copy {tflite_output_path}")
    print(f"      → app/src/main/assets/model.tflite")
    print(f"   2. Copy models/class_names.json")
    print(f"      → app/src/main/assets/class_names.json")
    print(f"   3. Rebuild Android app")
    print(f"   4. Test with real emergency sounds!")


if __name__ == "__main__":
    main()