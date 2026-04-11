import numpy as np
import tensorflow as tf
import json

def test_model_predictions():
    """Test model predictions on test set"""
    
    print("="*60)
    print("TESTING MULTI-CLASS MODEL PREDICTIONS")
    print("="*60)
    
    # Load model
    print("\n📂 Loading model...")
    model = tf.keras.models.load_model('models/best_model_multiclass.keras')
    
    # Load class names
    with open('models/class_names.json', 'r') as f:
        class_names = json.load(f)
    
    # Load test data
    print("📂 Loading test data...")
    data = np.load('data/processed/preprocessed_data_multiclass.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Make predictions
    print("🔮 Making predictions...\n")
    predictions = model.predict(X_test)
    
    # Show first 20 predictions
    print("📊 First 20 Test Predictions:\n")
    print(f"{'#':<4} {'True Label':<20} {'Predicted':<20} {'Confidence':<12} {'Status'}")
    print("-" * 70)
    
    correct = 0
    for i in range(min(20, len(X_test))):
        true_idx = y_test[i]
        pred_idx = np.argmax(predictions[i])
        confidence = predictions[i][pred_idx] * 100
        
        true_label = class_names[true_idx]
        pred_label = class_names[pred_idx]
        
        is_correct = true_idx == pred_idx
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        
        print(f"{i+1:<4} {true_label:<20} {pred_label:<20} {confidence:>6.2f}%     {status}")
    
    accuracy = (correct / min(20, len(X_test))) * 100
    print(f"\n📊 Accuracy on first 20 samples: {accuracy:.1f}%")
    
    # Show per-class accuracy
    print(f"\n📊 Per-Class Accuracy on Full Test Set:")
    print("-" * 50)
    
    pred_classes = np.argmax(predictions, axis=1)
    
    for i, name in enumerate(class_names):
        class_mask = y_test == i
        class_total = np.sum(class_mask)
        
        if class_total > 0:
            class_correct = np.sum((pred_classes == i) & class_mask)
            class_accuracy = (class_correct / class_total) * 100
            print(f"{name:<20}: {class_accuracy:>5.1f}% ({class_correct}/{class_total})")
    
    # Overall accuracy
    overall_accuracy = np.mean(pred_classes == y_test) * 100
    print(f"\n{'Overall Accuracy':<20}: {overall_accuracy:>5.1f}%")
    
    # Show confidence distribution
    print(f"\n📊 Prediction Confidence Distribution:")
    confidence_values = np.max(predictions, axis=1) * 100
    
    print(f"   Average confidence: {np.mean(confidence_values):.1f}%")
    print(f"   Min confidence: {np.min(confidence_values):.1f}%")
    print(f"   Max confidence: {np.max(confidence_values):.1f}%")
    
    # Show examples of each class
    print(f"\n🔍 Example Predictions for Each Class:")
    print("-" * 70)
    
    for class_idx, class_name in enumerate(class_names):
        # Find first occurrence of this class
        class_indices = np.where(y_test == class_idx)[0]
        
        if len(class_indices) > 0:
            idx = class_indices[0]
            pred_idx = np.argmax(predictions[idx])
            confidence = predictions[idx][pred_idx] * 100
            pred_name = class_names[pred_idx]
            
            status = "✅" if class_idx == pred_idx else "❌"
            print(f"{status} {class_name:<20} → Predicted: {pred_name:<20} ({confidence:.1f}%)")


if __name__ == "__main__":
    test_model_predictions()