import tensorflow as tf
import numpy as np
import json

def test_all_emergency_classes():
    """Test that model can detect ALL emergency types"""
    
    print("="*60)
    print("TESTING EMERGENCY DETECTION FOR ALL CLASSES")
    print("="*60)
    
    # Load TFLite model
    tflite_path = 'models/emergency_sound_model_multiclass_quantized.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Load test data
    data = np.load('data/processed/preprocessed_data_multiclass.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    
    class_names = ['Non-Emergency', 'Glass Breaking', 'Siren', 'Baby Crying']
    
    print("\n🔍 Testing Each Emergency Type:\n")
    
    # Test each class
    for class_idx in range(4):
        class_name = class_names[class_idx]
        
        # Find samples of this class
        class_indices = np.where(y_test == class_idx)[0]
        
        if len(class_indices) == 0:
            print(f"⚠️  No test samples for {class_name}")
            continue
        
        # Test first 5 samples of this class
        print(f"{'='*60}")
        print(f"Testing: {class_name}")
        print(f"{'='*60}")
        
        correct = 0
        for i, idx in enumerate(class_indices[:5]):
            test_input = X_test[idx:idx+1].astype(input_details[0]['dtype'])
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], test_input)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            
            predictions = output[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class] * 100
            
            is_correct = predicted_class == class_idx
            if is_correct:
                correct += 1
            
            symbol = "✅" if is_correct else "❌"
            
            print(f"\nSample {i+1}:")
            print(f"  True: {class_name}")
            print(f"  Predictions:")
            for j, (name, prob) in enumerate(zip(class_names, predictions)):
                marker = "👉" if j == predicted_class else "  "
                print(f"    {marker} {name:20s}: {prob*100:6.2f}%")
            print(f"  Result: {symbol} {'CORRECT' if is_correct else 'WRONG'}")
        
        accuracy = (correct / min(5, len(class_indices))) * 100
        print(f"\n📊 {class_name} Detection Rate: {accuracy:.0f}% ({correct}/5)")
        print()
    
    print("\n" + "="*60)
    print("✅ EMERGENCY DETECTION TEST COMPLETE!")
    print("="*60)
    
    # Overall test on all emergency sounds
    emergency_indices = np.where(y_test > 0)[0]  # All non-zero classes
    
    print(f"\n📊 Overall Emergency Detection:")
    print(f"   Total emergency samples: {len(emergency_indices)}")
    
    emergency_correct = 0
    for idx in emergency_indices:
        test_input = X_test[idx:idx+1].astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        predicted_class = np.argmax(output[0])
        true_class = y_test[idx]
        
        if predicted_class == true_class:
            emergency_correct += 1
    
    emergency_accuracy = (emergency_correct / len(emergency_indices)) * 100
    print(f"   Correctly detected: {emergency_correct}/{len(emergency_indices)}")
    print(f"   Accuracy: {emergency_accuracy:.1f}%")
    
    if emergency_accuracy == 100.0:
        print(f"\n🎉 PERFECT! Model detects ALL emergencies correctly!")
    elif emergency_accuracy >= 95.0:
        print(f"\n✅ EXCELLENT! Model is highly reliable!")
    else:
        print(f"\n⚠️  Model may need improvement")

if __name__ == "__main__":
    test_all_emergency_classes()