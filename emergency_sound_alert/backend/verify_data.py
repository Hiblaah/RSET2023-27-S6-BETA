import numpy as np
import json
import os

def verify_preprocessed_data():
    """Verify the preprocessed dataset"""
    
    # Check if preprocessed data exists
    data_path = 'data/processed/preprocessed_data.npz'
    metadata_path = 'data/processed/metadata.json'
    
    if not os.path.exists(data_path):
        print("❌ Error: Preprocessed data not found!")
        print(f"   Expected at: {data_path}")
        print("\n💡 You need to run preprocessing first:")
        print("   python src/preprocess_audio.py")
        return
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = np.load(data_path)
    
    print("\n" + "=" * 60)
    print("PREPROCESSED DATA VERIFICATION")
    print("=" * 60)
    
    # Training set
    print(f"\n📊 Training Set:")
    print(f"   Shape: {data['X_train'].shape}")
    print(f"   Labels shape: {data['y_train'].shape}")
    print(f"   Emergency samples: {np.sum(data['y_train'] == 1)}")
    print(f"   Non-emergency samples: {np.sum(data['y_train'] == 0)}")
    
    # Validation set
    print(f"\n📊 Validation Set:")
    print(f"   Shape: {data['X_val'].shape}")
    print(f"   Labels shape: {data['y_val'].shape}")
    print(f"   Emergency samples: {np.sum(data['y_val'] == 1)}")
    print(f"   Non-emergency samples: {np.sum(data['y_val'] == 0)}")
    
    # Test set
    print(f"\n📊 Test Set:")
    print(f"   Shape: {data['X_test'].shape}")
    print(f"   Labels shape: {data['y_test'].shape}")
    print(f"   Emergency samples: {np.sum(data['y_test'] == 1)}")
    print(f"   Non-emergency samples: {np.sum(data['y_test'] == 0)}")
    
    # Total summary
    total_samples = len(data['X_train']) + len(data['X_val']) + len(data['X_test'])
    total_emergency = (np.sum(data['y_train'] == 1) + 
                      np.sum(data['y_val'] == 1) + 
                      np.sum(data['y_test'] == 1))
    total_non_emergency = (np.sum(data['y_train'] == 0) + 
                          np.sum(data['y_val'] == 0) + 
                          np.sum(data['y_test'] == 0))
    
    print(f"\n📊 Total Dataset:")
    print(f"   Total samples: {total_samples}")
    print(f"   Total emergency: {total_emergency}")
    print(f"   Total non-emergency: {total_non_emergency}")
    
    # Feature dimensions
    print(f"\n🔍 Feature Details:")
    print(f"   MFCC coefficients: {data['X_train'].shape[1]}")
    print(f"   Time steps: {data['X_train'].shape[2]}")
    print(f"   Channels: {data['X_train'].shape[3]}")
    
    # Load and display metadata if available
    if os.path.exists(metadata_path):
        print(f"\n📄 Metadata:")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        for key, value in metadata.items():
            print(f"   {key}: {value}")
    
    # Check data quality
    print(f"\n✅ Data Quality Checks:")
    
    # Check for NaN or Inf
    has_nan = np.isnan(data['X_train']).any()
    has_inf = np.isinf(data['X_train']).any()
    
    if has_nan:
        print("   ⚠️  Warning: Dataset contains NaN values!")
    else:
        print("   ✓ No NaN values detected")
    
    if has_inf:
        print("   ⚠️  Warning: Dataset contains Inf values!")
    else:
        print("   ✓ No Inf values detected")
    
    # Check label distribution
    train_ratio = np.sum(data['y_train'] == 1) / len(data['y_train'])
    if 0.2 <= train_ratio <= 0.8:
        print(f"   ✓ Class distribution is balanced ({train_ratio:.2%} emergency)")
    else:
        print(f"   ⚠️  Warning: Imbalanced classes ({train_ratio:.2%} emergency)")
    
    print("\n" + "=" * 60)
    print("✅ Data verification complete!")
    print("=" * 60)
    
    # Recommendation
    if total_samples < 500:
        print("\n⚠️  Warning: Low sample count!")
        print("   Recommendation: Add more data or increase augmentation")
    elif total_samples < 1000:
        print("\n💡 Sample count is acceptable for prototyping")
        print("   Consider adding more data for production deployment")
    else:
        print("\n✅ Sample count looks good for training!")
    
    print("\n📍 Next step: Train the model")
    print("   Run: python src/train_model.py")

if __name__ == "__main__":
    verify_preprocessed_data()