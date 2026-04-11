import os
import shutil
import pandas as pd

def organize_custom_emergency_with_esc50():
    """
    Use your custom emergency sounds + ESC-50 non-emergency sounds
    """
    
    # Paths for YOUR custom files
    your_emergency_path = 'data/raw/custom/emergency'  # Your 50 files here
    
    # Paths for ESC-50
    esc50_audio_path = 'data/raw/ESC-50/audio'
    esc50_meta_path = 'data/raw/ESC-50/meta/esc50.csv'
    
    # Output paths
    emergency_output = 'data/raw/emergency'
    non_emergency_output = 'data/raw/non_emergency'
    
    # Create directories
    os.makedirs(emergency_output, exist_ok=True)
    os.makedirs(non_emergency_output, exist_ok=True)
    
    print("=" * 60)
    print("ORGANIZING DATASET")
    print("=" * 60)
    
    # Step 1: Copy YOUR emergency sounds
    print("\n📁 Step 1: Copying your custom emergency sounds...")
    
    if os.path.exists(your_emergency_path):
        custom_emergency_files = [f for f in os.listdir(your_emergency_path) 
                                  if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for filename in custom_emergency_files:
            source = os.path.join(your_emergency_path, filename)
            destination = os.path.join(emergency_output, filename)
            shutil.copy2(source, destination)
        
        print(f"✅ Copied {len(custom_emergency_files)} custom emergency files")
    else:
        print(f"⚠️  Warning: Custom emergency path not found: {your_emergency_path}")
        print("   Please put your 50 emergency files there first!")
        return
    
    # Step 2: Extract NON-EMERGENCY sounds from ESC-50
    print("\n📁 Step 2: Extracting non-emergency sounds from ESC-50...")
    
    if not os.path.exists(esc50_meta_path):
        print(f"❌ ESC-50 not found. Please download it first!")
        return
    
    # Load ESC-50 metadata
    df = pd.read_csv(esc50_meta_path)
    
    # Define NON-emergency categories from ESC-50
    non_emergency_categories = [
        'dog',                # dog bark (not emergency for this project)
        'rain',
        'sea_waves',
        'thunderstorm',
        'wind',
        'footsteps',
        'breathing',
        'coughing',
        'sneezing',
        'clapping',
        'keyboard_typing',
        'door_wood_knock',
        'mouse_click',
        'engine',
        'train',
        'airplane',
        'washing_machine',
        'vacuum_cleaner',
        'clock_alarm',
        'clock_tick',
        'water_drops',
        'crickets',
        'insects',
        'frog',
        'crow',
        'hen',
        'rooster',
        'cat',
        'sheep',
        'cow',
        'pig',
        'hand_saw',
        'chainsaw',
        'can_opening',
        'car_horn',
        'church_bells',
        'fireworks',
        'helicopter',
        'laughing',
        'pouring_water',
        'toilet_flush',
        'drinking_sipping',
        'brushing_teeth',
        'snoring',
    ]
    
    non_emergency_count = 0
    
    # Copy non-emergency files
    for idx, row in df.iterrows():
        filename = row['filename']
        category = row['category']
        source = os.path.join(esc50_audio_path, filename)
        
        if category in non_emergency_categories and os.path.exists(source):
            destination = os.path.join(non_emergency_output, filename)
            shutil.copy2(source, destination)
            non_emergency_count += 1
    
    print(f"✅ Copied {non_emergency_count} non-emergency files from ESC-50")
    
    # Step 3: Optionally copy YOUR custom non-emergency files too
    your_non_emergency_path = 'data/raw/custom/non_emergency'
    
    if os.path.exists(your_non_emergency_path):
        print(f"\n📁 Step 3: Copying your custom non-emergency sounds...")
        custom_non_emergency_files = [f for f in os.listdir(your_non_emergency_path) 
                                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for filename in custom_non_emergency_files:
            source = os.path.join(your_non_emergency_path, filename)
            destination = os.path.join(non_emergency_output, filename)
            shutil.copy2(source, destination)
        
        print(f"✅ Copied {len(custom_non_emergency_files)} custom non-emergency files")
        non_emergency_count += len(custom_non_emergency_files)
    
    # Final summary
    print("\n" + "=" * 60)
    print("DATASET ORGANIZATION COMPLETE!")
    print("=" * 60)
    print(f"✅ Emergency sounds: {len(custom_emergency_files)} files")
    print(f"✅ Non-emergency sounds: {non_emergency_count} files")
    print(f"📊 Total: {len(custom_emergency_files) + non_emergency_count} files")
    
    print(f"\n📍 Files organized in:")
    print(f"   - {emergency_output}")
    print(f"   - {non_emergency_output}")
    
    print(f"\n✅ Next step: Run 'python src/preprocess_audio.py'")
    
    # Save info
    import json
    info = {
        'emergency_files': len(custom_emergency_files),
        'non_emergency_files': non_emergency_count,
        'emergency_source': 'custom',
        'non_emergency_source': 'ESC-50 + custom (if available)'
    }
    
    with open('data/raw/dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)

if __name__ == "__main__":
    organize_custom_emergency_with_esc50()
