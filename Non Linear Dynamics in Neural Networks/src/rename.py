import os

root_dir = "dataset/taowu"

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".nii.gz"):
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, file.replace(".nii.gz", ".nii"))
            
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} → {new_path}")
