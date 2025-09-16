# FILE: scripts/prepare_dataset.py
import os, hashlib, shutil
from glob import glob
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def prepare_dataset(base_dir="data", output_dir="data/clean"):
    print("--- Step 1: Finding all image files ---")
    image_paths = glob(os.path.join(base_dir, "*", "*", "*.jpg"))
    print(f"Found {len(image_paths)} total image files.")
    print("\n--- Step 2: De-duplicating files via content hashing ---")
    unique_files = {}
    for path in tqdm(image_paths, desc="Hashing images"):
        file_hash = calculate_sha256(path)
        if file_hash not in unique_files:
            label = os.path.basename(os.path.dirname(path))
            unique_files[file_hash] = (path, label)
    print(f"Found {len(unique_files)} unique images.")
    unique_list = [{"filepath": path, "label": label} for path, label in unique_files.values()]
    df = pd.DataFrame(unique_list)
    print("\n--- Step 3: Performing stratified train-val-test split (70-15-15) ---")
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])
    print(f"Train set size: {len(train_df)} images")
    print(f"Validation set size: {len(val_df)} images")
    print(f"Test set size: {len(test_df)} images")
    print("\n--- Step 4: Copying files to new clean directory structure ---")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for split_name, split_df in splits.items():
        split_path = os.path.join(output_dir, split_name)
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split_name} files"):
            label_dir = os.path.join(split_path, row['label'])
            os.makedirs(label_dir, exist_ok=True)
            shutil.copy(row['filepath'], label_dir)
    print(f"\n--- Remediation Complete! Dataset is in '{output_dir}' ---")

if __name__ == "__main__":
    # First, ensure data is downloaded. This assumes download_data.py has been run.
    if not os.path.exists("data/Training"):
        print("Original data not found. Please run scripts/download_data.py first.")
    else:
        prepare_dataset()