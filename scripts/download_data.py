import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_dataset():
    """
    Downloads and extracts the Brain Tumor MRI dataset from Kaggle.
    """
    dataset_name = "masoudnickparvar/brain-tumor-mri-dataset"
    data_dir = "data"
    
    print("Initializing Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading dataset '{dataset_name}' to '{data_dir}'...")
    api.dataset_download_files(dataset_name, path=data_dir, quiet=False)
    
    zip_path = os.path.join(data_dir, f"{dataset_name.split('/')[1]}.zip")
    
    print(f"Extracting '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    print(f"Cleaning up zip file...")
    os.remove(zip_path)
    
    print("\nDataset is ready in the 'data' directory.")
    # List contents to verify
    for root, dirs, files in os.walk(data_dir):
        if root == data_dir:
            print(f"Contents of '{data_dir}': {dirs}")

if __name__ == "__main__":
    download_and_extract_dataset()