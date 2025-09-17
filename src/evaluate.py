# FILE: src/evaluate.py
import yaml
import torch
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Use explicit relative imports
from .dataloader import get_dataloaders
from .model import CNN_C1, CNN_C2

# --- Model Registry ---
models = {
    "CNN_C1": CNN_C1,
    "CNN_C2": CNN_C2
}

def evaluate_model(args):
    """Loads the best model for a given config and evaluates it on the test set."""
    print(f"--- Starting Final Evaluation on Test Set for {args.model_config} ---")

    # --- 1. Load Configuration ---
    with open('config.yaml', 'r') as f:
        full_config = yaml.safe_load(f)
    
    config = full_config[args.model_config]
    global_config = {k: v for k, v in full_config.items() if not isinstance(v, dict)}

    # --- 2. Setup Device and Load Data ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(
        config=config,
        global_config=global_config
    )
    
    # --- 3. Load Model Architecture and Best Weights ---
    ModelClass = models[config['MODEL_NAME']]
    model = ModelClass(num_classes=config['NUM_CLASSES']).to(device)
    
    model_path = f"results/best_model_{config['MODEL_NAME']}.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model weights not found at {model_path}. Please run training first.")
        return
        
    # --- 4. Evaluate on Test Set ---
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 5. Print Classification Report & Confusion Matrix ---
    print("\n--- Evaluation Report ---")
    target_names = config['CLASSES']
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print(report)

    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {config['MODEL_NAME']}")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    cm_save_path = f"results/confusion_matrix_{config['MODEL_NAME']}.png"
    plt.savefig(cm_save_path)
    print(f"Confusion matrix saved to {cm_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN model.")
    parser.add_argument('--model_config', type=str, required=True, choices=['C1_CONFIG', 'C2_CONFIG'],
                        help="The configuration key in config.yaml to use (e.g., 'C2_CONFIG').")
    args = parser.parse_args()
    evaluate_model(args)