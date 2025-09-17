# FILE: src/train.py
import yaml
import torch
import torch.nn as nn
import os
import argparse
from tqdm import tqdm

# Use explicit relative imports
from .dataloader import get_dataloaders
from .model import CNN_C1, CNN_C2

# --- Model Registry ---
# This allows us to select the model dynamically based on the config file.
models = {
    "CNN_C1": CNN_C1,
    "CNN_C2": CNN_C2
}

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)
    return running_loss / total_samples, (correct_predictions.double() / total_samples).item()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct_predictions, total_samples = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
    return running_loss / total_samples, (correct_predictions.double() / total_samples).item()

def main(args):
    # --- 1. Load Configuration ---
    with open('config.yaml', 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Select the configuration for the specified model
    config = full_config[args.model_config]
    global_config = {k: v for k, v in full_config.items() if not isinstance(v, dict)}

    # --- 2. Setup Device, Model, Loss, Optimizer ---
    device = torch.device(global_config['DEVICE'] if torch.cuda.is_available() else "cpu")
    print(f"--- Experiment: {config['MODEL_NAME']} ---")
    print(f"Using device: {device}")

    # Dynamically select the model class from the registry
    ModelClass = models[config['MODEL_NAME']]
    model = ModelClass(num_classes=config['NUM_CLASSES']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    
    # --- 3. Load Data ---
    train_loader, val_loader, _ = get_dataloaders(
        config=config,
        global_config=global_config
    )
    
    # --- 4. Training Loop ---
    best_val_acc = 0.0
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True) 

    for epoch in range(config['EPOCHS']):
        print(f"\n--- Epoch {epoch+1}/{config['EPOCHS']} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the model with a name corresponding to the experiment
            save_path = os.path.join(results_dir, f"best_model_{config['MODEL_NAME']}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with accuracy: {best_val_acc:.4f}")

    print(f"\n--- Training Complete for {config['MODEL_NAME']} ---")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a CNN model for brain tumor classification.")
    parser.add_argument('--model_config', type=str, required=True, choices=['C1_CONFIG', 'C2_CONFIG'],
                        help="The configuration key in config.yaml to use (e.g., 'C1_CONFIG').")
    args = parser.parse_args()
    main(args)