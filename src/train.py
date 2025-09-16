# FILE: src/train.py
import yaml, torch, torch.nn as nn, os
from tqdm import tqdm
from .dataloader import get_dataloaders
from .model import CNN_C1

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

def main():
    with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
    device = torch.device(config['DEVICE'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = CNN_C1(num_classes=config['NUM_CLASSES']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
    train_loader, val_loader, _ = get_dataloaders(data_dir=config['DATA_DIR'], image_size=config['IMAGE_SIZE'], batch_size=config['BATCH_SIZE'])
    best_val_acc = 0.0
    os.makedirs("results", exist_ok=True)
    for epoch in range(config['EPOCHS']):
        print(f"\n--- Epoch {epoch+1}/{config['EPOCHS']} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join("results", "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with accuracy: {best_val_acc:.4f}")
    print(f"\n--- Training Complete --- \nBest Validation Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__': main()