# FILE: src/evaluate.py
import yaml, torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from .dataloader import get_dataloaders
from .model import CNN_C1

def evaluate_model():
    print("--- Starting Final Evaluation on Test Set ---")
    with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    _, _, test_loader = get_dataloaders(data_dir=config['DATA_DIR'], image_size=config['IMAGE_SIZE'], batch_size=config['BATCH_SIZE'])
    model = CNN_C1(num_classes=config['NUM_CLASSES']).to(device)
    model_path = "results/best_model.pth"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model weights from {model_path}")
    except FileNotFoundError:
        print(f"ERROR: Model weights not found at {model_path}.")
        return
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print("\n--- Evaluation Report ---")
    target_names = config['CLASSES']
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print(report)
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to results/confusion_matrix.png")

if __name__ == '__main__': evaluate_model()