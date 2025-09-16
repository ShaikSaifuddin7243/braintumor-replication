import yaml
import torch
import torch.nn as nn

# Import our custom modules
from src.dataloader import get_dataloaders
from src.model import CNN_C1

def run_overfit_test():
    """
    Trains the model on a single batch of data to verify the pipeline.
    """
    print("--- Starting Small Overfit Test ---")

    # --- 1. Load Configuration ---
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Setup Device, Model, Loss, Optimizer ---
    device = torch.device(config['DEVICE'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNN_C1(num_classes=config['NUM_CLASSES']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])

    # --- 3. Load a SINGLE BATCH of Data ---
    train_loader, _, _ = get_dataloaders(
        data_dir=config['DATA_DIR'],
        image_size=config['IMAGE_SIZE'],
        batch_size=config['BATCH_SIZE']
    )

    try:
        inputs, labels = next(iter(train_loader))
    except StopIteration:
        print("ERROR: Dataloader is empty. Cannot perform overfit test.")
        return

    inputs, labels = inputs.to(device), labels.to(device)
    print(f"Loaded one batch of size: {inputs.shape}")

    # --- 4. Overfit Loop ---
    model.train()
    for i in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy on this single batch
        _, preds = torch.max(outputs, 1)
        correct_predictions = torch.sum(preds == labels.data)
        accuracy = correct_predictions.double() / labels.size(0)

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1:03d}/100 | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.4f}")

    print("\n--- Overfit Test Complete ---")
    if accuracy.item() > 0.95:
        print("✅ PASS: Model successfully overfit the batch. Pipeline is likely working correctly.")
    else:
        print("❌ FAIL: Model did not overfit the batch. There may be a bug in the model or training loop.")

if __name__ == '__main__':
    run_overfit_test()