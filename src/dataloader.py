# FILE: src/dataloader.py
import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transform = transform
        self.class_to_idx = {"notumor": 0, "glioma": 1, "meningioma": 1, "pituitary": 1}
        self.samples = self._create_samples()
    def _create_samples(self):
        samples = []
        for path in self.image_paths:
            label_name = os.path.basename(os.path.dirname(path))
            label_idx = self.class_to_idx[label_name]
            samples.append((path, label_idx))
        return samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_dir, image_size, batch_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    train_dataset = BrainTumorDataset(data_dir=train_dir, transform=data_transforms['train'])
    val_dataset = BrainTumorDataset(data_dir=val_dir, transform=data_transforms['val'])
    test_dataset = BrainTumorDataset(data_dir=test_dir, transform=data_transforms['val'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader