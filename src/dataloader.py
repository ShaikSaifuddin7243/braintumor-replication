# FILE: src/dataloader.py
import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BrainTumorDataset(Dataset):
    """A flexible PyTorch Dataset for the Brain Tumor MRI data."""
    
    def __init__(self, data_dir, class_map, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images for a given split.
            class_map (dict): A dictionary mapping folder names to class indices.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_paths = glob(os.path.join(data_dir, "*", "*.jpg"))
        self.transform = transform
        self.class_map = class_map
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for path in self.image_paths:
            label_name = os.path.basename(os.path.dirname(path))
            if label_name in self.class_map:
                label_idx = self.class_map[label_name]
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

def get_dataloaders(config, global_config):
    """
    Creates train, validation, and test dataloaders based on a config dictionary.
    
    Args:
        config (dict): The configuration dictionary for a specific experiment (e.g., C1_CONFIG).
        global_config (dict): The global configuration dictionary.
    """
    
    image_size = global_config['IMAGE_SIZE']
    batch_size = config['BATCH_SIZE']
    data_dir = global_config['DATA_DIR']
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- Class Mapping Logic ---
    if config['MODEL_NAME'] == 'CNN_C1':
        # Binary classification: map all tumor types to 1
        class_map = {"notumor": 0, "glioma": 1, "meningioma": 1, "pituitary": 1}
    else:
        # Multi-class: map each class name to a unique index (0, 1, 2, 3...)
        class_map = {class_name: i for i, class_name in enumerate(config['CLASSES'])}

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_dataset = BrainTumorDataset(data_dir=train_dir, class_map=class_map, transform=data_transforms['train'])
    val_dataset = BrainTumorDataset(data_dir=val_dir, class_map=class_map, transform=data_transforms['val'])
    test_dataset = BrainTumorDataset(data_dir=test_dir, class_map=class_map, transform=data_transforms['val']) # No augmentation on test

    # Set num_workers to 0 on Windows for compatibility in some environments
    num_workers = 0 if os.name == 'nt' else 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader