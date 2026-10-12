"""
Train EnhancedCNN-V2 for FER2013 Emotion Classification
Wider network + CBAM attention + deeper stack
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm

from src.models.enhanced_cnn import EnhancedCNN


# ============ Dataset ============

class FER2013Dataset(Dataset):
    """FER2013 Dataset from folder structure."""

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Class mapping
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Collect image paths and labels
        self.samples = []
        split_dir = os.path.join(root_dir, split)

        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                for img_path in glob.glob(os.path.join(class_dir, '*.jpg')):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        print(f"[FER2013Dataset] {split}: {len(self.samples)} images loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('L')  # grayscale

        if self.transform:
            image = self.transform(image)

        return image, label


def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============ Training ============

def compute_class_weights(dataset, num_classes=7):
    """Compute class weights for imbalanced dataset."""
    class_counts = [0] * num_classes
    for _, label in dataset.samples:
        class_counts[label] += 1

    total = sum(class_counts)
    weights = [total / (num_classes * c) for c in class_counts]
    return torch.FloatTensor(weights)


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.2):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


# ============ Main ============

def main():
    # Config
    DATA_DIR = os.environ.get("BANGBANG_FER2013_DIR", "data/fer2013")
    MODEL_DIR = os.environ.get("BANGBANG_MODEL_DIR", "models")
    MODEL_PATH = os.path.join(MODEL_DIR, 'enhanced_cnn_fer2013.pth')

    EPOCHS = 80
    BATCH_SIZE = 64
    LR = 0.001
    PATIENCE = 15
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms with enhanced augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(48, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Datasets
    train_dataset = FER2013Dataset(DATA_DIR, split='train', transform=train_transform)
    test_dataset = FER2013Dataset(DATA_DIR, split='test', transform=test_transform)

    # Class weights for imbalanced data
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Class weights: {class_weights}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = EnhancedCNN(num_classes=7).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} (~{num_params * 4 / 1024 / 1024:.1f} MB)")

    # Loss with label smoothing and optimizer with weight decay
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # Training
    best_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=USE_MIXUP, mixup_alpha=MIXUP_ALPHA
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Saved best model (Acc: {best_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete! Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == '__main__':
    main()