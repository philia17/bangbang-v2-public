"""
Simple CNN for FER2013 Emotion Classification
Trains on real FER2013 data, saves best model
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


# ============ Model ============

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with optional stride for downsampling."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=8)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += identity
        out = self.relu(out)
        return out


class SimpleCNN(nn.Module):
    """
    Enhanced CNN for 48x48 grayscale emotion classification.

    Architecture:
    Input (1, 48, 48)
    → ResBlock1 (32, stride=1) → 48x48
    → ResBlock2 (64, stride=2) → 24x24
    → ResBlock3 (128, stride=2) → 12x12
    → ResBlock4 (256, stride=2) → 6x6
    → GlobalAvgPool → Dropout(0.4) → FC(7)

    Uses residual connections and SE attention for better feature learning.
    """
    def __init__(self, num_classes=7):
        super().__init__()

        self.stages = nn.Sequential(
            # Stage 1: 48x48 → 48x48
            ResidualBlock(1, 32, stride=1),
            # Stage 2: 48x48 → 24x24
            ResidualBlock(32, 64, stride=2),
            # Stage 3: 24x24 → 12x12
            ResidualBlock(64, 128, stride=2),
            # Stage 4: 12x12 → 6x6
            ResidualBlock(128, 256, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stages(x)
        x = self.classifier(x)
        return x


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


# ============ Training ============

def compute_class_weights(dataset, num_classes=7):
    """Compute class weights for imbalanced dataset."""
    class_counts = [0] * num_classes
    for _, label in dataset.samples:
        class_counts[label] += 1

    total = sum(class_counts)
    weights = [total / (num_classes * c) for c in class_counts]
    return torch.FloatTensor(weights)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

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
    MODEL_PATH = os.path.join(MODEL_DIR, 'simple_cnn_fer2013.pth')

    EPOCHS = 50
    BATCH_SIZE = 64
    LR = 0.001
    PATIENCE = 10

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
    model = SimpleCNN(num_classes=7).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss with label smoothing and optimizer with weight decay
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training
    best_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
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
