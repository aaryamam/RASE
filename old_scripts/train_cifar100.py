"""
train_cifar100.py — Train ResNet-18 on CIFAR-100, save best checkpoint.

Two modes controlled by CREATE_VULNERABLE_MODEL:
  False → Standard training (augmentation + weight decay) → resnet18_cifar100.pt
  True  → Overfit training (no augment, no WD)           → resnet18_cifar100_vulnerable.pt

Usage:
  python scripts/train_cifar100.py          # standard model
  # Then flip CREATE_VULNERABLE_MODEL=True and run again for MIA-vulnerable model
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# ──────────────────────────────────────────────
# Seed everything
# ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# CIFAR-100 constants
# ──────────────────────────────────────────────
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# ──────────────────────────────────────────────
# Unlearning Config
# ──────────────────────────────────────────────
CREATE_VULNERABLE_MODEL = False  # Set to True to generate a high-MIA checkpoint

# ──────────────────────────────────────────────
# Model: ResNet-18 for CIFAR (3×3 conv1, no maxpool)
# ──────────────────────────────────────────────
def make_resnet18_cifar100():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 100)
    return model

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
def get_dataloaders(batch_size=128):
    if CREATE_VULNERABLE_MODEL:
        # No augmentation → forces memorization for high MIA signal
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    else:
        # Standard augmentation → generalised model
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_set = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader

# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return 100.0 * correct / total


def main():
    print(f"Device: {DEVICE}")
    print(f"Mode: {'VULNERABLE (no augment, no WD)' if CREATE_VULNERABLE_MODEL else 'STANDARD (augment + WD)'}")
    epochs = 100
    batch_size = 128

    train_loader, test_loader = get_dataloaders(batch_size)
    model = make_resnet18_cifar100().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    wd = 0.0 if CREATE_VULNERABLE_MODEL else 5e-4
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    save_path = ("checkpoints/resnet18_cifar100_vulnerable.pt"
                 if CREATE_VULNERABLE_MODEL
                 else "checkpoints/resnet18_cifar100.pt")

    print(f"Checkpoint will be saved to: {save_path}")
    print(f"Weight decay: {wd}")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'LR':>8}")
    print("-" * 55)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_acc = evaluate(model, test_loader)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:8.2f}% | {test_acc:7.2f}% | {lr:8.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s.  Best test accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    main()
