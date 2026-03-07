"""
train_resnet.py — Train ResNet-18 on CIFAR-10 from scratch and save checkpoint.

Usage: python scripts/train_resnet.py
"""

import os
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
# Shared constants
# ──────────────────────────────────────────────
SEED = 42
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
def make_resnet18_cifar10():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
def get_dataloaders(batch_size=128):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += images.size(0)
    return 100.0 * correct / total


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    EPOCHS = 200
    BATCH_SIZE = 128
    CKPT_PATH = "checkpoints/resnet18_cifar10.pt"
    BEST_CKPT_PATH = "checkpoints/resnet18_cifar10_best.pt"
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    model = make_resnet18_cifar10().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_epoch = 1

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        acc = evaluate(model, test_loader)
        scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {loss:.4f} | Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), BEST_CKPT_PATH)

    torch.save(model.state_dict(), CKPT_PATH)
    final_acc = evaluate(model, test_loader)

    print()
    print("Training complete.")
    print(f"Final test accuracy: {final_acc:.2f}%")
    print(f"Best test accuracy:  {best_acc:.2f}% (epoch {best_epoch})")
    print(f"Checkpoint saved to: {CKPT_PATH}")


if __name__ == "__main__":
    main()
