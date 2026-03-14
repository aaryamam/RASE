"""
train_resnet50_tinyimagenet.py — Train ResNet-50 on Tiny-ImageNet from scratch and save checkpoint.

Usage: python scripts/train_resnet50_tinyimagenet.py
"""

import os
import random
import urllib.request
import zipfile
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

# ──────────────────────────────────────────────
# Shared constants
# ──────────────────────────────────────────────
SEED = 42
# Tiny-ImageNet normalizations (same as ImageNet usually)
TINY_MEAN = (0.485, 0.456, 0.406)
TINY_STD  = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Data Preparation
# ──────────────────────────────────────────────
def download_and_prepare_tiny_imagenet(data_dir="./data"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_dir = os.path.join(data_dir, "tiny-imagenet-200")
    
    if os.path.exists(extract_dir):
        return extract_dir
    
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(zip_path):
        print("Downloading Tiny-ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting Tiny-ImageNet...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print("Reorganizing validation set for ImageFolder...")
    val_dir = os.path.join(extract_dir, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    if os.path.exists(val_annotations_file):
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_id = parts[1]
                
                class_dir = os.path.join(val_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                
                src = os.path.join(val_images_dir, img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.exists(src):
                    shutil.move(src, dst)
        
        # Optionally remove the now-empty images folder
        if os.path.exists(val_images_dir):
            shutil.rmtree(val_images_dir)
    
    return extract_dir


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
def make_resnet50_tinyimagenet():
    # ResNet-50 modified for Tiny-ImageNet (64x64 images, 200 classes)
    model = resnet50(weights=None)
    # Replace the initial 7x7 conv with a 3x3 conv, stride 1 (better for small images)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool to preserve spatial resolution
    model.maxpool = nn.Identity()
    # Replace final FC layer (2048 in_features for ResNet50) with 200 out
    model.fc = nn.Linear(2048, 200)
    return model


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
def get_dataloaders(batch_size=128):
    data_dir = download_and_prepare_tiny_imagenet("./data")
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TINY_MEAN, TINY_STD),
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(val_dir, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
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
    CKPT_PATH = "checkpoints/resnet50_tinyimagenet.pt"
    BEST_CKPT_PATH = "checkpoints/resnet50_tinyimagenet_best.pt"
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    model = make_resnet50_tinyimagenet().to(DEVICE)

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
