"""
train_vit.py — Train ViT-Small on CIFAR-10 from scratch and save checkpoint.

Usage: python scripts/train_vit.py
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
# Model: ViT-Small for CIFAR-10 (self-contained)
# ──────────────────────────────────────────────
class PatchEmbedding(nn.Module):
    """Convert 32×32 image into a sequence of patch embeddings + CLS token."""

    def __init__(self, in_channels=3, patch_size=4, embed_dim=256, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 64

        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)          # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)    # (B, num_patches, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)          # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN → MHA → residual → LN → MLP → residual."""

    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmallCIFAR10(nn.Module):
    """
    ViT-Small for CIFAR-10 (32×32).
    Patch 4×4, embed 256, depth 6, heads 4, MLP ratio 4.
    """

    def __init__(self, embed_dim=256, depth=6, num_heads=4, mlp_ratio=4,
                 dropout=0.1, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)    # (B, 65, 256)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])     # CLS token → LayerNorm
        x = self.head(x)           # (B, num_classes)
        return x


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
    BASE_LR = 1e-3
    WARMUP_LR = 1e-4
    WARMUP_EPOCHS = 10
    CKPT_PATH = "checkpoints/vit_small_cifar10.pt"
    BEST_CKPT_PATH = "checkpoints/vit_small_cifar10_best.pt"
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)
    model = ViTSmallCIFAR10().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_epoch = 1

    for epoch in range(1, EPOCHS + 1):
        # Linear warmup for first WARMUP_EPOCHS epochs
        if epoch <= WARMUP_EPOCHS:
            warmup_scale = WARMUP_LR + (BASE_LR - WARMUP_LR) * (epoch - 1) / (WARMUP_EPOCHS - 1)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_scale

        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        acc = evaluate(model, test_loader)

        if epoch > WARMUP_EPOCHS:
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
