"""
linear_probe_cifar100.py — Linear Probe Test for GPM-A Unlearning Verification

Freezes each model's backbone, extracts 512-dim features, trains a fresh
linear classifier (probe) on the entire CIFAR-100 training set, then
evaluates per-class accuracy.  If unlearning actually deleted features,
the probe cannot recover forget-class performance.

Checkpoints tested:
  1. Original          — checkpoints/resnet18_cifar100.pt
  2. GPM-A unlearned   — checkpoints/resnet18_cifar100_unlearn_gpma.pt
  3. GA unlearned      — checkpoints/resnet18_cifar100_unlearn_ga.pt

Usage:
  python scripts/linear_probe_cifar100.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
SEED = 42
FORGET_CLASS = 81
NUM_CLASSES = 100
FEATURE_DIM = 512
BATCH_SIZE = 256
PROBE_EPOCHS = 50
PROBE_LR = 0.1
PROBE_MOMENTUM = 0.9
PROBE_WD = 1e-4

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CHECKPOINTS = {
    "Original": "checkpoints/resnet18_cifar100.pt",
    "GPM-A":    "checkpoints/resnet18_cifar100_unlearn_gpma.pt",
    "GA":       "checkpoints/resnet18_cifar100_unlearn_ga.pt",
}

# ──────────────────────────────────────────────
# Seed everything
# ──────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Model: ResNet-18 for CIFAR-100 (same as training)
# ──────────────────────────────────────────────
def make_resnet18_cifar100():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 100)
    return model


def load_backbone(ckpt_path):
    """Load checkpoint, replace fc with Identity, freeze all params."""
    model = make_resnet18_cifar100()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.fc = nn.Identity()  # output = 512-dim features
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(DEVICE)


# ──────────────────────────────────────────────
# Data (test-time transforms only — no augmentation)
# ──────────────────────────────────────────────
def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_set = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    return train_set, test_set


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────
@torch.no_grad()
def extract_features(backbone, dataset):
    """Forward-pass entire dataset through frozen backbone, return (features, labels)."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)
    all_features = []
    all_labels = []
    for images, labels in loader:
        images = images.to(DEVICE)
        feats = backbone(images)  # (B, 512)
        all_features.append(feats.cpu())
        all_labels.append(labels)
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


# ──────────────────────────────────────────────
# Linear probe training
# ──────────────────────────────────────────────
def train_probe(train_features, train_labels):
    """Train a fresh linear probe on cached features."""
    probe = nn.Linear(FEATURE_DIM, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(probe.parameters(), lr=PROBE_LR,
                          momentum=PROBE_MOMENTUM, weight_decay=PROBE_WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROBE_EPOCHS)

    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)

    probe.train()
    for epoch in range(1, PROBE_EPOCHS + 1):
        epoch_loss = 0.0
        n = 0
        for feats, labels in loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = probe(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * feats.size(0)
            n += feats.size(0)
        scheduler.step()
        if epoch % 10 == 0:
            print(f"    Probe epoch {epoch:3d}/{PROBE_EPOCHS}  loss={epoch_loss/n:.4f}")

    return probe


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_probe(probe, test_features, test_labels):
    """Compute per-class accuracy, forget/retain/overall accuracy."""
    probe.eval()
    dataset = TensorDataset(test_features, test_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_targets = []
    for feats, labels in loader:
        feats = feats.to(DEVICE)
        logits = probe(feats)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(labels)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Per-class accuracy
    per_class_acc = {}
    for c in range(NUM_CLASSES):
        mask = all_targets == c
        if mask.sum() > 0:
            per_class_acc[c] = 100.0 * (all_preds[mask] == c).float().mean().item()
        else:
            per_class_acc[c] = 0.0

    forget_acc = per_class_acc[FORGET_CLASS]
    retain_classes = [c for c in range(NUM_CLASSES) if c != FORGET_CLASS]
    retain_acc = np.mean([per_class_acc[c] for c in retain_classes])
    overall_acc = 100.0 * (all_preds == all_targets).float().mean().item()

    return forget_acc, retain_acc, overall_acc, per_class_acc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Forget class: {FORGET_CLASS}")
    print(f"Probe config: epochs={PROBE_EPOCHS}, lr={PROBE_LR}, "
          f"batch_size={BATCH_SIZE}, wd={PROBE_WD}\n")

    # Verify checkpoints exist
    for name, path in CHECKPOINTS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"  ✓ {name}: {path}")
    print()

    # Load datasets once
    train_set, test_set = get_datasets()

    results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"{'='*60}")
        print(f"  Processing: {name} ({ckpt_path})")
        print(f"{'='*60}")

        # 1. Load frozen backbone
        print("  Loading backbone...")
        backbone = load_backbone(ckpt_path)

        # 2. Extract training features
        print("  Extracting training features...")
        train_features, train_labels = extract_features(backbone, train_set)
        print(f"    Train features shape: {train_features.shape}")

        # 3. Extract test features
        print("  Extracting test features...")
        test_features, test_labels = extract_features(backbone, test_set)
        print(f"    Test features shape: {test_features.shape}")

        # 4. Train linear probe
        print("  Training linear probe...")
        # Reset seed for fair comparison across probes
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        probe = train_probe(train_features, train_labels)

        # 5. Evaluate
        print("  Evaluating probe...")
        forget_acc, retain_acc, overall_acc, per_class_acc = evaluate_probe(
            probe, test_features, test_labels
        )
        results[name] = (forget_acc, retain_acc, overall_acc)
        print(f"    Forget acc: {forget_acc:.1f}%  |  Retain acc: {retain_acc:.1f}%  "
              f"|  Overall: {overall_acc:.1f}%\n")

        # Free GPU memory
        del backbone, probe
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────
    # Final summary table
    # ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("LINEAR PROBE TEST — CIFAR-100 ResNet-18, Forget Class 81")
    print("=" * 60)
    print(f"{'Backbone':<13}| {'Probe Forget Acc':^17}| {'Probe Retain Acc':^17}| {'Probe Overall':^15}")
    print(f"{'-'*13}|{'-'*17}|{'-'*17}|{'-'*15}")
    for name in ["Original", "GPM-A", "GA"]:
        fa, ra, oa = results[name]
        print(f"{name:<13}|{fa:^17.1f}%|{ra:^17.1f}%|{oa:^15.1f}%")
    print("=" * 60)

    # Feature deletion scores
    orig_forget = results["Original"][0]
    gpma_forget = results["GPM-A"][0]
    ga_forget = results["GA"][0]

    gpma_drop = orig_forget - gpma_forget
    ga_drop = orig_forget - ga_forget

    print(f"\nFeature deletion score (Original → GPM-A drop): {gpma_drop:.1f}pp")
    print(f"Feature deletion score (Original → GA drop): {ga_drop:.1f}pp")

    # Interpretation
    print()
    if gpma_forget < 10.0:
        print("GPM-A: Strong feature deletion confirmed — "
              "probe cannot recover forget class from backbone features.")
    elif gpma_forget < 30.0:
        print("GPM-A: Partial feature deletion — "
              "some forget-class information remains in backbone features.")
    else:
        print("GPM-A: Boundary surgery suspected — "
              "features intact, unlearning only warped the decision boundary.")

    if ga_forget < 10.0:
        print("GA:    Strong feature deletion confirmed — "
              "probe cannot recover forget class from backbone features.")
    elif ga_forget < 30.0:
        print("GA:    Partial feature deletion — "
              "some forget-class information remains in backbone features.")
    else:
        print("GA:    Boundary surgery suspected — "
              "features intact, unlearning only warped the decision boundary.")


if __name__ == "__main__":
    main()
