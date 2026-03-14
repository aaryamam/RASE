"""
relearn.py — Load unlearned models, try to teach class 0 back, print results.
Usage: python scripts/relearn.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
FORGET_CLASS = 0

CHECKPOINTS = {
    "GPM":     "checkpoints/resnet18_cifar10_unlearn_gpm.pt",
    "GA":      "checkpoints/resnet18_cifar10_unlearn_ga.pt",
    "GA+FT":   "checkpoints/resnet18_cifar10_unlearn_gaft.pt",
    "Retrain": "checkpoints/resnet18_cifar10_retrain.pt",
}

# ──────────────────────────────────────────────
# Model (same as other scripts)
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
def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_attack_data():
    """
    Returns:
      relearn_set: 100 class-0 test images (with train-time augmentation for relearning)
      eval_set: 900 class-0 test images (with test-time transforms for evaluation)
      retain_set: 1000 retain training images (with train-time augmentation)
      test_retain_loader: test set for classes 1-9 (for retain acc evaluation)
    """
    # Test set — use all 1000 class-0 images: 100 for relearn, 900 for eval
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=get_test_transform()
    )
    targets = np.array(test_set.targets)
    class0_idx = np.where(targets == FORGET_CLASS)[0].tolist()

    rng = np.random.RandomState(SEED)
    rng.shuffle(class0_idx)
    relearn_idx = class0_idx[:100]
    eval_idx = class0_idx[100:]  # 900 images for measuring recovery

    # For relearning, we want train-time augmentation on those 100 images
    test_set_aug = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=get_train_transform()
    )
    relearn_set = Subset(test_set_aug, relearn_idx)
    eval_set = Subset(test_set, eval_idx)  # test-time transforms for eval

    # Retain: 1000 random training images from classes 1-9
    train_set_aug = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=get_train_transform()
    )
    train_targets = np.array(train_set_aug.targets)
    retain_idx = np.where(train_targets != FORGET_CLASS)[0]
    chosen_retain = rng.choice(retain_idx, 1000, replace=False).tolist()
    retain_set = Subset(train_set_aug, chosen_retain)

    # Test set for classes 1-9 (retain accuracy evaluation)
    test_retain_idx = np.where(targets != FORGET_CLASS)[0].tolist()
    test_retain_loader = DataLoader(Subset(test_set, test_retain_idx),
                                    batch_size=256, num_workers=2)

    return relearn_set, eval_set, retain_set, test_retain_loader


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
@torch.no_grad()
def accuracy_on_loader(model, loader):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

# ──────────────────────────────────────────────
# Relearning attack
# ──────────────────────────────────────────────
def run_relearn_attack(model, relearn_set, eval_set, retain_set, test_retain_loader,
                       epochs=5, steps_per_epoch=50, lr=0.001):
    """
    Attack: fine-tune model on 100 class-0 images mixed with 1000 retain images.
    Returns list of (forget_acc, retain_acc) per epoch (including epoch 0 = before attack).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    eval_loader = DataLoader(eval_set, batch_size=256, num_workers=2)

    # Create iterators for cycling through data
    relearn_loader = DataLoader(relearn_set, batch_size=16, shuffle=True,
                                num_workers=0, drop_last=False)
    retain_loader = DataLoader(retain_set, batch_size=16, shuffle=True,
                               num_workers=0, drop_last=False)

    results = []

    # Epoch 0: before attack
    f_acc = accuracy_on_loader(model, eval_loader)
    r_acc = accuracy_on_loader(model, test_retain_loader)
    results.append((f_acc, r_acc))

    for epoch in range(1, epochs + 1):
        relearn_iter = iter(relearn_loader)
        retain_iter = iter(retain_loader)

        model.train()
        for step in range(1, steps_per_epoch + 1):
            # Get relearn batch (cycle)
            try:
                re_images, re_labels = next(relearn_iter)
            except StopIteration:
                relearn_iter = iter(relearn_loader)
                re_images, re_labels = next(relearn_iter)

            # Get retain batch (cycle)
            try:
                rt_images, rt_labels = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rt_images, rt_labels = next(retain_iter)

            # Concatenate
            images = torch.cat([re_images, rt_images], dim=0).to(DEVICE)
            labels = torch.cat([re_labels, rt_labels], dim=0).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate after epoch
        f_acc = accuracy_on_loader(model, eval_loader)
        r_acc = accuracy_on_loader(model, test_retain_loader)
        results.append((f_acc, r_acc))

    return results

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print("\n" + "=" * 50)
    print("RELEARNING ATTACK")
    print("=" * 50)
    print("Attacker has 100 class-0 test images. Measuring recovery over 5 epochs.\n")

    relearn_set, eval_set, retain_set, test_retain_loader = get_attack_data()
    print(f"Relearn set: {len(relearn_set)} | Eval set: {len(eval_set)} | Retain set: {len(retain_set)}")

    all_results = {}

    for method, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"\n--- {method} ---")
            print(f"  Checkpoint not found: {ckpt_path}. Skipping.")
            continue

        print(f"\n--- {method} ---")
        # Re-seed for reproducibility per method
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        model = make_resnet18_cifar10().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

        results = run_relearn_attack(model, relearn_set, eval_set, retain_set, test_retain_loader)

        for i, (f_acc, r_acc) in enumerate(results):
            label = "Epoch 0 (before attack)" if i == 0 else f"Epoch {i}                 "
            print(f"{label} | Forget Acc: {f_acc:5.1f}% | Retain Acc: {r_acc:5.1f}%")

        all_results[method] = results
        del model
        torch.cuda.empty_cache()

    # ─── Summary Table ───
    print("\n" + "=" * 60)
    print("RELEARNING SUMMARY (after 5 epochs)")
    print("=" * 60)
    print(f"{'Method':<12}| {'F-Acc Before':>12} | {'F-Acc After':>11} | {'Recovery':>10} | {'Retain After':>12}")
    print("-" * 68)
    for method, results in all_results.items():
        f_before = results[0][0]
        f_after = results[-1][0]
        delta = f_after - f_before
        r_after = results[-1][1]
        print(f"{method:<12}| {f_before:11.1f}% | {f_after:10.1f}% | {delta:+9.1f}% | {r_after:11.1f}%")
    print("=" * 68)
    print("\nLower Recovery (Δ) = better unlearning. Retrain is the gold standard.")


if __name__ == "__main__":
    main()
