git remote add origin https://github.com/aaryamam/RASE.git
git branch -M main
git push -u origin main"""
evaluate.py — Run MIA and linear probing on unlearned model checkpoints.

Loads checkpoints produced by unlearn.py.
Configuration: change MODEL_ARCH at the top to match the setting used in unlearn.py.
Usage: python scripts/evaluate.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
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

# ══════════════════════════════════════════════
#  CONFIGURATION — match settings used in unlearn.py
# ══════════════════════════════════════════════
MODEL_ARCH   = "resnet18"   # "resnet18" or "vit_small"
FORGET_CLASS = 0

CHECKPOINT_DIR = f"checkpoints/unlearned/{MODEL_ARCH}_forget{FORGET_CLASS}"

# Checkpoints to evaluate (add/remove as needed)
CHECKPOINTS = {
    "Original":  f"checkpoints/{MODEL_ARCH}_cifar10.pt",
    "GA":        f"{CHECKPOINT_DIR}/ga.pt",
    "GPM-W":     f"{CHECKPOINT_DIR}/gpm_w.pt",   # skip if not present
    "RASE":      f"{CHECKPOINT_DIR}/rase.pt",
}

# Feature dims for linear probe
_FEATURE_DIM = {"resnet18": 512, "vit_small": 256}[MODEL_ARCH]


# ──────────────────────────────────────────────
# Model definitions (both architectures)
# ──────────────────────────────────────────────
def make_resnet18_cifar10():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


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

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
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
# Model loading
# ──────────────────────────────────────────────
def load_model_from_path(ckpt_path):
    if MODEL_ARCH == "resnet18":
        model = make_resnet18_cifar10().to(DEVICE)
    elif MODEL_ARCH == "vit_small":
        model = ViTSmallCIFAR10().to(DEVICE)
    else:
        raise ValueError(f"Unknown MODEL_ARCH: {MODEL_ARCH}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ──────────────────────────────────────────────
# Evaluation 1: MIA (Membership Inference Attack)
# ──────────────────────────────────────────────
@torch.no_grad()
def compute_mia(model, forget_class):
    """
    Loss-based MIA with balanced accuracy.

    Members:     forget-class samples from TRAINING set
    Non-members: forget-class samples from TEST set

    Sweep thresholds on loss to find the best balanced accuracy:
        balanced_acc = (TPR + TNR) / 2
    Check both directions (low loss = member, high loss = member).

    Returns:
        mia_balanced_acc: float (50.0 = chance = perfect unlearning)
        avg_member_loss:  float
        avg_nonmember_loss: float
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    test_transform = get_test_transform()

    # 1. MEMBER set: forget-class examples from TRAINING set
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=test_transform
    )
    train_targets = np.array(train_set.targets)
    member_indices = np.where(train_targets == forget_class)[0].tolist()
    member_loader = DataLoader(Subset(train_set, member_indices),
                               batch_size=256, shuffle=False, num_workers=2)

    # 2. NON-MEMBER set: forget-class examples from TEST set
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=test_transform
    )
    test_targets = np.array(test_set.targets)
    nonmember_indices = np.where(test_targets == forget_class)[0].tolist()
    nonmember_loader = DataLoader(Subset(test_set, nonmember_indices),
                                  batch_size=256, shuffle=False, num_workers=2)

    # 3. Compute per-sample loss for both groups
    member_losses = []
    for images, labels in member_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        losses = criterion(model(images), labels)
        member_losses.extend(losses.cpu().tolist())

    nonmember_losses = []
    for images, labels in nonmember_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        losses = criterion(model(images), labels)
        nonmember_losses.extend(losses.cpu().tolist())

    # 4. Binary classification: find optimal threshold (balanced accuracy)
    all_losses = np.array(member_losses + nonmember_losses)
    all_labels = np.array([1] * len(member_losses) + [0] * len(nonmember_losses))

    n_members = np.sum(all_labels == 1)
    n_nonmembers = np.sum(all_labels == 0)

    best_acc = 50.0
    for thr in np.unique(all_losses):
        # Strategy: predict member if loss <= threshold
        preds = (all_losses <= thr).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        tpr = tp / n_members
        tnr = tn / n_nonmembers
        balanced_acc = 100.0 * (tpr + tnr) / 2.0
        if balanced_acc > best_acc:
            best_acc = balanced_acc
        # Reversed strategy: predict member if loss > threshold
        balanced_acc_rev = 100.0 * ((1 - tpr) + (1 - tnr)) / 2.0
        if balanced_acc_rev > best_acc:
            best_acc = balanced_acc_rev

    avg_member = float(np.mean(member_losses))
    avg_nonmember = float(np.mean(nonmember_losses))

    return best_acc, avg_member, avg_nonmember


# ──────────────────────────────────────────────
# Evaluation 2: Linear Probing
# ──────────────────────────────────────────────
@torch.no_grad()
def _extract_features(model, loader):
    """
    Extract backbone features from a frozen model.

    ResNet-18: hook on model.avgpool → flatten → 512-dim vector.
    ViT-Small: hook on model.norm → take output[:, 0, :] (CLS token) → 256-dim vector.
    """
    features = []
    labels_all = []

    hook_outputs = []

    if MODEL_ARCH == "resnet18":
        def hook_fn(module, inp, out):
            hook_outputs.append(out.detach().cpu().flatten(1))
        hook = model.avgpool.register_forward_hook(hook_fn)
    else:  # vit_small
        def hook_fn(module, inp, out):
            hook_outputs.append(out[:, 0, :].detach().cpu())
        hook = model.norm.register_forward_hook(hook_fn)

    model.eval()
    for images, lbls in loader:
        images = images.to(DEVICE)
        hook_outputs.clear()
        model(images)
        features.append(hook_outputs[0])
        labels_all.append(lbls)

    hook.remove()

    return torch.cat(features, dim=0), torch.cat(labels_all, dim=0)


def linear_probe(model, forget_class, epochs=20, lr=0.01):
    """
    1. Extract features from the frozen backbone for all training samples.
    2. Train a fresh nn.Linear(feature_dim, num_classes) on retain-class training features.
    3. Evaluate on forget-class test samples and retain-class test samples.

    Returns:
        forget_probe_acc: accuracy of linear probe on forget-class test samples
        retain_probe_acc: accuracy of linear probe on retain-class test samples
    """
    test_transform = get_test_transform()

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=test_transform
    )
    test_set_full = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=test_transform
    )

    train_targets = np.array(train_set.targets)
    test_targets = np.array(test_set_full.targets)

    retain_train_idx = np.where(train_targets != forget_class)[0].tolist()
    forget_test_idx = np.where(test_targets == forget_class)[0].tolist()
    retain_test_idx = np.where(test_targets != forget_class)[0].tolist()

    retain_train_loader = DataLoader(
        Subset(train_set, retain_train_idx), batch_size=256, shuffle=False, num_workers=2
    )
    forget_test_loader = DataLoader(
        Subset(test_set_full, forget_test_idx), batch_size=256, shuffle=False, num_workers=2
    )
    retain_test_loader = DataLoader(
        Subset(test_set_full, retain_test_idx), batch_size=256, shuffle=False, num_workers=2
    )

    # Feature extraction (no gradients, frozen backbone)
    model.eval()
    with torch.no_grad():
        train_feats, train_labels = _extract_features(model, retain_train_loader)
        forget_feats, forget_labels = _extract_features(model, forget_test_loader)
        retain_feats, retain_labels = _extract_features(model, retain_test_loader)

    # Train a fresh linear head on retain-class training features only
    feature_dim = _FEATURE_DIM
    num_classes = 10
    linear_head = nn.Linear(feature_dim, num_classes).to(DEVICE)
    optimizer = optim.SGD(linear_head.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    linear_head.train()
    for _ in range(epochs):
        for feats, lbls in train_loader:
            feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(linear_head(feats), lbls)
            loss.backward()
            optimizer.step()

    # Evaluate on forget-class and retain-class test samples
    linear_head.eval()
    with torch.no_grad():
        preds_forget = linear_head(forget_feats.to(DEVICE)).argmax(1).cpu()
        forget_probe_acc = 100.0 * (preds_forget == forget_labels).float().mean().item()

        preds_retain = linear_head(retain_feats.to(DEVICE)).argmax(1).cpu()
        retain_probe_acc = 100.0 * (preds_retain == retain_labels).float().mean().item()

    return forget_probe_acc, retain_probe_acc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Model: {MODEL_ARCH}  |  Forget class: {FORGET_CLASS}")
    print(f"Loading checkpoints from: {CHECKPOINT_DIR}\n")

    mia_results   = {}
    probe_results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"[{name}] checkpoint not found at {ckpt_path} — skipping.")
            continue

        print(f"{'=' * 55}")
        print(f"Evaluating: {name}")
        model = load_model_from_path(ckpt_path)

        # MIA
        mia_acc, mem_loss, nonmem_loss = compute_mia(model, FORGET_CLASS)
        mia_results[name] = (mia_acc, mem_loss, nonmem_loss)
        print(f"  MIA balanced acc: {mia_acc:.1f}%  "
              f"(member loss: {mem_loss:.4f}, non-member loss: {nonmem_loss:.4f})")

        # Linear probe
        forget_probe, retain_probe = linear_probe(model, FORGET_CLASS)
        probe_results[name] = (forget_probe, retain_probe)
        print(f"  Linear probe  —  Forget class: {forget_probe:.1f}%  "
              f"Retain classes: {retain_probe:.1f}%")

        del model
        torch.cuda.empty_cache()

    # ─────────────────────────────────
    # Summary tables
    # ─────────────────────────────────
    print("\n" + "=" * 65)
    print("MIA SUMMARY (50% = chance = perfect unlearning)")
    print("=" * 65)
    print(f"{'Method':<12} | {'MIA Acc':>9} | {'Member Loss':>12} | {'Non-Mem Loss':>13}")
    print("-" * 65)
    for name, (mia, ml, nml) in mia_results.items():
        print(f"{name:<12} | {mia:8.1f}%  | {ml:12.4f} | {nml:13.4f}")
    print("=" * 65)

    print("\n" + "=" * 55)
    print("LINEAR PROBE SUMMARY")
    print("=" * 55)
    print(f"{'Method':<12} | {'Forget Probe':>13} | {'Retain Probe':>13}")
    print("-" * 55)
    for name, (fp, rp) in probe_results.items():
        print(f"{name:<12} | {fp:12.1f}%  | {rp:12.1f}%")
    print("=" * 55)
    print("\nInterpretation:")
    print("  MIA ~50%:    forget-class samples indistinguishable from non-members")
    print("  Low forget probe acc: forget-class representations erased from backbone")


if __name__ == "__main__":
    main()
