"""
robustness_vit.py — GPM-A vs GA robustness over increasing unlearning steps (ViT-Small).

Same experiment structure as robustness.py but on a Vision Transformer.
Tests whether GPM-A's activation-space projection generalizes from ResNets to ViTs.

  GPM-A: Activation-space projection (backward hooks on transformer block outputs)
  GA:    Plain gradient ascent (no projection, baseline)

Hooks are placed on model.blocks[i] for i in range(6) instead of model.layer1-4.
Activation shape: (B, 65, 256) flattened to (B, 16640) for SVD and projection.

Usage: python scripts/robustness_vit.py
Requires: checkpoints/vit_small_cifar10.pt (run train_vit.py first)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

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
CHECKPOINT_PATH = "checkpoints/vit_small_cifar10.pt"


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


def load_trained_model():
    model = ViTSmallCIFAR10().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    return model


# ──────────────────────────────────────────────
# Data helpers
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


def get_forget_retain_datasets():
    """Split CIFAR-10 TRAINING set into forget (class 0) and retain (classes 1-9)."""
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=get_train_transform()
    )
    targets = np.array(train_set.targets)
    forget_idx = np.where(targets == FORGET_CLASS)[0].tolist()
    retain_idx = np.where(targets != FORGET_CLASS)[0].tolist()
    return Subset(train_set, forget_idx), Subset(train_set, retain_idx)


def get_test_set():
    return torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=get_test_transform()
    )


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


@torch.no_grad()
def compute_final_metrics(model, forget_dataset, retain_dataset, test_set):
    """Full-set evaluation for final summary table."""
    train_set_test = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)
    forget_idx = np.where(targets == FORGET_CLASS)[0].tolist()
    retain_idx = np.where(targets != FORGET_CLASS)[0].tolist()

    forget_loader = DataLoader(Subset(train_set_test, forget_idx), batch_size=256, num_workers=2)
    retain_loader = DataLoader(Subset(train_set_test, retain_idx), batch_size=256, num_workers=2)

    # Test acc on classes 1-9 only
    test_targets = np.array(test_set.targets)
    test_retain_idx = np.where(test_targets != FORGET_CLASS)[0].tolist()
    test_retain_loader = DataLoader(Subset(test_set, test_retain_idx), batch_size=256, num_workers=2)

    forget_acc = accuracy_on_loader(model, forget_loader)
    retain_acc = accuracy_on_loader(model, retain_loader)
    test_acc = accuracy_on_loader(model, test_retain_loader)
    return forget_acc, retain_acc, test_acc


# ──────────────────────────────────────────────
# GPM-A: Build activation-space projection bases
# ──────────────────────────────────────────────
def build_projection_matrices(model, samples_per_class=500, threshold=0.95):
    """
    Build per-block U_retain: retain subspace basis from transformer block outputs.
    Activations: (B, 65, 256) → flatten to (B, 16640) → SVD.
    """
    print("  Building activation-space projection matrices...")

    train_set_test = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)

    # Sample retain data (500 per class, classes 1-9) → 4500 total
    retain_indices = []
    rng = np.random.RandomState(SEED)
    for c in range(10):
        if c == FORGET_CLASS:
            continue
        class_idx = np.where(targets == c)[0]
        chosen = rng.choice(class_idx, min(samples_per_class, len(class_idx)), replace=False)
        retain_indices.extend(chosen.tolist())

    num_blocks = len(model.blocks)
    block_names = [f"block_{i}" for i in range(num_blocks)]

    # Collect activations (block outputs) for retain data
    all_acts = {name: [] for name in block_names}
    hooks = []

    def hook_fn(name):
        def hook(module, inp, out):
            all_acts[name].append(out.detach().cpu())
        return hook

    for i, blk in enumerate(model.blocks):
        hooks.append(blk.register_forward_hook(hook_fn(block_names[i])))

    sample_set = Subset(train_set_test, retain_indices)
    sample_loader = DataLoader(sample_set, batch_size=256, shuffle=False, num_workers=2)

    model.eval()
    with torch.no_grad():
        for images, _ in sample_loader:
            images = images.to(DEVICE)
            model(images)

    for h in hooks:
        h.remove()

    # Build retain subspace per block via SVD
    projection_bases = {}

    for name in block_names:
        A = torch.cat(all_acts[name], dim=0)  # (N, 65, 256)
        A_flat = A.reshape(A.shape[0], -1).float()  # (N, 16640)

        U, S, _ = torch.linalg.svd(A_flat.T, full_matrices=False)
        explained = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
        k = (explained < threshold).sum().item() + 1
        k = min(k, U.shape[1])
        U_retain = U[:, :k]  # (16640, k)

        projection_bases[name] = U_retain.to(DEVICE)
        print(f"    {name}: retain_dims={k}/{U.shape[1]}, act_dim={A_flat.shape[1]}")

    return projection_bases


# ──────────────────────────────────────────────
# GPM-A unlearning: Activation-space projected gradient ascent
# ──────────────────────────────────────────────
def run_gpm_unlearning(model, forget_loader, projection_bases, steps, lr):
    """
    Projected gradient ascent via backward hooks on transformer blocks.
    grad_output[0] shape: (B, 65, 256) → flatten → project → reshape back.
    """
    backward_hooks = []
    num_blocks = len(model.blocks)
    block_names = [f"block_{i}" for i in range(num_blocks)]

    def make_proj_backward_hook(U_k):
        def hook(module, grad_output):
            g = grad_output[0]  # (B, 65, 256)
            shape = g.shape
            B = g.shape[0]
            g_flat = g.reshape(B, -1)           # (B, 16640)
            g_proj = g_flat - (g_flat @ U_k) @ U_k.T
            return (g_proj.reshape(shape),)
        return hook

    for i, blk in enumerate(model.blocks):
        name = block_names[i]
        backward_hooks.append(blk.register_full_backward_pre_hook(
            make_proj_backward_hook(projection_bases[name])))

    criterion = nn.CrossEntropyLoss()
    forget_iter = iter(forget_loader)

    for step in range(steps):
        try:
            images, labels = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            images, labels = next(forget_iter)

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model.train()
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()  # Backward hooks project gradients automatically

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data += lr * p.grad  # Ascent with projected gradients

    # Clean up hooks
    for h in backward_hooks:
        h.remove()


# ──────────────────────────────────────────────
# GA unlearning: Plain gradient ascent
# ──────────────────────────────────────────────
def run_ga_unlearning(model, forget_loader, steps, lr):
    """Plain gradient ascent on forget data. No projection. No retain data."""
    criterion = nn.CrossEntropyLoss()
    forget_iter = iter(forget_loader)

    for step in range(steps):
        try:
            images, labels = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            images, labels = next(forget_iter)

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model.train()
        model.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data += lr * p.grad  # Ascent (no projection)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Run train_vit.py first: python scripts/train_vit.py")
        return

    forget_dataset, retain_dataset = get_forget_retain_datasets()
    test_set = get_test_set()
    print(f"Forget set: {len(forget_dataset)} samples | Retain set: {len(retain_dataset)} samples")

    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True,
                               num_workers=2, drop_last=True)

    # Verify baseline accuracy before unlearning
    print("\nBaseline (pre-unlearning) accuracy:")
    model_baseline = load_trained_model()
    f_acc, r_acc, t_acc = compute_final_metrics(model_baseline, forget_dataset, retain_dataset, test_set)
    print(f"  Forget: {f_acc:.1f}% | Retain: {r_acc:.1f}% | Test: {t_acc:.1f}%")
    del model_baseline
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # Build projection matrices ONCE from original checkpoint
    # ──────────────────────────────────────────────
    step_counts = [50, 100, 200, 400, 800]
    lr = 1e-3

    print("\nBuilding projection matrices (one-time cost)...")
    model_for_svd = load_trained_model()
    act_bases = build_projection_matrices(model_for_svd)
    del model_for_svd
    torch.cuda.empty_cache()

    results_gpm_a = []
    results_ga = []

    for steps in step_counts:
        print(f"\n{'='*60}")
        print(f"  Steps = {steps}")
        print(f"{'='*60}")

        # --- GPM-A (activation-space) ---
        print(f"\n  [GPM-A] Running {steps} steps of activation-projected gradient ascent...")
        model = load_trained_model()
        run_gpm_unlearning(model, forget_loader, act_bases, steps=steps, lr=lr)
        f_acc, r_acc, t_acc = compute_final_metrics(model, forget_dataset, retain_dataset, test_set)
        results_gpm_a.append((steps, f_acc, r_acc, t_acc))
        print(f"  [GPM-A] Steps={steps:4d} | Forget: {f_acc:5.1f}% | Retain: {r_acc:5.1f}% | Test: {t_acc:5.1f}%")
        del model
        torch.cuda.empty_cache()

        # --- GA (baseline) ---
        print(f"\n  [GA]  Running {steps} steps of plain gradient ascent...")
        model = load_trained_model()
        run_ga_unlearning(model, forget_loader, steps=steps, lr=lr)
        f_acc, r_acc, t_acc = compute_final_metrics(model, forget_dataset, retain_dataset, test_set)
        results_ga.append((steps, f_acc, r_acc, t_acc))
        print(f"  [GA]  Steps={steps:4d} | Forget: {f_acc:5.1f}% | Retain: {r_acc:5.1f}% | Test: {t_acc:5.1f}%")
        del model
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════
    #                 RESULTS TABLES
    # ══════════════════════════════════════════════

    # Table 1: GPM-A vs GA
    print()
    w = 70
    print("=" * w)
    print("ViT-Small CIFAR-10: GPM-A vs GA Robustness")
    print("=" * w)
    print(f"         |    GPM-A                |    GA (baseline)")
    print(f"Steps    | Forget | Retain | Test  | Forget | Retain | Test")
    print(f"---------|--------|--------|-------|--------|--------|------")
    for (s, fa, ra, ta), (_, fg, rg, tg) in zip(results_gpm_a, results_ga):
        print(f"  {s:5d} | {fa:5.1f}  | {ra:5.1f}  | {ta:4.1f}  "
              f"| {fg:5.1f}  | {rg:5.1f}  | {tg:4.1f}")
    print("=" * w)

    # Table 2: Retain gap
    print()
    print(f"Steps    | GPM-A Ret | GA Ret  | Gap")
    print(f"---------|----------|---------|------")
    for (s, _, ra, _), (_, _, rg, _) in zip(results_gpm_a, results_ga):
        gap = ra - rg
        print(f"  {s:5d} | {ra:7.1f}  | {rg:6.1f}  | {gap:+6.1f}")
    print("=" * w)


if __name__ == "__main__":
    main()
