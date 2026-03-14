"""
unlearn.py — Run GA, GPM-W, RASE, and SCRUB unlearning on a chosen model/checkpoint.

Configuration: change MODEL_ARCH at the top to switch between resnet18 / vit_small.

Usage:
  python scripts/unlearn.py                          # run all methods
  python scripts/unlearn.py --methods ga rase        # run only GA and RASE
  python scripts/unlearn.py --methods scrub          # run only SCRUB
  python scripts/unlearn.py --methods ga gpm_w rase scrub
"""

import os
import time
import copy
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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
#  CONFIGURATION — change these to switch model
# ══════════════════════════════════════════════
MODEL_ARCH      = "resnet50_tinyimagenet"          # "resnet18", "vit_small", or "resnet50_tinyimagenet"
FORGET_CLASSES  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # class index to forget (0-9 for CIFAR, 0-199 for TinyImageNet)
UNLEARN_STEPS   = 200                 # gradient ascent steps
UNLEARN_LR      = 1e-3               # learning rate for all methods
SVD_THRESHOLD   = 0.95           # variance threshold for subspace construction

NUM_CLASSES = 200 if MODEL_ARCH == "resnet50_tinyimagenet" else 10
SAMPLES_PER_CLASS = 20  # ~500 for CIFAR, 20 for Tiny-ImageNet (4000 total)
PRINT_EVERY     = 10    # print progress every N steps

TINY_MEAN = (0.485, 0.456, 0.406)
TINY_STD  = (0.229, 0.224, 0.225)

# ViT-specific RASE configuration
VIT_RASE_HOOK_BLOCKS = [0, 1, 2, 3, 4, 5]
VIT_RASE_USE_ADAM    = False

# SCRUB-specific
SCRUB_RETAIN_STEPS  = 3
SCRUB_KL_WEIGHT     = 1.0
SCRUB_RETAIN_BATCH  = 128

CHECKPOINT_PATH = {
    "resnet18":  "checkpoints/resnet18_cifar10.pt",
    "vit_small": "checkpoints/vit_small_cifar10_best.pt",
    "resnet50_tinyimagenet": "checkpoints/resnet50_tinyimagenet_best.pt",
}[MODEL_ARCH]

OUTPUT_DIR = f"checkpoints/unlearned/{MODEL_ARCH}_forget{'_'.join(map(str, FORGET_CLASSES))}"


# ──────────────────────────────────────────────
# Model definitions (both architectures)
# ──────────────────────────────────────────────
from torchvision.models import resnet50

def make_resnet50_tinyimagenet():
    model = resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(2048, 200)
    return model

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
# Model loading
# ──────────────────────────────────────────────
def load_model():
    if MODEL_ARCH == "resnet18":
        model = make_resnet18_cifar10().to(DEVICE)
    elif MODEL_ARCH == "vit_small":
        model = ViTSmallCIFAR10().to(DEVICE)
    elif MODEL_ARCH == "resnet50_tinyimagenet":
        model = make_resnet50_tinyimagenet().to(DEVICE)
    else:
        raise ValueError(f"Unknown MODEL_ARCH: {MODEL_ARCH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    return model


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
import urllib.request
import zipfile
import shutil

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
        
        if os.path.exists(val_images_dir):
            shutil.rmtree(val_images_dir)
    
    return extract_dir

def get_dataset(split, transform=None):
    if MODEL_ARCH == "resnet50_tinyimagenet":
        data_dir = download_and_prepare_tiny_imagenet("./data")
        split_dir = "train" if split == "train" else "val"
        return torchvision.datasets.ImageFolder(os.path.join(data_dir, split_dir), transform=transform)
    else:
        is_train = (split == "train")
        return torchvision.datasets.CIFAR10(root="./data", train=is_train, download=True, transform=transform)

def get_train_transform():
    if MODEL_ARCH == "resnet50_tinyimagenet":
        return transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ])
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_test_transform():
    if MODEL_ARCH == "resnet50_tinyimagenet":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_forget_retain_loaders():
    """
    Returns forget_loader, forget_dataset, retain_dataset.
    forget_loader uses augmented train transforms.
    """
    train_set = get_dataset("train", transform=get_train_transform())
    targets = np.array(train_set.targets)
    forget_idx = np.where(np.isin(targets, FORGET_CLASSES))[0].tolist()
    retain_idx = np.where(~np.isin(targets, FORGET_CLASSES))[0].tolist()
    forget_dataset = Subset(train_set, forget_idx)
    retain_dataset = Subset(train_set, retain_idx)
    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True,
                               num_workers=2, drop_last=True)
    return forget_loader, forget_dataset, retain_dataset


def get_test_set():
    return get_dataset("test", transform=get_test_transform())


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
def compute_final_metrics(model, forget_classes, test_set):
    """
    Full evaluation:
      - Forget acc: train samples of forget_classes (test transform, no augmentation)
      - Retain acc: train samples of all other classes
      - Test acc:   test set, retained classes only
    """
    train_set_test = get_dataset("train", transform=get_test_transform())
    targets = np.array(train_set_test.targets)
    forget_idx = np.where(np.isin(targets, forget_classes))[0].tolist()
    retain_idx = np.where(~np.isin(targets, forget_classes))[0].tolist()

    forget_loader = DataLoader(Subset(train_set_test, forget_idx), batch_size=256, num_workers=2)
    retain_loader = DataLoader(Subset(train_set_test, retain_idx), batch_size=256, num_workers=2)

    test_targets = np.array(test_set.targets)
    test_retain_idx = np.where(~np.isin(test_targets, forget_classes))[0].tolist()
    test_retain_loader = DataLoader(Subset(test_set, test_retain_idx), batch_size=256, num_workers=2)

    forget_acc = accuracy_on_loader(model, forget_loader)
    retain_acc = accuracy_on_loader(model, retain_loader)
    test_acc = accuracy_on_loader(model, test_retain_loader)
    return forget_acc, retain_acc, test_acc


# ──────────────────────────────────────────────
# Method 1: GA (Gradient Ascent)
# ──────────────────────────────────────────────
def unlearn_ga(model, forget_loader, steps, lr,
               eval_forget_loader=None, eval_retain_loader=None):
    """Plain gradient ascent on forget data. No projection. No retain data."""
    criterion = nn.CrossEntropyLoss()
    forget_iter = iter(forget_loader)

    for step in range(1, steps + 1):
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
                    p.data += lr * p.grad  # Ascent

        if step % PRINT_EVERY == 0:
            if eval_forget_loader is not None and eval_retain_loader is not None:
                fa = accuracy_on_loader(model, eval_forget_loader)
                ra = accuracy_on_loader(model, eval_retain_loader)
                print(f"[GA] Step {step}/{steps} | Loss: {loss.item():.4f} | "
                      f"Forget: {fa:.1f}% | Retain: {ra:.1f}%")
            else:
                print(f"[GA] Step {step}/{steps} | Loss: {loss.item():.4f}")


# ──────────────────────────────────────────────
# Method 2: GPM-W (Weight-space projection — ResNet only)
# ──────────────────────────────────────────────
def build_full_weight_space_bases(model, retain_dataset, samples_per_class=SAMPLES_PER_CLASS,
                                  threshold=SVD_THRESHOLD, max_patches=10000):
    """
    Build projection bases for EVERY Conv2d (except conv1) and the FC layer.
    Hooks capture the INPUT to each Conv2d; F.unfold extracts patches.

    Returns:
        projection_bases: dict mapping layer name -> U_k tensor (patch_dim, k)
        named_convs: list of (name, module) for all projected convs
    """
    print("  Building full weight-space projection matrices (every Conv2d)...")

    train_set_test = get_dataset("train", transform=get_test_transform())
    targets = np.array(train_set_test.targets)

    # Sample retain data (SAMPLES_PER_CLASS per class, all classes except FORGET_CLASSES)
    retain_indices = []
    rng = np.random.RandomState(SEED)
    for c in range(NUM_CLASSES):
        if c in FORGET_CLASSES:
            continue
        class_idx = np.where(targets == c)[0]
        chosen = rng.choice(class_idx, min(samples_per_class, len(class_idx)), replace=False)
        retain_indices.extend(chosen.tolist())

    sample_set = Subset(train_set_test, retain_indices)
    sample_loader = DataLoader(sample_set, batch_size=64, shuffle=False, num_workers=2)

    # Collect ALL Conv2d layers (except conv1) and the fc layer
    conv_inputs = {}
    hooks = []

    named_convs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name != "conv1":
            named_convs.append((name, module))

    for name, module in named_convs:
        conv_inputs[name] = []

        def make_hook(n):
            def hook(mod, inp, out):
                conv_inputs[n].append(inp[0].detach().cpu())
            return hook

        hooks.append(module.register_forward_hook(make_hook(name)))

    # Also capture fc input
    conv_inputs["fc"] = []

    def fc_hook(mod, inp, out):
        conv_inputs["fc"].append(inp[0].detach().cpu())

    hooks.append(model.fc.register_forward_hook(fc_hook))

    model.eval()
    with torch.no_grad():
        for images, _ in sample_loader:
            images = images.to(DEVICE)
            model(images)

    for h in hooks:
        h.remove()

    # Build projection basis for each conv
    projection_bases = {}

    for name, module in named_convs:
        all_inputs = torch.cat(conv_inputs[name], dim=0)  # (N, C_in, H, W)

        kernel_size = module.kernel_size
        padding = module.padding
        stride = module.stride

        patches = F.unfold(all_inputs, kernel_size=kernel_size,
                           padding=padding, stride=stride)
        # patches: (N, C_in*kH*kW, L)

        patch_dim = patches.shape[1]
        patches = patches.permute(0, 2, 1).reshape(-1, patch_dim).float()

        if patches.shape[0] > max_patches:
            idx = rng.choice(patches.shape[0], max_patches, replace=False)
            patches = patches[idx]

        U, S, _ = torch.linalg.svd(patches.T, full_matrices=False)
        explained = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
        k = (explained < threshold).sum().item() + 1
        k = min(k, U.shape[1])
        U_k = U[:, :k]

        projection_bases[name] = U_k.to(DEVICE)
        print(f"    {name}: patch_dim={patch_dim}, kept {k}, "
              f"weight_shape={tuple(module.weight.shape)}")

    # FC layer
    fc_inputs = torch.cat(conv_inputs["fc"], dim=0).float()
    U_fc, S_fc, _ = torch.linalg.svd(fc_inputs.T, full_matrices=False)
    explained_fc = torch.cumsum(S_fc ** 2, dim=0) / (S_fc ** 2).sum()
    k_fc = (explained_fc < threshold).sum().item() + 1
    k_fc = min(k_fc, U_fc.shape[1])
    projection_bases["fc"] = U_fc[:, :k_fc].to(DEVICE)
    print(f"    fc: input_dim={fc_inputs.shape[1]}, kept {k_fc}")

    return projection_bases, named_convs


def unlearn_gpm_w(model, forget_loader, projection_bases, named_convs, steps, lr):
    """
    Weight-space GPM with every conv layer projected.
    Normal backward, then project each conv's weight.grad, then apply ascent.
    """
    criterion = nn.CrossEntropyLoss()
    forget_iter = iter(forget_loader)
    f_loader_small, r_loader_small = _make_small_eval_loaders()

    # Re-find modules by name in this model instance
    # (named_convs references the SVD model's modules)
    conv_lookup = {}
    for name, _ in named_convs:
        mod = model
        for p in name.split("."):
            mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
        conv_lookup[name] = mod

    for step in range(1, steps + 1):
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
            # Project each conv's weight gradient
            for name, _ in named_convs:
                conv_module = conv_lookup[name]
                if conv_module.weight.grad is None:
                    continue

                U_k = projection_bases[name]
                W_grad = conv_module.weight.grad
                C_out = W_grad.shape[0]
                G = W_grad.reshape(C_out, -1)  # (C_out, patch_dim)

                if G.shape[1] == U_k.shape[0]:
                    G_proj = G - (G @ U_k) @ U_k.T
                    conv_module.weight.grad = G_proj.reshape(W_grad.shape)

            # Project fc weight gradient
            if model.fc.weight.grad is not None and "fc" in projection_bases:
                U_fc = projection_bases["fc"]
                G_fc = model.fc.weight.grad
                if G_fc.shape[1] == U_fc.shape[0]:
                    G_fc_proj = G_fc - (G_fc @ U_fc) @ U_fc.T
                    model.fc.weight.grad = G_fc_proj

            # Apply gradient ASCENT on all parameters
            for p in model.parameters():
                if p.grad is not None:
                    p.data += lr * p.grad

        if step % PRINT_EVERY == 0:
            fa = accuracy_on_loader(model, f_loader_small)
            ra = accuracy_on_loader(model, r_loader_small)
            print(f"[GPM-W] Step {step}/{steps} | Loss: {loss.item():.4f} | "
                  f"Forget: {fa:.1f}% | Retain: {ra:.1f}%")


# ──────────────────────────────────────────────
# Method 3: RASE (Activation-space projection)
# ──────────────────────────────────────────────
def build_rase_bases(model, samples_per_class=SAMPLES_PER_CLASS, threshold=SVD_THRESHOLD):
    """
    Build per-layer activation subspace bases for RASE.

    ResNet-18: hooks on layer1–layer4 block outputs, shape (N, C, H, W) → flatten.
    ViT-Small: hooks on model.blocks[0–5] outputs, shape (N, 65, 256) → flatten.
    """
    print(f"  Building RASE activation-space bases (arch={MODEL_ARCH})...")

    train_set_test = get_dataset("train", transform=get_test_transform())
    targets = np.array(train_set_test.targets)

    retain_indices = []
    rng = np.random.RandomState(SEED)
    for c in range(NUM_CLASSES):
        if c in FORGET_CLASSES:
            continue
        class_idx = np.where(targets == c)[0]
        chosen = rng.choice(class_idx, min(samples_per_class, len(class_idx)), replace=False)
        retain_indices.extend(chosen.tolist())

    # ResNet-50 Tiny-ImageNet acts have 1M+ dimensions. 4975 samples -> 40GB+ OOM.
    # Cap total samples to 500 to keep GPU memory for SVD bases < 10GB.
    if MODEL_ARCH == "resnet50_tinyimagenet" and len(retain_indices) > 4000:
        retain_indices = rng.choice(retain_indices, 4000, replace=False).tolist()

    sample_set = Subset(train_set_test, retain_indices)
    sample_loader = DataLoader(sample_set, batch_size=64, shuffle=False, num_workers=2)

    # Determine hook targets and names
    if "resnet" in MODEL_ARCH:
        hook_layers = [
            ("layer1", model.layer1),
            ("layer2", model.layer2),
            ("layer3", model.layer3),
            ("layer4", model.layer4),
        ]
    else:  # vit_small — only hook blocks in VIT_RASE_HOOK_BLOCKS
        hook_layers = [
            (f"block_{i}", model.blocks[i]) for i in VIT_RASE_HOOK_BLOCKS
        ]

    projection_bases = {}
    max_rank = 1500
    N = len(sample_set)

    model.eval()
    for name, layer in hook_layers:
        A_flat_cpu = None
        idx = [0]
        
        def hook(module, inp, out):
            nonlocal A_flat_cpu
            batch_acts = out.detach().cpu().reshape(out.shape[0], -1)
            if A_flat_cpu is None:
                D = batch_acts.shape[1]
                A_flat_cpu = torch.empty((N, D), dtype=torch.float32)
            B = batch_acts.shape[0]
            A_flat_cpu[idx[0]:idx[0]+B] = batch_acts
            idx[0] += B
            
        h = layer.register_forward_hook(hook)
        
        with torch.no_grad():
            for images, _ in sample_loader:
                images = images.to(DEVICE)
                model(images)
                
        h.remove()

        total_variance = (A_flat_cpu ** 2).sum().item()
        
        q = min(max_rank, A_flat_cpu.shape[0], A_flat_cpu.shape[1])
        
        try:
            U, S, _ = torch.svd_lowrank(A_flat_cpu.T, q=q)
        except Exception as e:
            print(f"    [WARNING] svd_lowrank failed: {e}. Falling back to standard SVD.")
            U, S, _ = torch.linalg.svd(A_flat_cpu.T, full_matrices=False)
            
        explained = torch.cumsum(S ** 2, dim=0) / total_variance
        
        k = (explained < threshold).sum().item() + 1
        
        if k > q:
            max_var = explained[-1].item() if len(explained) > 0 else 0
            print(f"    [WARNING] max_rank={q} only reached {max_var:.4f} variance (target {threshold}). Using all {q} components.")
            k = q

        U_retain = U[:, :k]

        projection_bases[name] = U_retain.to(DEVICE)
        print(f"    {name}: retain_dims={k}/{U.shape[1]}, act_dim={A_flat_cpu.shape[1]}")
        
        del A_flat_cpu, U, S, explained

    return projection_bases


def unlearn_rase(model, forget_loader, rase_bases, steps, lr):
    """
    Activation-space projected gradient ascent (RASE).
    Backward hooks project grad_output away from the retain activation subspace.

    ResNet-18: hooks on layer1–layer4, raw SGD ascent.
    ViT-Small: hooks on VIT_RASE_HOOK_BLOCKS only, Adam optimizer for ascent.
               (ViT's shared residual stream causes compound projection that kills
               gradients if all blocks are hooked; Adam's adaptive LR amplifies
               the tiny surviving projected gradients.)
    """
    backward_hooks = []
    f_loader_small, r_loader_small = _make_small_eval_loaders()

    def make_proj_backward_hook(U_k):
        def hook(module, grad_output):
            g = grad_output[0]  # (B, ...)
            shape = g.shape
            B = g.shape[0]
            g_flat = g.reshape(B, -1)
            g_proj = g_flat - (g_flat @ U_k) @ U_k.T
            return (g_proj.reshape(shape),)
        return hook

    if "resnet" in MODEL_ARCH:
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, layer_name)
            backward_hooks.append(layer.register_full_backward_pre_hook(
                make_proj_backward_hook(rase_bases[layer_name])))
    else:  # vit_small — only hook selected blocks
        for i in VIT_RASE_HOOK_BLOCKS:
            name = f"block_{i}"
            backward_hooks.append(model.blocks[i].register_full_backward_pre_hook(
                make_proj_backward_hook(rase_bases[name])))

    criterion = nn.CrossEntropyLoss()
    forget_iter = iter(forget_loader)

    # ViT: use Adam for gradient ascent (adaptive LR amplifies tiny projected grads)
    # ResNet: use raw SGD ascent (original, proven approach)
    use_adam = (MODEL_ARCH == "vit_small" and VIT_RASE_USE_ADAM)
    if use_adam:
        # Negate lr: Adam minimizes, but we want to MAXIMIZE loss (ascent)
        # So we use a NegCE wrapper or simply negate the loss
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        try:
            images, labels = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            images, labels = next(forget_iter)

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model.train()

        if use_adam:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            (-loss).backward()  # Negate: Adam minimizes, so minimize(-loss) = maximize(loss)
            optimizer.step()
        else:
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # Backward hooks project gradients automatically
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data += lr * p.grad  # Ascent with projected gradients

        if step % PRINT_EVERY == 0:
            fa = accuracy_on_loader(model, f_loader_small)
            ra = accuracy_on_loader(model, r_loader_small)
            print(f"[RASE] Step {step}/{steps} | Loss: {loss.item():.4f} | "
                  f"Forget: {fa:.1f}% | Retain: {ra:.1f}%")

    for h in backward_hooks:
        h.remove()


# ──────────────────────────────────────────────
# Method 4: SCRUB (Teacher-Student KL unlearning)
# ──────────────────────────────────────────────
def unlearn_scrub(model, forget_loader, retain_dataset, steps, lr,
                  eval_forget_loader=None, eval_retain_loader=None):
    """
    SCRUB: teacher-student unlearning.

    Each outer step:
      1. Gradient ASCENT on one forget batch (maximize CE loss on forget data).
      2. SCRUB_RETAIN_STEPS steps of gradient DESCENT minimizing KL divergence
         between student (model) and teacher (frozen copy of original model)
         on retain batches.

    The teacher is a frozen deep copy of the model as it was at the START of
    unlearning (i.e., the original trained checkpoint). It never updates.

    Loss:
      forget_loss  =  CE(student(x_f), y_f)           [ascent: maximize]
      retain_loss  =  KL(student(x_r) || teacher(x_r)) [descent: minimize]
      total retain update = SCRUB_KL_WEIGHT * retain_loss

    Both the forget ascent and retain descent use the same lr.
    """
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    criterion_ce = nn.CrossEntropyLoss()
    retain_loader = DataLoader(
        retain_dataset, batch_size=SCRUB_RETAIN_BATCH,
        shuffle=True, num_workers=2, drop_last=True
    )

    forget_iter  = iter(forget_loader)
    retain_iter  = iter(retain_loader)

    def next_batch(it, loader):
        try:
            return next(it), it
        except StopIteration:
            it = iter(loader)
            return next(it), it

    for step in range(1, steps + 1):
        model.train()

        # ── Step A: gradient ASCENT on forget batch ──
        (x_f, y_f), forget_iter = next_batch(forget_iter, forget_loader)
        x_f, y_f = x_f.to(DEVICE), y_f.to(DEVICE)
        model.zero_grad()
        loss_f = criterion_ce(model(x_f), y_f)
        loss_f.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.data += lr * p.grad   # ascent

        # ── Step B: SCRUB_RETAIN_STEPS of KL-descent on retain batches ──
        for _ in range(SCRUB_RETAIN_STEPS):
            (x_r, _), retain_iter = next_batch(retain_iter, retain_loader)
            x_r = x_r.to(DEVICE)
            model.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(x_r)
            student_logits = model(x_r)
            # KL(student || teacher): student is the "distribution" being updated
            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=1),
                F.softmax(teacher_logits, dim=1),
                reduction="batchmean",
            )
            retain_loss = SCRUB_KL_WEIGHT * kl_loss
            retain_loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data -= lr * p.grad   # descent

        # ── Progress print ──
        if step % PRINT_EVERY == 0 or step == 1:
            if eval_forget_loader is not None and eval_retain_loader is not None:
                fa = accuracy_on_loader(model, eval_forget_loader)
                ra = accuracy_on_loader(model, eval_retain_loader)
                print(f"  [SCRUB] Step {step:04d}/{steps} | "
                      f"ForgetLoss: {loss_f.item():.4f} | "
                      f"KL: {kl_loss.item():.4f} | "
                      f"Forget: {fa:.1f}% | Retain: {ra:.1f}%")
            else:
                print(f"  [SCRUB] Step {step:04d}/{steps} | "
                      f"ForgetLoss: {loss_f.item():.4f} | "
                      f"KL: {kl_loss.item():.4f}")


def _make_small_eval_loaders(n=512):
    """Make small subsets (512 samples each) for mid-training progress prints."""
    train_set_test = get_dataset("train", transform=get_test_transform())
    targets = np.array(train_set_test.targets)
    rng = np.random.RandomState(SEED)

    forget_idx = np.where(np.isin(targets, FORGET_CLASSES))[0]
    retain_idx = np.where(~np.isin(targets, FORGET_CLASSES))[0]

    f_chosen = rng.choice(forget_idx, min(n, len(forget_idx)), replace=False).tolist()
    r_chosen = rng.choice(retain_idx, min(n, len(retain_idx)), replace=False).tolist()

    f_loader = DataLoader(Subset(train_set_test, f_chosen), batch_size=256, num_workers=2)
    r_loader = DataLoader(Subset(train_set_test, r_chosen), batch_size=256, num_workers=2)
    return f_loader, r_loader


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Machine unlearning — run selected methods.")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["ga", "gpm_w", "rase", "scrub"],
        default=["ga", "gpm_w", "rase", "scrub"],
        help=(
            "Which unlearning methods to run. "
            "Options: ga  gpm_w  rase  scrub. "
            "Example: --methods ga rase scrub"
        ),
    )
    parser.add_argument(
        "--forget_classes",
        nargs="+",
        type=int,
        default=None,
        help="List of classes to forget (overrides global FORGET_CLASSES)",
    )
    args = parser.parse_args()
    run_methods = set(args.methods)
    
    if args.forget_classes is not None:
        global FORGET_CLASSES, OUTPUT_DIR
        FORGET_CLASSES = args.forget_classes
        OUTPUT_DIR = f"checkpoints/unlearned/{MODEL_ARCH}_forget{'_'.join(map(str, FORGET_CLASSES))}"

    print(f"Methods selected: {sorted(run_methods)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Model:        {MODEL_ARCH}")
    print(f"Forget classes: {FORGET_CLASSES}")
    print(f"Steps:        {UNLEARN_STEPS}  LR: {UNLEARN_LR}")
    print(f"Output dir:   {OUTPUT_DIR}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"Run the corresponding training script first.")
        return

    forget_loader, forget_dataset, retain_dataset = get_forget_retain_loaders()
    test_set = get_test_set()

    # Small eval subsets for mid-training progress prints
    rng_eval = np.random.RandomState(SEED)
    train_set_eval = get_dataset("train", transform=get_test_transform())
    all_targets = np.array(train_set_eval.targets)
    f_idx_eval = np.where(np.isin(all_targets, FORGET_CLASSES))[0]
    r_idx_eval = np.where(~np.isin(all_targets, FORGET_CLASSES))[0]
    eval_forget_subset = Subset(train_set_eval,
        rng_eval.choice(f_idx_eval, min(512, len(f_idx_eval)), replace=False).tolist())
    eval_retain_subset = Subset(train_set_eval,
        rng_eval.choice(r_idx_eval, min(512, len(r_idx_eval)), replace=False).tolist())
    eval_forget_loader = DataLoader(eval_forget_subset, batch_size=256, num_workers=2)
    eval_retain_loader = DataLoader(eval_retain_subset, batch_size=256, num_workers=2)

    # --- Baseline: print original model metrics ---
    print("\n[Original model]")
    model = load_model()
    f, r, t_acc = compute_final_metrics(model, FORGET_CLASSES, test_set)
    print(f"  Forget: {f:.1f}%  Retain: {r:.1f}%  Test: {t_acc:.1f}%")
    del model; torch.cuda.empty_cache()

    results = {}
    # Each entry: {"online": float, "offline": float or None}
    timing = {}

    # ─────────────────────────────────
    # GA
    # ─────────────────────────────────
    if "ga" in run_methods:
        print("\n" + "=" * 60)
        print(f"[GA] {UNLEARN_STEPS} steps, lr={UNLEARN_LR}")
        model = load_model()
        t0 = time.time()
        unlearn_ga(model, forget_loader, UNLEARN_STEPS, UNLEARN_LR,
                   eval_forget_loader=eval_forget_loader,
                   eval_retain_loader=eval_retain_loader)
        online_time = time.time() - t0
        f, r, t_acc = compute_final_metrics(model, FORGET_CLASSES, test_set)
        results["GA"] = (f, r, t_acc)
        timing["GA"] = {"online": online_time, "offline": None}
        print(f"[GA] FINAL  Forget: {f:.1f}%  Retain: {r:.1f}%  "
              f"Test: {t_acc:.1f}%  (online: {online_time:.0f}s)")
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/ga.pt")
        del model; torch.cuda.empty_cache()

    # ─────────────────────────────────
    # GPM-W (ResNet only)
    # ─────────────────────────────────
    if "gpm_w" in run_methods:
        if "resnet" in MODEL_ARCH:
            print("\n" + "=" * 60)
            print(f"[GPM-W] Building weight-space bases...")
            model_svd = load_model()

            # Offline: SVD base construction
            t_offline_start = time.time()
            weight_bases, named_convs = build_full_weight_space_bases(model_svd, retain_dataset)
            offline_time = time.time() - t_offline_start

            del model_svd; torch.cuda.empty_cache()

            model = load_model()
            # Online: unlearning loop only
            t0 = time.time()
            unlearn_gpm_w(model, forget_loader, weight_bases, named_convs, UNLEARN_STEPS, UNLEARN_LR)
            online_time = time.time() - t0

            f, r, t_acc = compute_final_metrics(model, FORGET_CLASSES, test_set)
            results["GPM-W"] = (f, r, t_acc)
            timing["GPM-W"] = {"online": online_time, "offline": offline_time}
            print(f"[GPM-W] FINAL  Forget: {f:.1f}%  Retain: {r:.1f}%  "
                  f"Test: {t_acc:.1f}%  (online: {online_time:.0f}s)")
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/gpm_w.pt")
            del model; torch.cuda.empty_cache()
        else:
            print("\n[GPM-W] Skipped — not applicable to ViT architecture (no Conv2d in transformer blocks).")
            results["GPM-W"] = None

    # ─────────────────────────────────
    # RASE
    # ─────────────────────────────────
    if "rase" in run_methods:
        print("\n" + "=" * 60)
        print(f"[RASE] Building activation-space bases...")
        model_svd = load_model()

        # Offline: SVD base construction
        t_offline_start = time.time()
        rase_bases = build_rase_bases(model_svd)
        offline_time = time.time() - t_offline_start

        del model_svd; torch.cuda.empty_cache()

        model = load_model()
        # Online: unlearning loop only
        t0 = time.time()
        unlearn_rase(model, forget_loader, rase_bases, UNLEARN_STEPS, UNLEARN_LR)
        online_time = time.time() - t0

        f, r, t_acc = compute_final_metrics(model, FORGET_CLASSES, test_set)
        results["RASE"] = (f, r, t_acc)
        timing["RASE"] = {"online": online_time, "offline": offline_time}
        print(f"[RASE] FINAL  Forget: {f:.1f}%  Retain: {r:.1f}%  "
              f"Test: {t_acc:.1f}%  (online: {online_time:.0f}s)")
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/rase.pt")
        del model; torch.cuda.empty_cache()

    # ─────────────────────────────────
    # SCRUB
    # ─────────────────────────────────
    if "scrub" in run_methods:
        print("\n" + "=" * 60)
        print(f"[SCRUB] {UNLEARN_STEPS} steps, lr={UNLEARN_LR}, "
              f"retain_steps_per_forget={SCRUB_RETAIN_STEPS}, "
              f"kl_weight={SCRUB_KL_WEIGHT}")
        model = load_model()
        t0 = time.time()
        unlearn_scrub(model, forget_loader, retain_dataset,
                      UNLEARN_STEPS, UNLEARN_LR,
                      eval_forget_loader=eval_forget_loader,
                      eval_retain_loader=eval_retain_loader)
        elapsed = time.time() - t0
        f, r, t_acc = compute_final_metrics(model, FORGET_CLASSES, test_set)
        results["SCRUB"] = (f, r, t_acc)
        timing["SCRUB"] = {"online": elapsed, "offline": None}
        print(f"[SCRUB] FINAL  Forget: {f:.1f}%  Retain: {r:.1f}%  "
              f"Test: {t_acc:.1f}%  (online: {elapsed:.0f}s)")
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/scrub.pt")
        del model; torch.cuda.empty_cache()

    # ─────────────────────────────────
    # Summary table
    # ─────────────────────────────────
    print("\n" + "=" * 80)
    print(f"UNLEARNING SUMMARY  —  {MODEL_ARCH}, forget classes {FORGET_CLASSES}")
    print("=" * 80)

    # Header
    print(f"{'Method':<10} | {'Forget%':>7} | {'Retain%':>7} | {'Test%':>7} | "
          f"{'Online(s)':>10} | {'Offline(s)':>11} | {'Note'}")
    print("-" * 80)

    for method, val in results.items():
        if val is None:
            print(f"{method:<10} | {'— skipped —':>46}")
            continue
        f_acc, r_acc, t_acc = val
        t_info = timing.get(method, {})
        online  = t_info.get("online",  None)
        offline = t_info.get("offline", None)

        online_str  = f"{online:.1f}"  if online  is not None else "—"
        offline_str = f"{offline:.1f}" if offline is not None else "—"

        # Note column: explain what offline means for projection methods
        if offline is not None:
            note = "(offline = 1-time SVD cost, not paid per request)"
        else:
            note = ""

        print(f"{method:<10} | {f_acc:6.1f}%  | {r_acc:6.1f}%  | {t_acc:6.1f}%  | "
              f"{online_str:>10} | {offline_str:>11} | {note}")

    print("=" * 80)

    # Timing interpretation block
    print()
    print("Timing interpretation:")
    print("  Online time  = cost paid per unlearning request (the fair comparison).")
    print("  Offline time = one-time SVD base construction (amortized across all")
    print("                 future requests; not counted in online cost for RASE/GPM-W).")
    if "RASE" in timing and "GA" in timing:
        rase_online = timing["RASE"]["online"]
        ga_online   = timing["GA"]["online"]
        ratio = rase_online / ga_online if ga_online > 0 else float("inf")
        print(f"  RASE online overhead vs GA: {ratio:.1f}x  "
              f"({rase_online:.1f}s vs {ga_online:.1f}s)")
    if "SCRUB" in timing and "GA" in timing:
        scrub_online = timing["SCRUB"]["online"]
        ga_online    = timing["GA"]["online"]
        ratio = scrub_online / ga_online if ga_online > 0 else float("inf")
        print(f"  SCRUB online overhead vs GA: {ratio:.1f}x  "
              f"({scrub_online:.1f}s vs {ga_online:.1f}s)")

    print(f"\nCheckpoints saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

# NOTE: To evaluate SCRUB checkpoints, add this entry to CHECKPOINTS in evaluate.py:
#   "SCRUB": f"{CHECKPOINT_DIR}/scrub.pt",
