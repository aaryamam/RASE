"""
unlearn.py — Run GA, GPM-W, and RASE unlearning on a chosen model/checkpoint.

Configuration: change MODEL_ARCH at the top to switch between resnet18 / vit_small.
Usage: python scripts/unlearn.py
"""

import os
import time
import random
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
MODEL_ARCH      = "resnet18"          # "resnet18" or "vit_small"
FORGET_CLASS    = 0                   # class index to forget (0-9)
UNLEARN_STEPS   = 800                 # gradient ascent steps
UNLEARN_LR      = 1e-3               # learning rate for all methods
SVD_THRESHOLD   = 0.95               # variance threshold for subspace construction
SAMPLES_PER_CLASS = 500              # retain samples per class for SVD
PRINT_EVERY     = 50                 # print progress every N steps

CHECKPOINT_PATH = {
    "resnet18":  "checkpoints/resnet18_cifar10.pt",
    "vit_small": "checkpoints/vit_small_cifar10.pt",
}[MODEL_ARCH]

OUTPUT_DIR = f"checkpoints/unlearned/{MODEL_ARCH}_forget{FORGET_CLASS}"


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
    else:
        raise ValueError(f"Unknown MODEL_ARCH: {MODEL_ARCH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    return model


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_forget_retain_loaders():
    """
    Returns forget_loader, forget_dataset, retain_dataset.
    forget_loader uses augmented train transforms.
    """
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=get_train_transform()
    )
    targets = np.array(train_set.targets)
    forget_idx = np.where(targets == FORGET_CLASS)[0].tolist()
    retain_idx = np.where(targets != FORGET_CLASS)[0].tolist()
    forget_dataset = Subset(train_set, forget_idx)
    retain_dataset = Subset(train_set, retain_idx)
    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True,
                               num_workers=2, drop_last=True)
    return forget_loader, forget_dataset, retain_dataset


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
def compute_final_metrics(model, forget_class, test_set):
    """
    Full evaluation:
      - Forget acc: train samples of forget_class (test transform, no augmentation)
      - Retain acc: train samples of all other classes
      - Test acc:   test set, retained classes only
    """
    train_set_test = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)
    forget_idx = np.where(targets == forget_class)[0].tolist()
    retain_idx = np.where(targets != forget_class)[0].tolist()

    forget_loader = DataLoader(Subset(train_set_test, forget_idx), batch_size=256, num_workers=2)
    retain_loader = DataLoader(Subset(train_set_test, retain_idx), batch_size=256, num_workers=2)

    test_targets = np.array(test_set.targets)
    test_retain_idx = np.where(test_targets != forget_class)[0].tolist()
    test_retain_loader = DataLoader(Subset(test_set, test_retain_idx), batch_size=256, num_workers=2)

    forget_acc = accuracy_on_loader(model, forget_loader)
    retain_acc = accuracy_on_loader(model, retain_loader)
    test_acc = accuracy_on_loader(model, test_retain_loader)
    return forget_acc, retain_acc, test_acc


def _make_small_eval_loaders(n=512):
    """Make small subsets (512 samples each) for mid-training progress prints."""
    train_set_test = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)
    rng = np.random.RandomState(SEED)

    forget_idx = np.where(targets == FORGET_CLASS)[0]
    retain_idx = np.where(targets != FORGET_CLASS)[0]

    f_chosen = rng.choice(forget_idx, min(n, len(forget_idx)), replace=False).tolist()
    r_chosen = rng.choice(retain_idx, min(n, len(retain_idx)), replace=False).tolist()

    f_loader = DataLoader(Subset(train_set_test, f_chosen), batch_size=256, num_workers=2)
    r_loader = DataLoader(Subset(train_set_test, r_chosen), batch_size=256, num_workers=2)
    return f_loader, r_loader


# ──────────────────────────────────────────────
# Method 1: GA (Gradient Ascent)
# ──────────────────────────────────────────────
def unlearn_ga(model, forget_loader, steps, lr):
    """Plain gradient ascent on forget data. No projection. No retain data."""
    criterion = nn.CrossEntropyLoss()
    forget_iter = iter(forget_loader)
    f_loader_small, r_loader_small = _make_small_eval_loaders()

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
            fa = accuracy_on_loader(model, f_loader_small)
            ra = accuracy_on_loader(model, r_loader_small)
            print(f"[GA] Step {step}/{steps} | Loss: {loss.item():.4f} | "
                  f"Forget: {fa:.1f}% | Retain: {ra:.1f}%")


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

    train_set_test = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)

    # Sample retain data (SAMPLES_PER_CLASS per class, all classes except FORGET_CLASS)
    retain_indices = []
    rng = np.random.RandomState(SEED)
    for c in range(10):
        if c == FORGET_CLASS:
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

    train_set_test = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)

    retain_indices = []
    rng = np.random.RandomState(SEED)
    for c in range(10):
        if c == FORGET_CLASS:
            continue
        class_idx = np.where(targets == c)[0]
        chosen = rng.choice(class_idx, min(samples_per_class, len(class_idx)), replace=False)
        retain_indices.extend(chosen.tolist())

    sample_set = Subset(train_set_test, retain_indices)
    sample_loader = DataLoader(sample_set, batch_size=256, shuffle=False, num_workers=2)

    # Determine hook targets and names
    if MODEL_ARCH == "resnet18":
        hook_layers = [
            ("layer1", model.layer1),
            ("layer2", model.layer2),
            ("layer3", model.layer3),
            ("layer4", model.layer4),
        ]
    else:  # vit_small
        hook_layers = [
            (f"block_{i}", model.blocks[i]) for i in range(len(model.blocks))
        ]

    all_acts = {name: [] for name, _ in hook_layers}
    hooks = []

    def hook_fn(name):
        def hook(module, inp, out):
            all_acts[name].append(out.detach().cpu())
        return hook

    for name, layer in hook_layers:
        hooks.append(layer.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        for images, _ in sample_loader:
            images = images.to(DEVICE)
            model(images)

    for h in hooks:
        h.remove()

    projection_bases = {}
    for name, _ in hook_layers:
        A = torch.cat(all_acts[name], dim=0)          # (N, ...) any shape
        A_flat = A.reshape(A.shape[0], -1).float()    # (N, D)

        U, S, _ = torch.linalg.svd(A_flat.T, full_matrices=False)
        explained = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
        k = (explained < threshold).sum().item() + 1
        k = min(k, U.shape[1])
        U_retain = U[:, :k]

        projection_bases[name] = U_retain.to(DEVICE)
        print(f"    {name}: retain_dims={k}/{U.shape[1]}, act_dim={A_flat.shape[1]}")

    return projection_bases


def unlearn_rase(model, forget_loader, rase_bases, steps, lr):
    """
    Activation-space projected gradient ascent (RASE).
    Backward hooks project grad_output away from the retain activation subspace.

    ResNet-18: hooks on layer1–layer4 (grad shape: (B, C, H, W) → flatten).
    ViT-Small: hooks on blocks[0–5] (grad shape: (B, 65, 256) → flatten).
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

    if MODEL_ARCH == "resnet18":
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, layer_name)
            backward_hooks.append(layer.register_full_backward_pre_hook(
                make_proj_backward_hook(rase_bases[layer_name])))
    else:  # vit_small
        for i, blk in enumerate(model.blocks):
            name = f"block_{i}"
            backward_hooks.append(blk.register_full_backward_pre_hook(
                make_proj_backward_hook(rase_bases[name])))

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
# Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Model:        {MODEL_ARCH}")
    print(f"Forget class: {FORGET_CLASS}")
    print(f"Steps:        {UNLEARN_STEPS}  LR: {UNLEARN_LR}")
    print(f"Output dir:   {OUTPUT_DIR}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print(f"Run the corresponding training script first.")
        return

    forget_loader, forget_dataset, retain_dataset = get_forget_retain_loaders()
    test_set = get_test_set()

    # --- Baseline: print original model metrics ---
    print("\n[Original model]")
    model = load_model()
    f, r, t = compute_final_metrics(model, FORGET_CLASS, test_set)
    print(f"  Forget: {f:.1f}%  Retain: {r:.1f}%  Test: {t:.1f}%")
    del model; torch.cuda.empty_cache()

    results = {}

    # ─────────────────────────────────
    # GA
    # ─────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[GA] {UNLEARN_STEPS} steps, lr={UNLEARN_LR}")
    model = load_model()
    t0 = time.time()
    unlearn_ga(model, forget_loader, UNLEARN_STEPS, UNLEARN_LR)
    elapsed = time.time() - t0
    f, r, t = compute_final_metrics(model, FORGET_CLASS, test_set)
    results["GA"] = (f, r, t, elapsed)
    print(f"[GA] FINAL  Forget: {f:.1f}%  Retain: {r:.1f}%  Test: {t:.1f}%  ({elapsed:.0f}s)")
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/ga.pt")
    del model; torch.cuda.empty_cache()

    # ─────────────────────────────────
    # GPM-W (ResNet only)
    # ─────────────────────────────────
    if MODEL_ARCH == "resnet18":
        print("\n" + "=" * 60)
        print(f"[GPM-W] Building weight-space bases...")
        model_svd = load_model()
        weight_bases, named_convs = build_full_weight_space_bases(model_svd, retain_dataset)
        del model_svd; torch.cuda.empty_cache()

        model = load_model()
        t0 = time.time()
        unlearn_gpm_w(model, forget_loader, weight_bases, named_convs, UNLEARN_STEPS, UNLEARN_LR)
        elapsed = time.time() - t0
        f, r, t = compute_final_metrics(model, FORGET_CLASS, test_set)
        results["GPM-W"] = (f, r, t, elapsed)
        print(f"[GPM-W] FINAL  Forget: {f:.1f}%  Retain: {r:.1f}%  Test: {t:.1f}%  ({elapsed:.0f}s)")
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/gpm_w.pt")
        del model; torch.cuda.empty_cache()
    else:
        print("\n[GPM-W] Skipped — not applicable to ViT architecture (no Conv2d in transformer blocks).")
        results["GPM-W"] = None

    # ─────────────────────────────────
    # RASE
    # ─────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[RASE] Building activation-space bases...")
    model_svd = load_model()
    rase_bases = build_rase_bases(model_svd)
    del model_svd; torch.cuda.empty_cache()

    model = load_model()
    t0 = time.time()
    unlearn_rase(model, forget_loader, rase_bases, UNLEARN_STEPS, UNLEARN_LR)
    elapsed = time.time() - t0
    f, r, t = compute_final_metrics(model, FORGET_CLASS, test_set)
    results["RASE"] = (f, r, t, elapsed)
    print(f"[RASE] FINAL  Forget: {f:.1f}%  Retain: {r:.1f}%  Test: {t:.1f}%  ({elapsed:.0f}s)")
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/rase.pt")
    del model; torch.cuda.empty_cache()

    # ─────────────────────────────────
    # Summary table
    # ─────────────────────────────────
    print("\n" + "=" * 65)
    print(f"UNLEARNING SUMMARY  ({MODEL_ARCH}, forget class {FORGET_CLASS})")
    print("=" * 65)
    print(f"{'Method':<10} | {'Forget':>7} | {'Retain':>7} | {'Test':>7} | {'Time':>7}")
    print("-" * 65)
    for method, val in results.items():
        if val is None:
            print(f"{method:<10} | {'skipped':>37}")
        else:
            f, r, t, elapsed = val
            print(f"{method:<10} | {f:6.1f}%  | {r:6.1f}%  | {t:6.1f}%  | {elapsed:5.0f}s")
    print("=" * 65)
    print(f"\nCheckpoints saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
