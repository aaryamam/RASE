"""
robustness.py — GPM-A vs GPM-W vs GA robustness over increasing unlearning steps.

Compares three unlearning methods under increasing step counts:
  GPM-A: Activation-space projection (backward hooks on layer outputs)
  GPM-W: Weight-space projection (original GPM paper — project weight gradients
         using SVD of unfolded input patches)
  GA:    Plain gradient ascent (no projection, baseline)

Hypothesis: Both GPM variants protect retain accuracy because the projection
geometrically constrains updates. GA collapses because unconstrained gradient
ascent damages shared features.

Neither method uses retain data during the unlearning loop.
GPM uses retain data ONCE upfront to build projection matrices.
GA never touches retain data at all.

Usage: python scripts/robustness.py
"""

import os
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
CHECKPOINT_PATH = "checkpoints/resnet18_cifar10.pt"


# ──────────────────────────────────────────────
# Model (same as train.py / unlearn.py)
# ──────────────────────────────────────────────
def make_resnet18_cifar10():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def load_trained_model():
    model = make_resnet18_cifar10().to(DEVICE)
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
# GPM: Build projection bases (retain subspace only — no forget-only bases needed)
# ──────────────────────────────────────────────
def build_projection_matrices(model, retain_dataset, forget_dataset,
                              samples_per_class=500, threshold=0.95):
    """
    Build per-layer U_retain: retain subspace basis (for gradient projection).
    This is a simplified version — we only need the retain subspace for
    the pure projected gradient ascent experiment.
    """
    print("  Building projection matrices...")

    train_set_test = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)

    # --- Sample retain data (500 per class, classes 1-9) ---
    retain_indices = []
    rng = np.random.RandomState(SEED)
    for c in range(10):
        if c == FORGET_CLASS:
            continue
        class_idx = np.where(targets == c)[0]
        chosen = rng.choice(class_idx, min(samples_per_class, len(class_idx)), replace=False)
        retain_indices.extend(chosen.tolist())

    layer_names = ["layer1", "layer2", "layer3", "layer4"]

    # --- Collect activations for retain data ---
    def collect_activations(model, indices):
        sample_set = Subset(train_set_test, indices)
        sample_loader = DataLoader(sample_set, batch_size=256, shuffle=False, num_workers=2)

        all_acts = {name: [] for name in layer_names}
        hooks = []

        def hook_fn(name):
            def hook(module, inp, out):
                all_acts[name].append(out.detach().cpu())
            return hook

        hooks.append(model.layer1.register_forward_hook(hook_fn("layer1")))
        hooks.append(model.layer2.register_forward_hook(hook_fn("layer2")))
        hooks.append(model.layer3.register_forward_hook(hook_fn("layer3")))
        hooks.append(model.layer4.register_forward_hook(hook_fn("layer4")))

        model.eval()
        with torch.no_grad():
            for images, _ in sample_loader:
                images = images.to(DEVICE)
                model(images)

        for h in hooks:
            h.remove()

        result = {}
        for name in layer_names:
            A = torch.cat(all_acts[name], dim=0)  # (N, C, H, W)
            result[name] = A.reshape(A.shape[0], -1).float()  # (N, D)
        return result

    retain_acts = collect_activations(model, retain_indices)

    # --- Build retain subspace per layer ---
    projection_bases = {}

    for name in layer_names:
        R = retain_acts[name]  # (N_r, D)

        # Retain subspace via SVD
        U_r, S_r, _ = torch.linalg.svd(R.T, full_matrices=False)
        explained_r = torch.cumsum(S_r ** 2, dim=0) / (S_r ** 2).sum()
        k_r = (explained_r < threshold).sum().item() + 1
        k_r = min(k_r, U_r.shape[1])
        U_retain = U_r[:, :k_r]  # (D, k_r)

        projection_bases[name] = U_retain.to(DEVICE)
        print(f"    {name}: retain_dims={k_r}")

    return projection_bases


# ──────────────────────────────────────────────
# GPM unlearning: Pure projected gradient ascent
# ──────────────────────────────────────────────
def run_gpm_unlearning(model, forget_loader, projection_bases, steps, lr):
    """
    Pure projected gradient ascent. No retain data. No extras.
    Backward hooks project gradients away from the retain subspace automatically.
    """
    # --- Register backward hooks for gradient projection ---
    backward_hooks = []

    def make_proj_backward_hook(U_k):
        def hook(module, grad_output):
            g = grad_output[0]  # (B, C, H, W)
            B = g.shape[0]
            g_flat = g.reshape(B, -1)
            g_proj = g_flat - (g_flat @ U_k) @ U_k.T
            return (g_proj.reshape(g.shape),)
        return hook

    backward_hooks.append(model.layer1.register_full_backward_pre_hook(
        make_proj_backward_hook(projection_bases["layer1"])))
    backward_hooks.append(model.layer2.register_full_backward_pre_hook(
        make_proj_backward_hook(projection_bases["layer2"])))
    backward_hooks.append(model.layer3.register_full_backward_pre_hook(
        make_proj_backward_hook(projection_bases["layer3"])))
    backward_hooks.append(model.layer4.register_full_backward_pre_hook(
        make_proj_backward_hook(projection_bases["layer4"])))

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
# GPM-W: Build weight-space projection bases from unfolded input patches
# ──────────────────────────────────────────────
def build_weight_space_bases(model, retain_dataset, samples_per_class=500, threshold=0.95):
    """
    Build per-layer projection bases from UNFOLDED INPUT patches.
    This is the correct GPM formulation for conv layers (Saha et al., ICLR 2021).

    For conv layers: unfold layer input with first conv's kernel params → SVD.
    For fc layer: collect 512-dim input vectors after avgpool → SVD.
    """
    print("  Building weight-space projection matrices...")

    train_set_test = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)

    # Sample retain data (500 per class, classes 1-9)
    retain_indices = []
    rng = np.random.RandomState(SEED)
    for c in range(10):
        if c == FORGET_CLASS:
            continue
        class_idx = np.where(targets == c)[0]
        chosen = rng.choice(class_idx, min(samples_per_class, len(class_idx)), replace=False)
        retain_indices.extend(chosen.tolist())

    sample_set = Subset(train_set_test, retain_indices)
    sample_loader = DataLoader(sample_set, batch_size=128, shuffle=False, num_workers=2)

    # Capture the INPUT to each layer block and the fc layer
    layer_inputs = {"layer1": [], "layer2": [], "layer3": [], "layer4": [], "fc": []}
    hooks = []

    def capture_input(name):
        def hook(module, inp, out):
            layer_inputs[name].append(inp[0].detach().cpu())
        return hook

    hooks.append(model.layer1.register_forward_hook(capture_input("layer1")))
    hooks.append(model.layer2.register_forward_hook(capture_input("layer2")))
    hooks.append(model.layer3.register_forward_hook(capture_input("layer3")))
    hooks.append(model.layer4.register_forward_hook(capture_input("layer4")))
    hooks.append(model.fc.register_forward_hook(capture_input("fc")))

    model.eval()
    with torch.no_grad():
        for images, _ in sample_loader:
            images = images.to(DEVICE)
            model(images)

    for h in hooks:
        h.remove()

    # Build projection bases per layer
    projection_bases = {}

    # --- Conv layers: unfold inputs using first conv's kernel params ---
    layer_conv_params = {
        "layer1": model.layer1[0].conv1,
        "layer2": model.layer2[0].conv1,
        "layer3": model.layer3[0].conv1,
        "layer4": model.layer4[0].conv1,
    }

    for name in ["layer1", "layer2", "layer3", "layer4"]:
        conv = layer_conv_params[name]
        kernel_size = conv.kernel_size
        padding = conv.padding
        stride = conv.stride

        all_inputs = torch.cat(layer_inputs[name], dim=0)  # (N, C_in, H, W)

        # Unfold: extract patches that each filter position operates on
        # F.unfold output shape: (N, C_in * kH * kW, L) where L = H_out * W_out
        patches = F.unfold(
            all_inputs,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )  # (N, C_in*kH*kW, L)

        # Reshape to (N*L, C_in*kH*kW) — each row is one input patch
        patch_dim = patches.shape[1]
        patches = patches.permute(0, 2, 1).reshape(-1, patch_dim).float()  # (N*L, patch_dim)

        # Subsample if too many patches
        max_patches = 10000
        if patches.shape[0] > max_patches:
            idx = rng.choice(patches.shape[0], max_patches, replace=False)
            patches = patches[idx]

        # SVD to find the principal input directions
        U, S, _ = torch.linalg.svd(patches.T, full_matrices=False)

        # Keep components explaining threshold% of variance
        explained = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
        k = (explained < threshold).sum().item() + 1
        k = min(k, U.shape[1])
        U_k = U[:, :k]  # (patch_dim, k)

        projection_bases[name] = {
            "U_k": U_k.to(DEVICE),
            "patch_dim": patch_dim,
        }

        print(f"    {name}: input_patch_dim={patch_dim}, kept {k}/{U.shape[1]} components")

    # --- FC layer ---
    fc_inputs = torch.cat(layer_inputs["fc"], dim=0).float()  # (N, 512)
    U_fc, S_fc, _ = torch.linalg.svd(fc_inputs.T, full_matrices=False)
    explained_fc = torch.cumsum(S_fc ** 2, dim=0) / (S_fc ** 2).sum()
    k_fc = (explained_fc < threshold).sum().item() + 1
    k_fc = min(k_fc, U_fc.shape[1])
    U_fc_k = U_fc[:, :k_fc]  # (512, k)

    projection_bases["fc"] = {
        "U_k": U_fc_k.to(DEVICE),
        "patch_dim": fc_inputs.shape[1],
    }
    print(f"    fc: input_dim={fc_inputs.shape[1]}, kept {k_fc}/{U_fc.shape[1]} components")

    return projection_bases


# ──────────────────────────────────────────────
# GPM-W unlearning: Weight-space projected gradient ascent
# ──────────────────────────────────────────────
def run_gpm_w_unlearning(model, forget_loader, projection_bases, steps, lr):
    """
    Weight-space GPM: Normal backward pass, then project weight gradients
    using input-covariance bases before applying ascent.

    Only projects conv layers whose flattened weight dim matches the patch dim
    (first conv in each block). Other convs get unprojected ascent.
    """
    criterion = nn.CrossEntropyLoss()
    forget_iter = iter(forget_loader)

    # Map layer names to their conv modules
    layer_convs = {
        "layer1": [m for m in model.layer1.modules() if isinstance(m, nn.Conv2d)],
        "layer2": [m for m in model.layer2.modules() if isinstance(m, nn.Conv2d)],
        "layer3": [m for m in model.layer3.modules() if isinstance(m, nn.Conv2d)],
        "layer4": [m for m in model.layer4.modules() if isinstance(m, nn.Conv2d)],
    }

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
        loss.backward()  # Normal backward — no hooks

        # Project weight gradients, then apply ascent
        with torch.no_grad():
            # --- Project conv weight gradients ---
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                U_k = projection_bases[layer_name]["U_k"]  # (patch_dim, k)

                for conv in layer_convs[layer_name]:
                    if conv.weight.grad is None:
                        continue

                    W_grad = conv.weight.grad  # (C_out, C_in, kH, kW)
                    C_out = W_grad.shape[0]

                    # Reshape to (C_out, C_in*kH*kW)
                    G = W_grad.reshape(C_out, -1)  # (C_out, patch_dim)

                    # Only project if dimensions match (first conv in block)
                    if G.shape[1] == U_k.shape[0]:
                        G_proj = G - (G @ U_k) @ U_k.T
                        conv.weight.grad = G_proj.reshape(W_grad.shape)

            # --- Project fc weight gradient ---
            if model.fc.weight.grad is not None:
                U_fc = projection_bases["fc"]["U_k"]  # (512, k)
                G_fc = model.fc.weight.grad  # (10, 512)

                if G_fc.shape[1] == U_fc.shape[0]:
                    G_fc_proj = G_fc - (G_fc @ U_fc) @ U_fc.T
                    model.fc.weight.grad = G_fc_proj

            # --- Apply gradient ASCENT on all parameters ---
            for p in model.parameters():
                if p.grad is not None:
                    p.data += lr * p.grad


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def find_zero_forget_point(results, threshold=1.0):
    """Find the first entry where forget acc drops below threshold%."""
    for steps, f_acc, r_acc, t_acc in results:
        if f_acc < threshold:
            return steps, f_acc, r_acc, t_acc
    # If never reached, return the last entry
    return results[-1]


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")

    forget_dataset, retain_dataset = get_forget_retain_datasets()
    test_set = get_test_set()
    print(f"Forget set: {len(forget_dataset)} samples | Retain set: {len(retain_dataset)} samples")

    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True,
                               num_workers=2, drop_last=True)

    # ──────────────────────────────────────────────
    # Part 1: GPM-A vs GPM-W vs GA (normal lr)
    # ──────────────────────────────────────────────
    step_counts = [50, 100, 200, 400, 800]
    lr = 1e-3

    # Build BOTH sets of projection matrices ONCE from the original checkpoint
    print("\nBuilding projection matrices (one-time cost)...")
    model_for_svd = load_trained_model()
    act_bases = build_projection_matrices(model_for_svd, retain_dataset, forget_dataset)
    weight_bases = build_weight_space_bases(model_for_svd, retain_dataset)
    del model_for_svd
    torch.cuda.empty_cache()

    results_gpm_a = []  # Activation-space (our method)
    results_gpm_w = []  # Weight-space (original GPM formulation)
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

        # --- GPM-W (weight-space) ---
        print(f"\n  [GPM-W] Running {steps} steps of weight-projected gradient ascent...")
        model = load_trained_model()
        run_gpm_w_unlearning(model, forget_loader, weight_bases, steps=steps, lr=lr)
        f_acc, r_acc, t_acc = compute_final_metrics(model, forget_dataset, retain_dataset, test_set)
        results_gpm_w.append((steps, f_acc, r_acc, t_acc))
        print(f"  [GPM-W] Steps={steps:4d} | Forget: {f_acc:5.1f}% | Retain: {r_acc:5.1f}% | Test: {t_acc:5.1f}%")
        del model
        torch.cuda.empty_cache()

        # --- GA (baseline, lr=1e-3) ---
        print(f"\n  [GA]  Running {steps} steps of plain gradient ascent...")
        model = load_trained_model()
        run_ga_unlearning(model, forget_loader, steps=steps, lr=lr)
        f_acc, r_acc, t_acc = compute_final_metrics(model, forget_dataset, retain_dataset, test_set)
        results_ga.append((steps, f_acc, r_acc, t_acc))
        print(f"  [GA]  Steps={steps:4d} | Forget: {f_acc:5.1f}% | Retain: {r_acc:5.1f}% | Test: {t_acc:5.1f}%")
        del model
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # Part 2: Learning Rate Control — GA at reduced lrs
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LEARNING RATE CONTROL: GA at reduced learning rates")
    print("=" * 60)

    # GA-Slow (lr=1e-4, 10x smaller)
    step_counts_slow = [500, 1000, 2000, 4000, 8000]
    lr_slow = 1e-4
    results_ga_slow = []

    for steps in step_counts_slow:
        print(f"\n  [GA lr=1e-4] Running {steps} steps...")
        model = load_trained_model()
        run_ga_unlearning(model, forget_loader, steps=steps, lr=lr_slow)
        f_acc, r_acc, t_acc = compute_final_metrics(model, forget_dataset, retain_dataset, test_set)
        results_ga_slow.append((steps, f_acc, r_acc, t_acc))
        print(f"  [GA lr=1e-4] Steps={steps:5d} | Forget: {f_acc:5.1f}% | Retain: {r_acc:5.1f}% | Test: {t_acc:5.1f}%")
        del model
        torch.cuda.empty_cache()

    # GA-Slower (lr=5e-5, 20x smaller)
    step_counts_slower = [1000, 2000, 4000, 8000, 16000]
    lr_slower = 5e-5
    results_ga_slower = []

    for steps in step_counts_slower:
        print(f"\n  [GA lr=5e-5] Running {steps} steps...")
        model = load_trained_model()
        run_ga_unlearning(model, forget_loader, steps=steps, lr=lr_slower)
        f_acc, r_acc, t_acc = compute_final_metrics(model, forget_dataset, retain_dataset, test_set)
        results_ga_slower.append((steps, f_acc, r_acc, t_acc))
        print(f"  [GA lr=5e-5] Steps={steps:5d} | Forget: {f_acc:5.1f}% | Retain: {r_acc:5.1f}% | Test: {t_acc:5.1f}%")
        del model
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════
    #                 RESULTS TABLES
    # ══════════════════════════════════════════════

    # ──────────────────────────────────────────────
    # Table 1: GPM-A vs GPM-W vs GA (normal lr)
    # ──────────────────────────────────────────────
    print()
    w = 93
    print("=" * w)
    print("ROBUSTNESS: GPM-A vs GPM-W vs GA")
    print("=" * w)
    print(f"         |    GPM-A (ours)         |    GPM-W (weight)       |    GA (baseline)")
    print(f"Steps    | Forget | Retain | Test  | Forget | Retain | Test  | Forget | Retain | Test")
    print(f"---------|--------|--------|-------|--------|--------|-------|--------|--------|------")
    for (s, fa, ra, ta), (_, fw, rw, tw), (_, fg, rg, tg) in zip(results_gpm_a, results_gpm_w, results_ga):
        print(f"  {s:5d} | {fa:5.1f}  | {ra:5.1f}  | {ta:4.1f}  "
              f"| {fw:5.1f}  | {rw:5.1f}  | {tw:4.1f}  "
              f"| {fg:5.1f}  | {rg:5.1f}  | {tg:4.1f}")
    print("=" * w)

    # ──────────────────────────────────────────────
    # Table 2: Retain gap
    # ──────────────────────────────────────────────
    print()
    print(f"Steps    | GPM-A Ret | GPM-W Ret | GA Ret  | Gap A-GA | Gap W-GA")
    print(f"---------|----------|----------|---------|----------|----------")
    for (s, _, ra, _), (_, _, rw, _), (_, _, rg, _) in zip(results_gpm_a, results_gpm_w, results_ga):
        gap_a = ra - rg
        gap_w = rw - rg
        print(f"  {s:5d} | {ra:7.1f}  | {rw:7.1f}  | {rg:6.1f}  | {gap_a:+7.1f}  | {gap_w:+7.1f}")
    print("=" * w)

    # ──────────────────────────────────────────────
    # Table 3: GA-Slow full curve
    # ──────────────────────────────────────────────
    print()
    print("=" * 50)
    print("GA-Slow (lr=1e-4):")
    print("=" * 50)
    print(f"Steps    | Forget% | Retain% | Test%")
    print(f"---------|---------|---------|--------")
    for s, f, r, t in results_ga_slow:
        print(f"  {s:5d} | {f:6.1f}  | {r:6.1f}  | {t:5.1f}")
    print("=" * 50)

    # ──────────────────────────────────────────────
    # Table 4: GA-Slower full curve
    # ──────────────────────────────────────────────
    print()
    print("=" * 50)
    print("GA-Slower (lr=5e-5):")
    print("=" * 50)
    print(f"Steps    | Forget% | Retain% | Test%")
    print(f"---------|---------|---------|--------")
    for s, f, r, t in results_ga_slower:
        print(f"  {s:5d} | {f:6.1f}  | {r:6.1f}  | {t:5.1f}")
    print("=" * 50)

    # ──────────────────────────────────────────────
    # Table 5: Learning Rate Control Summary
    # ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("LEARNING RATE CONTROL EXPERIMENT")
    print("=" * 70)
    print("Does a slower GA learning rate match GPM-A's retain protection?")
    print()
    print(f"{'Method':<14}|{'lr':>10} | {'Steps to <1% Forget':>20} | {'Retain at that point':>20}")
    print(f"{'-'*14}|{'-'*10}-|{'-'*21}-|{'-'*20}")

    all_methods = [
        ("GPM-A",     "1e-3",  results_gpm_a),
        ("GA",        "1e-3",  results_ga),
        ("GA-Slow",   "1e-4",  results_ga_slow),
        ("GA-Slower",  "5e-5", results_ga_slower),
    ]

    for name, lr_str, results in all_methods:
        s, f, r, t = find_zero_forget_point(results)
        if f < 1.0:
            print(f"{name:<14}|{lr_str:>10} | {s:>20d} | {r:>19.1f}%")
        else:
            print(f"{name:<14}|{lr_str:>10} | {'never reached':>20} | {r:>19.1f}%")

    print("=" * 70)
    print()
    print("If GPM-A retain >> GA retain at all learning rates, the projection")
    print("changes gradient DIRECTION, not just magnitude. That's the result.")


if __name__ == "__main__":
    main()
