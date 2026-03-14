"""
cifar100_unlearn_mia.py — Unlearn 1 random CIFAR-100 class via GPM-A and GA,
then compute MIA on original and both unlearned models.

Saves checkpoints after each unlearning method.

Requires: checkpoints/resnet18_cifar100.pt
Usage:    python scripts/cifar100_unlearn_mia.py
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

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
NUM_CLASSES = 100

# ──────────────────────────────────────────────
# Unlearning Config
# ──────────────────────────────────────────────
USE_VULNERABLE_MODEL = False  # Set to True to use the overfit/high-MIA checkpoint
CHECKPOINT_PATH = ("checkpoints/resnet18_cifar100_vulnerable.pt"
                   if USE_VULNERABLE_MODEL
                   else "checkpoints/resnet18_cifar100.pt")

# Pick 1 random class to forget (seeded for reproducibility)
_rng = random.Random(SEED)
FORGET_CLASS = _rng.randint(0, NUM_CLASSES - 1)

STEPS = 800
LR = 1e-3


# ──────────────────────────────────────────────
# Model (same arch as train_cifar100.py)
# ──────────────────────────────────────────────
def make_resnet18_cifar100():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, NUM_CLASSES)
    return model


def load_trained_model():
    model = make_resnet18_cifar100().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True))
    return model


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────
def get_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def get_train_transform():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def get_forget_retain_loaders(forget_class):
    """Split CIFAR-100 TRAINING set into forget and retain based on forget_class."""
    train_set = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=get_train_transform()
    )
    targets = np.array(train_set.targets)
    forget_idx = np.where(targets == forget_class)[0].tolist()
    retain_idx = np.where(targets != forget_class)[0].tolist()
    forget_dataset = Subset(train_set, forget_idx)
    retain_dataset = Subset(train_set, retain_idx)

    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True,
                               num_workers=2, drop_last=True)
    return forget_loader, forget_dataset, retain_dataset


def get_test_set():
    return torchvision.datasets.CIFAR100(
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
def compute_final_metrics(model, test_set, forget_class):
    """Full-set evaluation for the given forget_class."""
    # Use test-time transforms (no augmentation) for evaluation
    train_set_test = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)
    forget_idx = np.where(targets == forget_class)[0].tolist()
    retain_idx = np.where(targets != forget_class)[0].tolist()

    forget_loader = DataLoader(Subset(train_set_test, forget_idx), batch_size=256, num_workers=2)
    retain_loader = DataLoader(Subset(train_set_test, retain_idx), batch_size=256, num_workers=2)

    # Test acc on retained classes only
    test_targets = np.array(test_set.targets)
    test_retain_idx = np.where(test_targets != forget_class)[0].tolist()
    test_retain_loader = DataLoader(Subset(test_set, test_retain_idx), batch_size=256, num_workers=2)

    forget_acc = accuracy_on_loader(model, forget_loader)
    retain_acc = accuracy_on_loader(model, retain_loader)
    test_acc = accuracy_on_loader(model, test_retain_loader)
    return forget_acc, retain_acc, test_acc


# ──────────────────────────────────────────────
# MIA: Loss-based membership inference attack
# ──────────────────────────────────────────────
@torch.no_grad()
def compute_mia_score(model, forget_class):
    """
    Simple loss-based MIA: can an attacker distinguish forget-set training
    examples from held-out test examples of the same class?

    Returns:
        mia_accuracy: float (50.0 = perfect unlearning, >50 = residual memorization)
        avg_member_loss: float (mean loss on forget-set training examples)
        avg_nonmember_loss: float (mean loss on same-class test examples)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    test_transform = get_test_transform()

    # 1. MEMBER set: forget-class examples from TRAINING set
    train_set = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=False, transform=test_transform
    )
    train_targets = np.array(train_set.targets)
    member_indices = np.where(train_targets == forget_class)[0].tolist()
    member_loader = DataLoader(Subset(train_set, member_indices),
                               batch_size=256, shuffle=False, num_workers=2)

    # 2. NON-MEMBER set: forget-class examples from TEST set
    test_set = torchvision.datasets.CIFAR100(
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

    # 4. Binary classification: find optimal threshold on loss
    #    Members = label 1, non-members = label 0.
    #    Check both directions: member loss < threshold (memorized model)
    #    and member loss > threshold (over-forgotten model).
    all_losses = np.array(member_losses + nonmember_losses)
    all_labels = np.array([1] * len(member_losses) + [0] * len(nonmember_losses))

    n_members = np.sum(all_labels == 1)
    n_nonmembers = np.sum(all_labels == 0)

    best_acc = 50.0
    for t in np.unique(all_losses):
        # Strategy: predict member if loss <= threshold
        preds = (all_losses <= t).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        tn = np.sum((preds == 0) & (all_labels == 0))
        tpr = tp / n_members       # sensitivity
        tnr = tn / n_nonmembers    # specificity
        balanced_acc = 100.0 * (tpr + tnr) / 2.0
        if balanced_acc > best_acc:
            best_acc = balanced_acc
        # Reversed strategy: predict member if loss > threshold
        balanced_acc_rev = 100.0 * ((1 - tpr) + (1 - tnr)) / 2.0
        if balanced_acc_rev > best_acc:
            best_acc = balanced_acc_rev

    avg_member = np.mean(member_losses)
    avg_nonmember = np.mean(nonmember_losses)

    return best_acc, avg_member, avg_nonmember


# ──────────────────────────────────────────────
# GPM-A: Build activation-space projection bases
# ──────────────────────────────────────────────
def build_activation_bases(model, forget_class, samples_per_class=50, threshold=0.95):
    """
    Build per-layer U_retain: retain subspace basis (for gradient projection).
    Parameterized by forget_class — samples from all classes except forget_class.

    samples_per_class=50 (vs 500 for CIFAR-10) because 99 retain classes × 50 = 4950 samples,
    which is comparable to 9 × 500 = 4500 in the CIFAR-10 version.
    """
    print(f"  Building activation projection bases (forget_class={forget_class})...")

    train_set_test = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=False, transform=get_test_transform()
    )
    targets = np.array(train_set_test.targets)

    # Sample retain data (50 per class, all classes except forget_class)
    retain_indices = []
    rng = np.random.RandomState(SEED)
    for c in range(NUM_CLASSES):
        if c == forget_class:
            continue
        class_idx = np.where(targets == c)[0]
        chosen = rng.choice(class_idx, min(samples_per_class, len(class_idx)), replace=False)
        retain_indices.extend(chosen.tolist())

    print(f"    Retain samples: {len(retain_indices)} ({samples_per_class}/class × {NUM_CLASSES - 1} classes)")

    layer_names = ["layer1", "layer2", "layer3", "layer4"]

    # Collect activations for retain data
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

    # Build retain subspace per layer
    projection_bases = {}
    dims_info = {}

    for name in layer_names:
        R = retain_acts[name]  # (N_r, D)

        # Retain subspace via SVD
        U_r, S_r, _ = torch.linalg.svd(R.T, full_matrices=False)
        explained_r = torch.cumsum(S_r ** 2, dim=0) / (S_r ** 2).sum()
        k_r = (explained_r < threshold).sum().item() + 1
        k_r = min(k_r, U_r.shape[1])
        U_retain = U_r[:, :k_r]  # (D, k_r)

        projection_bases[name] = U_retain.to(DEVICE)
        dims_info[name] = k_r
        print(f"    {name}: retain_dims={k_r}")

    return projection_bases, dims_info


# ──────────────────────────────────────────────
# GPM-A unlearning: Pure projected gradient ascent
# ──────────────────────────────────────────────
def run_gpm_a_unlearning(model, forget_loader, projection_bases, steps, lr):
    """
    Pure projected gradient ascent. No retain data. No extras.
    Backward hooks project gradients away from the retain subspace automatically.
    """
    # Register backward hooks for gradient projection
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
# Summary table
# ──────────────────────────────────────────────
def print_summary_table(forget_class, class_name, original_mia, gpma_results, ga_results):
    """Print comparison table: Original vs GPM-A vs GA."""
    w = 90
    print()
    print("=" * w)
    print(f"CIFAR-100 Unlearning + MIA: ResNet-18 (steps={STEPS}, lr={LR})")
    print(f"Forget class: {forget_class} ({class_name})")
    print("=" * w)
    print(f"{'Method':<12} | {'Forget':>7} {'Retain':>7} {'Test':>7} | {'MIA':>6} {'MemL':>7} {'NmL':>7}")
    print("-" * w)

    # Original (no unlearning)
    orig_mia, orig_mem, orig_nmem = original_mia
    print(f"{'Original':<12} | {'—':>7} {'—':>7} {'—':>7} | "
          f"{orig_mia:6.1f} {orig_mem:7.3f} {orig_nmem:7.3f}")

    # GPM-A
    a = gpma_results
    print(f"{'GPM-A':<12} | {a['forget']:7.1f} {a['retain']:7.1f} {a['test']:7.1f} | "
          f"{a['mia']:6.1f} {a['mem_loss']:7.3f} {a['nonmem_loss']:7.3f}")

    # GA
    g = ga_results
    print(f"{'GA':<12} | {g['forget']:7.1f} {g['retain']:7.1f} {g['test']:7.1f} | "
          f"{g['mia']:6.1f} {g['mem_loss']:7.3f} {g['nonmem_loss']:7.3f}")

    print("=" * w)
    print()
    print("MIA Interpretation: 50% = perfect unlearning, >50% = residual memorization")
    print(f"  Original MIA: {orig_mia:.1f}%  (baseline — model has memorized training data)")
    print(f"  GPM-A    MIA: {a['mia']:.1f}%  "
          f"({'good' if a['mia'] < 60 else 'moderate' if a['mia'] < 70 else 'high'} unlearning)")
    print(f"  GA       MIA: {g['mia']:.1f}%  "
          f"({'good' if g['mia'] < 60 else 'moderate' if g['mia'] < 70 else 'high'} unlearning)")
    print()
    print(f"  Retain Acc Gap (GPM-A vs GA): {a['retain'] - g['retain']:+.1f}pp")
    print(f"  Test   Acc Gap (GPM-A vs GA): {a['test'] - g['test']:+.1f}pp")
    print("=" * w)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Forget class: {FORGET_CLASS}")
    print(f"Steps: {STEPS}, LR: {LR}")

    # Retrieve class name
    test_set_meta = torchvision.datasets.CIFAR100(root="./data", train=False, download=True)
    class_names = test_set_meta.classes  # list of 100 fine-label names
    forget_class_name = class_names[FORGET_CLASS]
    print(f"Forget class name: {forget_class_name}")

    test_set = get_test_set()
    os.makedirs("checkpoints", exist_ok=True)

    # ── 1. Original model: MIA baseline ──
    print("\n" + "=" * 60)
    print("  STEP 1: MIA on original model (no unlearning)")
    print("=" * 60)
    model = load_trained_model()
    orig_f, orig_r, orig_t = compute_final_metrics(model, test_set, FORGET_CLASS)
    print(f"  Original model — Forget={orig_f:.1f}% | Retain={orig_r:.1f}% | Test={orig_t:.1f}%")
    orig_mia, orig_mem, orig_nmem = compute_mia_score(model, FORGET_CLASS)
    print(f"  Original MIA: {orig_mia:.1f}% | MemL={orig_mem:.4f} | NmL={orig_nmem:.4f}")
    del model
    torch.cuda.empty_cache()

    # ── 2. GPM-A unlearning ──
    print("\n" + "=" * 60)
    print(f"  STEP 2: GPM-A unlearning (class {FORGET_CLASS}: {forget_class_name})")
    print("=" * 60)

    forget_loader, _, _ = get_forget_retain_loaders(FORGET_CLASS)

    # Build projection bases
    model_tmp = load_trained_model()
    act_bases, dims_info = build_activation_bases(model_tmp, FORGET_CLASS)
    del model_tmp
    torch.cuda.empty_cache()

    # Run GPM-A
    print(f"\n  [GPM-A] Running {STEPS} steps of activation-projected gradient ascent...")
    model = load_trained_model()
    run_gpm_a_unlearning(model, forget_loader, act_bases, STEPS, LR)

    # Save checkpoint
    gpma_ckpt = "checkpoints/resnet18_cifar100_unlearn_gpma.pt"
    torch.save(model.state_dict(), gpma_ckpt)
    print(f"  Saved GPM-A checkpoint: {gpma_ckpt}")

    fa, ra, ta = compute_final_metrics(model, test_set, FORGET_CLASS)
    mia_a, mem_a, nmem_a = compute_mia_score(model, FORGET_CLASS)
    print(f"  [GPM-A] Forget={fa:.1f}% | Retain={ra:.1f}% | Test={ta:.1f}% | MIA={mia_a:.1f}%")
    del model
    del act_bases
    torch.cuda.empty_cache()

    gpma_results = {'forget': fa, 'retain': ra, 'test': ta,
                    'mia': mia_a, 'mem_loss': mem_a, 'nonmem_loss': nmem_a}

    # ── 3. GA unlearning ──
    print("\n" + "=" * 60)
    print(f"  STEP 3: GA unlearning (class {FORGET_CLASS}: {forget_class_name})")
    print("=" * 60)

    print(f"\n  [GA] Running {STEPS} steps of plain gradient ascent...")
    model = load_trained_model()
    run_ga_unlearning(model, forget_loader, STEPS, LR)

    # Save checkpoint
    ga_ckpt = "checkpoints/resnet18_cifar100_unlearn_ga.pt"
    torch.save(model.state_dict(), ga_ckpt)
    print(f"  Saved GA checkpoint: {ga_ckpt}")

    fg, rg, tg = compute_final_metrics(model, test_set, FORGET_CLASS)
    mia_g, mem_g, nmem_g = compute_mia_score(model, FORGET_CLASS)
    print(f"  [GA]    Forget={fg:.1f}% | Retain={rg:.1f}% | Test={tg:.1f}% | MIA={mia_g:.1f}%")
    del model
    torch.cuda.empty_cache()

    ga_results = {'forget': fg, 'retain': rg, 'test': tg,
                  'mia': mia_g, 'mem_loss': mem_g, 'nonmem_loss': nmem_g}

    # ── 4. Summary ──
    print_summary_table(FORGET_CLASS, forget_class_name,
                        (orig_mia, orig_mem, orig_nmem),
                        gpma_results, ga_results)

    # Print SVD dimension info
    print(f"\nRetain subspace dimensions (SVD k values per layer):")
    for layer, k in dims_info.items():
        print(f"  {layer}: {k}")


if __name__ == "__main__":
    main()
