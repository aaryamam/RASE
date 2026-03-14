# RASE — Retain-Activation Subspace Erasure

**RASE** is a machine unlearning method that erases a model's knowledge of a specific class by projecting gradient updates away from the retain-set's activation subspace during gradient ascent. This keeps retained classes intact while efficiently forgetting a target class — without retraining from scratch.

This repository compares RASE against three baselines on CIFAR-10 and Tiny-ImageNet:
- **GA** — plain Gradient Ascent on forget-class data
- **GPM-W** — Weight-space projected gradient ascent (ResNet only)
- **SCRUB** — Teacher-Student KL unlearning

---

## Method Overview

Standard gradient ascent on forget-class data degrades retain-class accuracy quickly. RASE fixes this by:

1. **Building a retain subspace** — Forward-pass retain-set samples through the model and collect layer activations. Run SVD to find the principal directions that span the retain-set's activation space.
2. **Projecting gradients at each step** — During gradient ascent on forget-class batches, backward hooks project each layer's gradient *away* from the retain subspace before the weight update. This confines forgetting to directions irrelevant to the retained classes.

RASE works on **ResNet-18**, **ResNet-50**, and **ViT-Small** models. ViT unlearning uses Adam optimizer with selective block hooking for stable gradient projection.

---

## Repository Structure

```
RASE/
├── scripts/
│   ├── train_resnet.py                # Train ResNet-18 on CIFAR-10
│   ├── train_vit.py                   # Train ViT-Small on CIFAR-10
│   ├── train_resnet50_tinyimagenet.py # Train ResNet-50 on Tiny-ImageNet
│   ├── unlearn.py                     # Run GA / GPM-W / RASE / SCRUB
│   └── evaluate.py                    # Evaluate unlearned checkpoints with MIA metrics
├── old_scripts/                   # Various prior experimentation scripts
├── checkpoints/                   # Saved model weights (not tracked by git)
├── data/                          # CIFAR/Tiny-ImageNet datasets (auto-downloaded)
├── REPO_EXPLAINED.md              # In-depth explanation of all scripts and concepts
└── README.md
```

---

## Setup

**Requirements:** Python 3.9+, PyTorch ≥ 2.0, torchvision, numpy

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision numpy
```

> Datasets (CIFAR-10 / Tiny-ImageNet) are downloaded automatically to `./data/` on first run.

---

## Quick Start

### 1. Train a model

```bash
# Train ResNet-18 on CIFAR-10
python scripts/train_resnet.py

# Train ViT-Small on CIFAR-10
python scripts/train_vit.py

# Train ResNet-50 on Tiny-ImageNet
python scripts/train_resnet50_tinyimagenet.py
```

Checkpoints are saved to `checkpoints/`.

### 2. Run unlearning

Open `scripts/unlearn.py` and set the overarching configuration at the top:

```python
MODEL_ARCH      = "resnet50_tinyimagenet"          # "resnet18", "vit_small", or "resnet50_tinyimagenet"
FORGET_CLASSES  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # class index to forget (0-9 for CIFAR, 0-199 for TinyImageNet)
UNLEARN_STEPS   = 200                 # gradient ascent steps
UNLEARN_LR      = 1e-3                # learning rate for all methods
SVD_THRESHOLD   = 0.95                # variance threshold for subspace construction
```

Then run unlearning via CLI by passing the desired methods:

```bash
# Run all methods
python scripts/unlearn.py

# Run specific methods
python scripts/unlearn.py --methods ga gpm_w rase scrub
```

Unlearned checkpoints are saved to `checkpoints/unlearned/<arch>_forget<class>/`.

### 3. Evaluate

Open `scripts/evaluate.py` to ensure it is targeting the correct unlearning checkpoints, then run:

```bash
python scripts/evaluate.py
```

Reports Forget accuracy, Retain accuracy, Test accuracy, and **MIA balanced accuracy** (ideally ≈ 50% after unlearning, indicating random-guess level membership inference).

---

## Configuration Reference

Key variables to tune in `scripts/unlearn.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_ARCH` | `"resnet50_tinyimagenet"` | Architecture: `"resnet18"`, `"vit_small"`, `"resnet50_tinyimagenet"` |
| `FORGET_CLASSES`| `[0, ...]` | Class indices to unlearn |
| `UNLEARN_STEPS` | `200` | Number of gradient ascent steps |
| `UNLEARN_LR` | `1e-3` | Learning rate for unlearning |
| `SVD_THRESHOLD` | `0.95` | Fraction of activation variance to span with basis |
| `SAMPLES_PER_CLASS` | `20` | Retain samples per class used to build SVD basis |

---

## Acknowledgements
This repository includes an implementation of the SCRUB baseline for comparison purposes. We thank the authors for making their code publicly available. 

* **SCRUB:** Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. "Towards Unbounded Machine Unlearning." *NeurIPS*, 2024. 
  * Original implementation: [https://github.com/meghdadk/SCRUB.git](https://github.com/meghdadk/SCRUB.git)

## Citation / Acknowledgements

This project is a research implementation. If you build on this work, please cite appropriately.
