# RASE — Retain-Activation Subspace Erasure

**RASE** is a machine unlearning method that erases a model's knowledge of a specific class by projecting gradient updates away from the retain-set's activation subspace during gradient ascent. This keeps retained classes intact while efficiently forgetting a target class — without retraining from scratch.

This repository compares RASE against two baselines on CIFAR-10 and CIFAR-100:
- **GA** — plain Gradient Ascent on forget-class data
- **GPM-W** — Weight-space projected gradient ascent (ResNet-18 only)

---

## Method Overview

Standard gradient ascent on forget-class data degrades retain-class accuracy quickly. RASE fixes this by:

1. **Building a retain subspace** — Forward-pass retain-set samples through the model and collect layer activations. Run SVD to find the principal directions that span the retain-set's activation space.
2. **Projecting gradients at each step** — During gradient ascent on forget-class batches, backward hooks project each layer's gradient *away* from the retain subspace before the weight update. This confines forgetting to directions irrelevant to the retained classes.

RASE works on both **ResNet-18** (hooks on `layer1`–`layer4`) and **ViT-Small** (hooks on transformer blocks).

---

## Repository Structure

```
RASE/
├── scripts/
│   ├── train_resnet.py    # Train ResNet-18 on CIFAR-10 (200 epochs, SGD + cosine LR)
│   ├── train_vit.py       # Train ViT-Small on CIFAR-10
│   ├── unlearn.py         # Run GA / GPM-W / RASE unlearning (switch arch at top)
│   └── evaluate.py        # Evaluate unlearned checkpoints with MIA metrics
├── checkpoints/           # Saved model weights (not tracked by git — see .gitignore)
├── data/                  # CIFAR datasets (auto-downloaded, not tracked by git)
├── REPO_EXPLAINED.md      # In-depth explanation of all scripts and concepts
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

> Datasets (CIFAR-10 / CIFAR-100) are downloaded automatically to `./data/` on first run.

---

## Quick Start

### 1. Train a model

```bash
# Train ResNet-18 on CIFAR-10
python scripts/train_resnet.py

# Train ViT-Small on CIFAR-10
python scripts/train_vit.py
```

Checkpoints are saved to `checkpoints/`.

### 2. Run unlearning

Open `scripts/unlearn.py` and set the configuration at the top:

```python
MODEL_ARCH   = "resnet18"   # or "vit_small"
FORGET_CLASS = 0            # class index to forget (0–9)
UNLEARN_STEPS = 800
UNLEARN_LR    = 1e-3
SVD_THRESHOLD = 0.95        # fraction of activation variance to retain
```

Then run:

```bash
python scripts/unlearn.py
```

Unlearned checkpoints are saved to `checkpoints/unlearned/<arch>_forget<class>/`.

### 3. Evaluate

```bash
python scripts/evaluate.py
```

Reports Forget accuracy, Retain accuracy, Test accuracy, and **MIA balanced accuracy** (ideally ≈ 50% after unlearning, indicating random-guess level membership inference).

---

## Key Results

| Method  | Forget Acc ↓ | Retain Acc ↑ | Test Acc ↑ | MIA Acc → 50% |
|---------|:------------:|:------------:|:----------:|:-------------:|
| Original | high        | high         | high       | high          |
| GA      | low          | degrades     | degrades   | ~50%          |
| GPM-W   | low          | moderate     | moderate   | ~50%          |
| **RASE**| **low**      | **preserved**| **preserved** | **~50%**   |

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_ARCH` | `"resnet18"` | Architecture: `"resnet18"` or `"vit_small"` |
| `FORGET_CLASS` | `0` | Class index to unlearn |
| `UNLEARN_STEPS` | `800` | Number of gradient ascent steps |
| `UNLEARN_LR` | `1e-3` | Learning rate for unlearning |
| `SVD_THRESHOLD` | `0.95` | Fraction of activation variance to span with basis |
| `SAMPLES_PER_CLASS` | `500` | Retain samples per class used to build SVD basis |

---

## Citation / Acknowledgements

This project is a research implementation. If you build on this work, please cite appropriately.
