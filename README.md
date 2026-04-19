# AI HW2: SimCLR Self-Supervised Learning on CIFAR-10

**Course:** 535505 Artificial Intelligence, NYCU Spring 2026
**Due:** 2026-05-03

## What this project does

This project implements [SimCLR](https://arxiv.org/abs/2002.05709), a self-supervised learning (SSL) method that trains a backbone network to produce useful image representations **without using any labels**.

The core idea: for each image, generate two randomly augmented versions. Train the network to produce similar feature vectors for both versions of the same image, while pushing features of different images apart. After training, the backbone has learned to capture the *content* of images, not just their surface appearance.

## Project structure

```
main.py        — Entry point; runs all experiments
config.py      — All hyperparameters in one place (edit here for ablations)
dataset.py     — CIFAR-10 data loading + SimCLR double-augmentation pipeline
model.py       — Modified ResNet-18 backbone, projector head, supervised model
loss.py        — NT-Xent contrastive loss
evaluate.py    — kNN monitor + linear probing evaluation
trainer.py     — SSL and supervised learning training loops
```

## Key concepts

| Term | Plain English |
|---|---|
| **Backbone** | ResNet-18 network that converts a 32×32 image into a 512-dimensional feature vector |
| **Projector head** | Small MLP on top of the backbone; used only during SSL training, then discarded |
| **NT-Xent loss** | "Find which of the 2N images is the twin of this one" — a matching game that forces the backbone to learn meaningful features |
| **kNN monitor** | During training, periodically check if backbone features are useful — without needing labels |
| **Linear probing** | After SSL training: freeze the backbone, attach one linear layer, train only that layer — the standard benchmark for SSL quality |

## How to run

### Step 1 — Install dependencies

```bash
uv sync
```

> **If you have an NVIDIA GPU**, install the CUDA version of PyTorch for much faster training.
> Visit https://pytorch.org/get-started/locally/ for the correct command.

### Step 2 — Run all required experiments

```bash
uv run python main.py
```

This will:
1. Download CIFAR-10 automatically (~170 MB) into `./data/`
2. Train SimCLR for 200 epochs (prints loss + kNN accuracy every 10 epochs)
3. Run linear probing on the SSL backbone (100 epochs, backbone frozen)
4. Train a supervised learning baseline (200 epochs, same backbone architecture)
5. Save plots to `./results/` and model checkpoints to `./checkpoints/`

### Step 3 — If you run out of GPU memory

Reduce the batch size:

```bash
uv run python main.py --batch-size 128
# or even smaller:
uv run python main.py --batch-size 64
```

### Step 4 — SSL only (skip supervised baseline)

```bash
uv run python main.py --skip-supervised
```

## Ablation experiments (optional report sections)

All settings can be overridden from the command line:

```bash
# Temperature ablation (default is 0.5)
uv run python main.py --temperature 0.1   # very sharp — strong gradient signal
uv run python main.py --temperature 5.0   # very flat  — weak gradient signal

# Batch size ablation
uv run python main.py --batch-size 512
uv run python main.py --batch-size 256
uv run python main.py --batch-size 128
uv run python main.py --batch-size 64
uv run python main.py --batch-size 32

# Shorter run for quick testing
uv run python main.py --ssl-epochs 50
```

Alternatively, edit `config.py` directly to change any hyperparameter permanently.

## Output files

After running, check these directories:

```
results/
  ssl_curves.png    — SSL training loss + kNN accuracy over epochs
  sl_curves.png     — Supervised training loss + test accuracy over epochs
  summary.txt       — Final accuracy numbers for your report table

checkpoints/
  simclr.pt         — Trained SimCLR model weights
  supervised.pt     — Trained supervised model weights
```

## Experiment roadmap

| # | Experiment | Status | Command |
|---|---|---|---|
| 1 | SimCLR SSL baseline + linear probing | Required | `python main.py` |
| 2 | Supervised learning baseline | Required | `python main.py` |
| 3 | Random frozen backbone (lower bound) | Optional | TBD |
| 4 | Temperature ablation (0.1, 0.5, 5.0) | Optional | `--temperature X` |
| 5 | Batch size ablation (32 → 512) | Optional | `--batch-size X` |
| 6 | No projector head (use backbone output for loss) | Optional | TBD |
| 7 | Use projector output as representation | Optional | TBD |
| 8 | Transfer to CIFAR-100 / STL-10 | Optional | TBD |

## Hyperparameters (from `config.py`)

| Parameter | Value | Notes |
|---|---|---|
| SSL epochs | 200 | Full SSL training duration |
| Batch size | 256 | Reduce if GPU memory is limited |
| Optimizer | Adam | Learning rate 3e-4, weight decay 1e-6 |
| Temperature | 0.5 | NT-Xent loss scaling factor |
| Projector | 512 → 512 → 128 | Two-layer MLP |
| kNN k | 20 | Neighbors for kNN monitor |
| kNN interval | every 5 epochs | How often to run kNN check |
| Linear probe | 100 epochs | Adam, LR 1e-3, backbone frozen |

## References

- SimCLR paper: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (2020). https://arxiv.org/abs/2002.05709
- ResNet: He et al., "Deep Residual Learning for Image Recognition" (2016).
- PyTorch: https://pytorch.org
- torchvision: https://pytorch.org/vision
