# AI HW2: SimCLR Self-Supervised Learning

**Course:** 535505 Artificial Intelligence, NYCU Spring 2026
**Due:** 2026-05-03

## What this project does

This project implements [SimCLR](https://arxiv.org/abs/2002.05709), a self-supervised learning (SSL) method that trains a backbone network to produce useful image representations **without using any labels**.

The core idea: for each image, generate two randomly augmented versions. Train the network to produce similar feature vectors for both versions of the same image, while pushing features of different images apart. After training, the backbone has learned to capture the *content* of images, not just their surface appearance.

## Project structure

```
main.py                    — Entry point; runs all experiments
config.py                  — All hyperparameters in one place (edit here for ablations)
dataset.py                 — Data loading + augmentation pipelines for 4 datasets
model.py                   — Modified ResNet-18 backbone, projector head, supervised model
loss.py                    — NT-Xent contrastive loss
evaluate.py                — kNN monitor + linear probing evaluation
trainer.py                 — SSL and supervised learning training loops

simclr_colab.ipynb         — Main Colab notebook (all experiments, multi-dataset)
ablation_temperature.ipynb — Temperature ablation (τ = 0.1 / 0.5 / 5.0)
ablation_batchsize.ipynb   — Batch size ablation (32 → 512)
ablation_augmentation.ipynb— Augmentation strategy ablation (9 strategies)
```

## Supported datasets

All datasets are resized to **32×32** to match the modified ResNet-18 backbone.
All classes are used — no subsampling.

| Dataset | Classes | SSL training images | Notes |
|---|---|---|---|
| `cifar10` | 10 | 50k | Default |
| `stl10` | 10 | 105k | Uses train+unlabeled for SSL |
| `flowers102` | 102 | ~1k | Transfer learning target |
| `food101` | 101 | 75k | Transfer learning target |

## Key concepts

| Term | Plain English |
|---|---|
| **Backbone** | ResNet-18 that converts a 32×32 image into a 512-dim feature vector |
| **Projector head** | Small MLP on top of the backbone; used only during SSL, then discarded |
| **NT-Xent loss** | "Find which of the 2N images is the twin of this one" — forces the backbone to learn meaningful features |
| **kNN monitor** | Periodically checks if backbone features are useful — without needing labels |
| **Linear probing** | Freeze backbone, train one linear layer — the standard SSL quality benchmark |

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

```bash
uv run python main.py --batch-size 128
uv run python main.py --batch-size 64
```

### Step 4 — SSL only (skip supervised baseline)

```bash
uv run python main.py --skip-supervised
```

## Command-line flags

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `cifar10` | Dataset: `cifar10`, `stl10`, `flowers102`, `food101` |
| `--augmentation` | `best` | SSL augmentation strategy (see table below) |
| `--batch-size` | `256` | Override SSL batch size |
| `--temperature` | `0.5` | Override NT-Xent temperature |
| `--ssl-epochs` | `200` | Override SSL training epochs |
| `--skip-supervised` | off | Skip Experiment 2 (supervised baseline) |
| `--random-baseline` | off | Run Experiment 3 (random frozen backbone lower bound) |

## Augmentation strategies

Select with `--augmentation <name>`. All strategies use CIFAR-10 statistics for normalization (except `sobel`).

| Name | Pipeline | What the model learns to ignore |
|---|---|---|
| `best` | Crop + flip + colour + grayscale | Position, scale, colour ← **paper recommendation, default** |
| `crop` | RandomResizedCrop + flip | Position, scale only |
| `color` | ColorJitter + grayscale | Colour only |
| `rotate` | RandomRotation ± 30° + flip | Orientation |
| `blur` | GaussianBlur + flip | Fine texture / sharpness |
| `noise` | Gaussian additive noise + flip | Pixel-level perturbations |
| `cutout` | RandomErasing + flip | Random rectangular occlusion |
| `sobel` | Sobel edge filter | Everything except edge structure |
| `full` | All of the above combined | All invariances simultaneously |

Example:
```bash
uv run python main.py --augmentation rotate
uv run python main.py --augmentation crop
```

## Ablation experiments

### Temperature ablation
```bash
uv run python main.py --temperature 0.1   # sharp: strong gradient, may overfit easy pairs
uv run python main.py --temperature 0.5   # default
uv run python main.py --temperature 5.0   # flat: weak signal

# Or run all three with the dedicated notebook:
#   ablation_temperature.ipynb
```

### Batch size ablation
```bash
uv run python main.py --batch-size 512
uv run python main.py --batch-size 256
uv run python main.py --batch-size 128
uv run python main.py --batch-size 64
uv run python main.py --batch-size 32

# Or run all five with the dedicated notebook:
#   ablation_batchsize.ipynb
```

### Augmentation ablation
```bash
# Run all 9 strategies with the dedicated notebook:
#   ablation_augmentation.ipynb
# Or test one at a time:
uv run python main.py --augmentation crop --skip-supervised
uv run python main.py --augmentation best --skip-supervised
```

## Output files

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

| # | Experiment | Command / Notebook |
|---|---|---|
| 1 | SimCLR SSL + linear probing | `python main.py` |
| 2 | Supervised learning baseline | `python main.py` |
| 3 | Random frozen backbone (lower bound) | `python main.py --random-baseline` |
| 4 | Temperature ablation | `ablation_temperature.ipynb` |
| 5 | Batch size ablation | `ablation_batchsize.ipynb` |
| 6 | Augmentation strategy ablation | `ablation_augmentation.ipynb` |
| 7 | Multi-dataset / transfer learning | `--dataset stl10` / `flowers102` |

## Hyperparameters (from `config.py`)

| Parameter | Value | Notes |
|---|---|---|
| SSL epochs | 200 | Full SSL training duration |
| Batch size | 256 | Reduce if GPU memory is limited |
| Optimizer | Adam | LR 3e-4, weight decay 1e-6 |
| Temperature | 0.5 | NT-Xent loss scaling factor |
| Augmentation | best | crop + flip + colour + grayscale |
| Projector | 512 → 512 → 128 | Two-layer MLP |
| kNN k | 20 | Neighbors for kNN monitor |
| kNN interval | every 5 epochs | How often to run kNN check |
| Linear probe | 100 epochs | Adam, LR 1e-3, backbone frozen |

## References

- SimCLR: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020. https://arxiv.org/abs/2002.05709
- ResNet: He et al., "Deep Residual Learning for Image Recognition," CVPR 2016.
- PyTorch: https://pytorch.org
- torchvision: https://pytorch.org/vision
