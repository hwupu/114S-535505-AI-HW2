"""
trainer.py — Training loops for SimCLR (SSL) and Supervised Learning.

What is a "training loop"?
  Machine learning is an iterative optimization process:
    1. Feed a batch of data through the model → get predictions
    2. Measure how wrong the predictions are (the "loss")
    3. Compute gradients: how does each parameter affect the loss?
    4. Update parameters in the direction that reduces the loss
    5. Repeat for many "epochs" (passes over the full dataset)

What is an "epoch"?
  One complete pass over the entire training dataset.
  With 50,000 CIFAR-10 training images and batch size 256,
  one epoch = 50000 / 256 ≈ 195 batches.

What is a "gradient"?
  A vector that points in the direction of steepest increase in the loss.
  We step in the OPPOSITE direction (gradient descent) to reduce the loss.
  PyTorch computes gradients automatically via `loss.backward()`.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import config
from loss import NTXentLoss
from evaluate import knn_monitor, evaluate_accuracy


def train_simclr(model: nn.Module,
                 ssl_loader: DataLoader,
                 eval_train_loader: DataLoader,
                 eval_test_loader: DataLoader,
                 device: torch.device) -> dict:
    """
    SimCLR Self-Supervised Learning training loop.

    Unlike supervised training, we NEVER look at the image labels.
    The only training signal comes from the contrastive loss:
    "make the two augmented views of the same image similar,
     and push views of different images apart."

    Args:
        model:             SimCLRModel (backbone + projector).
        ssl_loader:        DataLoader yielding ((view1, view2), label) batches.
        eval_train_loader: Clean (no augment) train loader for kNN evaluation.
        eval_test_loader:  Clean (no augment) test  loader for kNN evaluation.
        device:            Torch device (cuda / mps / cpu).

    Returns:
        history dict with keys:
          'loss'    : list of average loss per epoch
          'knn_acc' : list of (epoch, knn_accuracy) tuples
    """
    optimizer = Adam(model.parameters(), lr=config.SSL_LR,
                     weight_decay=config.SSL_WEIGHT_DECAY)
    criterion = NTXentLoss(temperature=config.TEMPERATURE)

    history = {'loss': [], 'knn_acc': []}

    print(f"\n{'='*65}")
    print(f"  SimCLR SSL Training")
    print(f"  Epochs: {config.SSL_EPOCHS}  |  Batch size: {config.SSL_BATCH_SIZE}  "
          f"|  Temperature: {config.TEMPERATURE}")
    print(f"{'='*65}")

    for epoch in range(1, config.SSL_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for (view1, view2), _ in ssl_loader:
            # view1, view2: two random augmentations of the SAME images, shape [N, 3, 32, 32]
            # _           : labels — intentionally ignored (this is UN-supervised learning)
            view1, view2 = view1.to(device), view2.to(device)

            # --- Forward pass ---
            # Each view goes through backbone + projector → 128-dim projection
            _, z_i = model(view1)   # z_i: [N, 128]
            _, z_j = model(view2)   # z_j: [N, 128]

            # --- Compute contrastive loss ---
            loss = criterion(z_i, z_j)

            # --- Backward pass + parameter update ---
            optimizer.zero_grad()   # Clear gradients from the previous step
            loss.backward()         # Compute new gradients
            optimizer.step()        # Update all parameters

            total_loss += loss.item()

        avg_loss = total_loss / len(ssl_loader)
        history['loss'].append(avg_loss)

        # --- kNN Monitor (runs every KNN_INTERVAL epochs) ---
        # This is the only way to observe learning progress in SSL,
        # since the loss alone doesn't tell us about representation quality.
        run_knn = (epoch % config.KNN_INTERVAL == 0)
        if run_knn:
            knn_acc = knn_monitor(
                model, eval_train_loader, eval_test_loader, device,
                k=config.KNN_K, temperature=config.KNN_TEMPERATURE,
                num_classes=config.NUM_CLASSES
            )
            history['knn_acc'].append((epoch, knn_acc))

        # --- Print progress ---
        if epoch % 10 == 0 or epoch == 1:
            knn_str = f"  kNN: {knn_acc*100:.2f}%" if run_knn and epoch % 10 == 0 else ""
            print(f"  Epoch {epoch:3d}/{config.SSL_EPOCHS}  "
                  f"loss={avg_loss:.4f}{knn_str}")

    print(f"\n  SSL Training complete.")
    return history


def train_supervised(model: nn.Module,
                     train_loader: DataLoader,
                     test_loader: DataLoader,
                     device: torch.device) -> dict:
    """
    Standard supervised classification training loop.

    Unlike SimCLR, we DO use the image labels here.
    The loss is cross-entropy: penalize the model when its predicted
    class probability for the correct label is low.

    Args:
        model:        SupervisedModel (backbone + linear classifier).
        train_loader: DataLoader with standard augmentation.
        test_loader:  DataLoader with no augmentation (for evaluation).
        device:       Torch device.

    Returns:
        history dict with keys:
          'loss'     : list of average training loss per epoch
          'test_acc' : list of (epoch, test_accuracy) tuples
    """
    optimizer = Adam(model.parameters(), lr=config.SL_LR,
                     weight_decay=config.SL_WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    history = {'loss': [], 'test_acc': []}

    print(f"\n{'='*65}")
    print(f"  Supervised Learning Training (baseline)")
    print(f"  Epochs: {config.SL_EPOCHS}  |  Batch size: {config.SSL_BATCH_SIZE}")
    print(f"{'='*65}")

    for epoch in range(1, config.SL_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)             # [batch, 10]
            loss   = loss_fn(logits, labels)   # cross-entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history['loss'].append(avg_loss)

        # Evaluate on test set every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            test_acc = evaluate_accuracy(model, test_loader, device)
            history['test_acc'].append((epoch, test_acc))
            print(f"  Epoch {epoch:3d}/{config.SL_EPOCHS}  "
                  f"loss={avg_loss:.4f}  test_acc={test_acc*100:.2f}%")

    print(f"\n  Supervised Training complete.")
    return history
