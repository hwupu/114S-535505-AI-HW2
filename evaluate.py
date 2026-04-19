"""
evaluate.py — kNN Monitor and Linear Probing.

Two ways to measure how good the learned representations are:

  kNN Monitor (used DURING training to track progress):
    Given a test image, find its k nearest neighbors in the training set
    using the backbone features.  Predict the class by majority vote.
    No training required — it directly tests whether the backbone features
    are meaningful.  If the SSL loss decreases but kNN accuracy stays flat,
    the model is not learning useful representations.

  Linear Probing (used AFTER SSL training for the final evaluation):
    Freeze the backbone completely.  Add one linear layer on top.
    Train only that layer on labeled data.
    A good representation should be linearly separable — meaning a simple
    linear boundary in feature space should be enough to classify images.
    This is THE standard benchmark for SSL methods.

Why not just use test accuracy as a measure during SSL training?
  SSL does not use labels at all, so there is no natural "test accuracy"
  during training.  kNN is a label-free proxy that correlates well with
  the final linear probing accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helper: extract all features from a dataset in one pass
# ---------------------------------------------------------------------------

@torch.no_grad()   # Decorator: disables gradient tracking (saves memory + speed)
def extract_features(model: nn.Module, loader: DataLoader,
                     device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run all images through the model backbone and collect the feature vectors.

    Returns:
        features: FloatTensor of shape [N, 512]
        labels:   LongTensor  of shape [N]
    """
    model.eval()   # Important: switches BatchNorm to use running statistics (not batch stats)

    all_features = []
    all_labels   = []

    for images, labels in loader:
        images = images.to(device)
        features = model.encode(images)   # [batch, 512] — backbone only, no projector
        all_features.append(features.cpu())   # Move back to CPU to avoid GPU memory buildup
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# kNN Monitor
# ---------------------------------------------------------------------------

@torch.no_grad()
def knn_monitor(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                device: torch.device, k: int = 20, temperature: float = 0.1,
                num_classes: int = 10) -> float:
    """
    Evaluate representation quality using k-Nearest Neighbor classification.

    Algorithm:
      1. Extract features for all training images → these become the "database".
      2. For each test image, find its k most similar training images
         using cosine similarity.
      3. Predict the test image's class by weighted voting:
         - Each neighbor casts a vote for its class.
         - Vote weight = softmax(similarity / temperature).
         - Higher similarity → larger weight.
      4. Return accuracy = correct predictions / total test images.

    Args:
        model:        The SimCLR model with an .encode() method.
        train_loader: DataLoader for training set (eval transform, no augmentation).
        test_loader:  DataLoader for test set.
        device:       Torch device.
        k:            Number of nearest neighbors (default 20 per assignment spec).
        temperature:  Softmax temperature for vote weighting.
        num_classes:  Number of output classes.

    Returns:
        Accuracy as a float in [0, 1].
    """
    # Step 1: build the feature database from the training set
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features,  test_labels  = extract_features(model, test_loader,  device)

    # Normalize to unit length so dot product = cosine similarity
    train_features = F.normalize(train_features, dim=1).to(device)
    test_features  = F.normalize(test_features,  dim=1).to(device)
    train_labels   = train_labels.to(device)

    correct = 0
    total   = 0

    # Process test set in mini-batches to avoid running out of GPU memory
    # (computing similarity between all test and all train at once is huge)
    chunk_size = 512
    for start in range(0, len(test_features), chunk_size):
        end = min(start + chunk_size, len(test_features))
        batch_feats  = test_features[start:end]   # [chunk, 512]
        batch_labels = test_labels[start:end]     # [chunk]

        # Cosine similarity: batch_feats @ train_features.T → [chunk, N_train]
        sim = torch.mm(batch_feats, train_features.T)   # [chunk, N_train]

        # Scale by temperature and get top-k similarities + their indices
        sim_scaled = sim / temperature
        topk_vals, topk_idx = sim_scaled.topk(k, dim=1)   # both [chunk, k]

        # Convert similarities to weights via softmax (along the k-neighbor axis)
        weights = F.softmax(topk_vals, dim=1)   # [chunk, k]

        # Labels of the k nearest neighbors
        neighbor_labels = train_labels[topk_idx]   # [chunk, k]

        # Weighted vote: for each of the chunk images, accumulate weight per class
        vote_scores = torch.zeros(end - start, num_classes, device=device)
        vote_scores.scatter_add_(
            dim=1,
            index=neighbor_labels,   # which class gets the weight
            src=weights              # how much weight to add
        )

        # Predicted class = class with highest total weight
        predictions = vote_scores.argmax(dim=1)
        correct += (predictions == batch_labels).sum().item()
        total   += len(batch_labels)

    return correct / total


# ---------------------------------------------------------------------------
# Linear Probing
# ---------------------------------------------------------------------------

def linear_probing(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                   device: torch.device, num_classes: int = 10,
                   epochs: int = 100, lr: float = 1e-3,
                   weight_decay: float = 1e-6) -> float:
    """
    Evaluate SSL representations by training a single linear layer on top.

    The backbone is FROZEN (weights are not updated).
    Only the new linear layer's weights are trained.

    This tests whether the backbone has learned linearly separable features,
    which is the gold standard for self-supervised representation quality.

    Returns:
        Final test accuracy as a float in [0, 1].
    """
    # --- Freeze backbone ---
    # requires_grad=False tells PyTorch not to compute or store gradients
    # for these parameters, saving memory and computation.
    for param in model.backbone.parameters():
        param.requires_grad = False

    model.eval()   # Keep backbone in eval mode (BatchNorm uses running stats)

    # --- Create a fresh linear classifier ---
    feature_dim = model.backbone.output_dim   # 512
    linear = nn.Linear(feature_dim, num_classes).to(device)

    # We only pass linear.parameters() to the optimizer — backbone params are excluded
    optimizer = Adam(linear.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.CrossEntropyLoss()

    print(f"\n  [Linear Probing] Training linear layer for {epochs} epochs "
          f"(backbone frozen)")

    for epoch in range(1, epochs + 1):
        linear.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Extract backbone features — no gradient needed (backbone is frozen)
            with torch.no_grad():
                features = model.encode(images)   # [batch, 512]

            # Forward through the linear layer
            logits = linear(features)   # [batch, num_classes]
            loss   = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()      # Only computes gradients for linear layer
            optimizer.step()

            total_loss += loss.item()

        # Print progress every 20 epochs
        if epoch % 20 == 0:
            acc = _eval_linear(linear, model, test_loader, device)
            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"test_acc={acc*100:.2f}%")

    final_acc = _eval_linear(linear, model, test_loader, device)
    print(f"  Linear Probing Final Test Accuracy: {final_acc*100:.2f}%")

    # --- Unfreeze backbone for any future use ---
    for param in model.backbone.parameters():
        param.requires_grad = True

    return final_acc


@torch.no_grad()
def _eval_linear(linear: nn.Linear, model: nn.Module,
                 loader: DataLoader, device: torch.device) -> float:
    """Evaluate accuracy of a linear layer on top of frozen backbone."""
    linear.eval()
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        features    = model.encode(images)
        predictions = linear(features).argmax(dim=1)
        correct    += (predictions == labels).sum().item()
        total      += len(labels)
    return correct / total


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader,
                      device: torch.device) -> float:
    """
    Measure the classification accuracy of a fully supervised model
    (one that produces class logits directly from forward()).
    """
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits      = model(images)
        predictions = logits.argmax(dim=1)
        correct    += (predictions == labels).sum().item()
        total      += len(labels)
    return correct / total
