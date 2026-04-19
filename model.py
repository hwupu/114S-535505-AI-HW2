"""
model.py — Neural network architectures for SimCLR and supervised learning.

There are three model classes here:

  1. ModifiedResNet18   — the "backbone" (feature extractor)
  2. SimCLRModel        — backbone + projector head (used during SSL training)
  3. SupervisedModel    — backbone + classification head (used for supervised baseline)

What is a "backbone"?
  A deep neural network that takes a raw image as input and outputs a compact
  feature vector that (ideally) captures the image's semantic content.
  Think of it as an automatic feature extractor.

Why ResNet-18?
  ResNet (Residual Network) is a well-known architecture that uses "skip connections"
  to train deep networks without the gradients vanishing. Version "18" refers to the
  number of layers (18 layers deep). It's a good balance of speed and accuracy.

Why must we modify it for CIFAR-10?
  Original ResNet-18 was designed for ImageNet (224×224 images).
  Its first layer aggressively downsamples: 224×224 → 56×56 (4× reduction).
  For CIFAR-10 (32×32), the same reduction gives only 8×8 feature maps — too small.
  We remove the aggressive early downsampling to preserve spatial information.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ModifiedResNet18(nn.Module):
    """
    ResNet-18 with its first layers adjusted for 32×32 CIFAR-10 images.

    Changes from the original ResNet-18:
      - First conv:   kernel 7×7, stride 2, padding 3  →  kernel 3×3, stride 1, padding 1
      - Max-pool:     3×3 with stride 2                →  Identity (pass-through, no change)

    These two changes mean the feature maps stay at 32×32 through the first block
    instead of immediately dropping to 8×8, giving the network more spatial detail
    to work with at the early layers.

    Output: a 512-dimensional feature vector per image (after global average pooling).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # Load a standard ResNet-18 (with or without ImageNet pretrained weights).
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # --- Modification 1: Replace the first convolution layer ---
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # New:      Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # This keeps the spatial size unchanged (32→32) instead of halving it (32→16).
        resnet.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        # --- Modification 2: Remove the max-pool layer ---
        # Original max-pool would halve spatial size again (32→16 or 16→8).
        # nn.Identity() is a no-op layer: output = input.
        resnet.maxpool = nn.Identity()

        # --- Remove the final fully-connected (classification) layer ---
        # We want the raw 512-dim feature vector, not class predictions.
        # resnet.children() gives: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # We take everything EXCEPT the last element (fc) by using [:-1].
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        # After encoder: shape is [batch_size, 512, 1, 1] (after global average pool)

        self.output_dim = 512  # Useful to reference when building downstream heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, 3, 32, 32]
        features = self.encoder(x)                       # → [batch_size, 512, 1, 1]
        features = torch.flatten(features, start_dim=1)  # → [batch_size, 512]
        return features


class ProjectorHead(nn.Module):
    """
    Two-layer MLP that projects backbone features into a lower-dimensional space.

    In SimCLR, the contrastive NT-Xent loss is computed on the PROJECTOR output,
    not the backbone output. This is a key insight from the paper:
    the projection discards information that is useful for downstream tasks
    (like specific colors or textures) but not needed for contrastive learning.
    Keeping this information in the backbone improves downstream performance.

    Architecture: 512 → ReLU → 512 → 128  (no activation after the last layer)
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),     # inplace=True saves a tiny bit of memory
            nn.Linear(hidden_dim, output_dim),
            # No ReLU here — the output can be negative; L2 normalization handles scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SimCLRModel(nn.Module):
    """
    Full SimCLR model: Backbone (ResNet-18) + Projector Head (MLP).

    During SSL training:  both backbone and projector are used.
                          Loss is computed on projector output (128-dim).
    After SSL training:   projector is discarded.
                          Only backbone output (512-dim) is used as "the representation".

    The forward() method returns BOTH the backbone output and the projector output.
    Use encode() when you just want the backbone features (for evaluation).
    """

    def __init__(self, pretrained_backbone: bool = False,
                 projector_hidden: int = 512, projector_out: int = 128):
        super().__init__()
        self.backbone  = ModifiedResNet18(pretrained=pretrained_backbone)
        self.projector = ProjectorHead(
            input_dim=self.backbone.output_dim,
            hidden_dim=projector_hidden,
            output_dim=projector_out
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (backbone_features, projector_output)."""
        h = self.backbone(x)    # 512-dim representation
        z = self.projector(h)   # 128-dim projection (used for contrastive loss)
        return h, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: returns only the backbone features (512-dim)."""
        return self.backbone(x)


class SupervisedModel(nn.Module):
    """
    Supervised learning baseline: same backbone as SimCLR, but with a direct
    classification head instead of a projector head.

    Trained end-to-end using cross-entropy loss on labeled CIFAR-10 data.
    This is the "traditional" way to train a classifier — compare its accuracy
    against SSL + linear probing to see how the two approaches differ.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone   = ModifiedResNet18(pretrained=False)  # always train from scratch
        # Linear classification head: 512 backbone features → 10 class scores
        self.classifier = nn.Linear(self.backbone.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class logits (raw unnormalized scores) for each image."""
        features = self.backbone(x)      # [batch_size, 512]
        logits   = self.classifier(features)  # [batch_size, 10]
        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns backbone features — used when evaluating transfer learning."""
        return self.backbone(x)
