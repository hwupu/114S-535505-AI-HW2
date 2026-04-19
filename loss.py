"""
loss.py — NT-Xent Loss (Normalized Temperature-scaled Cross Entropy).

This is the core of SimCLR.  Here is the intuition:

  Imagine you have N images in a batch.  Each image gets augmented TWICE,
  so you end up with 2N images total.  For each image, its "positive pair"
  is the other augmented version of the SAME source image.  Every other
  image in the batch is a "negative" — a distractor.

  The loss asks: "Given image view A, can you identify its twin B from the
  2N-1 other views in the batch?"

  Concretely:
    1. Compute cosine similarity between every pair of images.
    2. Divide similarities by a "temperature" τ to control sharpness.
    3. Apply softmax — this gives a probability distribution over all pairs.
    4. Loss = -log(probability assigned to the TRUE positive pair).
    5. Average over all 2N images.

  This is exactly cross-entropy loss where each image's "class" is the index
  of its twin in the batch.

Temperature intuition:
  - τ = 0.5 (default): Moderate sharpness.  Most used in practice.
  - τ → 0  (e.g. 0.1): Very sharp — tiny similarity differences get huge gradients.
                        The model tries very hard to separate every pair, which can
                        cause instability but fast learning.
  - τ → ∞  (e.g. 5.0): Very flat — all pairs look equally similar to the model.
                        Gradients are tiny, learning is very slow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent contrastive loss from the SimCLR paper (Chen et al., 2020).

    Args:
        temperature (float): Scaling factor τ.  Default 0.5.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of positive pairs.

        Args:
            z_i: Projections of the FIRST  augmentation.  Shape [N, D].
            z_j: Projections of the SECOND augmentation.  Shape [N, D].
                 z_i[k] and z_j[k] come from the same source image k.

        Returns:
            Scalar loss value (average over all 2N images).
        """
        N = z_i.shape[0]   # number of images in the original batch

        # ----------------------------------------------------------------
        # Step 1: L2-normalize every projection vector.
        # After normalization, the dot product of two vectors equals their
        # cosine similarity (a number in [-1, 1]).
        # ----------------------------------------------------------------
        z_i = F.normalize(z_i, dim=1)   # [N, D]
        z_j = F.normalize(z_j, dim=1)   # [N, D]

        # ----------------------------------------------------------------
        # Step 2: Concatenate into one matrix of 2N vectors.
        # Layout: [ z_i_0, ..., z_i_{N-1},  z_j_0, ..., z_j_{N-1} ]
        #   indices 0..N-1  correspond to the first augmentation views
        #   indices N..2N-1 correspond to the second augmentation views
        # ----------------------------------------------------------------
        z = torch.cat([z_i, z_j], dim=0)   # [2N, D]

        # ----------------------------------------------------------------
        # Step 3: Compute the full 2N × 2N cosine similarity matrix, scaled
        # by temperature.  sim[a, b] = cos(z_a, z_b) / τ
        # ----------------------------------------------------------------
        sim = torch.mm(z, z.T) / self.temperature   # [2N, 2N]

        # ----------------------------------------------------------------
        # Step 4: Build the ground-truth positive-pair labels.
        # For index a in range [0, N):   its positive is at a + N  (the z_j twin)
        # For index a in range [N, 2N):  its positive is at a - N  (the z_i twin)
        # ----------------------------------------------------------------
        labels = torch.arange(N, device=z_i.device)
        labels = torch.cat([labels + N, labels], dim=0)   # [2N]

        # ----------------------------------------------------------------
        # Step 5: Mask out self-similarity (diagonal).
        # Without this, each image would always "match" itself with similarity 1/τ,
        # which would dominate the softmax and give a trivially low loss.
        # Setting diagonal entries to -inf makes softmax(−∞) = 0, effectively
        # removing self-comparisons from the probability distribution.
        # ----------------------------------------------------------------
        self_mask = torch.eye(2 * N, dtype=torch.bool, device=z_i.device)
        sim = sim.masked_fill(self_mask, float('-inf'))

        # ----------------------------------------------------------------
        # Step 6: Cross-entropy loss.
        # F.cross_entropy(logits, targets) computes:
        #   loss = mean_over_rows( -logits[row, target[row]]
        #                          + log( sum_j( exp(logits[row, j]) ) ) )
        # Which is exactly -log( softmax(sim)[row, positive_index] ).
        # ----------------------------------------------------------------
        loss = F.cross_entropy(sim, labels)

        return loss
