"""
Loss Functions for Face Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss: L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)

    Minimizes distance between anchor-positive pairs
    Maximizes distance between anchor-negative pairs
    """

    def __init__(self, margin: float = 0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(
            self,
            anchor: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate triplet loss

        Args:
            anchor: Anchor embeddings [B, embedding_dim]
            positive: Positive embeddings [B, embedding_dim]
            negative: Negative embeddings [B, embedding_dim]

        Returns:
            Scalar loss value
        """
        # Calculate Euclidean distances
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss with margin
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()