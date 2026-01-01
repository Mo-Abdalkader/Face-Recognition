"""
Evaluation Metrics for Face Recognition
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


class FaceRecognitionMetrics:
    """
    Calculate validation metrics for face recognition
    """

    @staticmethod
    def cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """Calculate cosine similarity between embeddings"""
        return F.cosine_similarity(embedding1, embedding2, dim=1)

    @staticmethod
    def calculate_metrics(
            anchor_embeddings: torch.Tensor,
            positive_embeddings: torch.Tensor,
            negative_embeddings: torch.Tensor,
            threshold: float = 0.6
    ) -> dict:
        """
        Calculate accuracy, precision, recall

        Args:
            anchor_embeddings: Anchor embeddings [N, D]
            positive_embeddings: Positive embeddings [N, D]
            negative_embeddings: Negative embeddings [N, D]
            threshold: Similarity threshold for matching

        Returns:
            Dictionary with metrics
        """
        # Calculate similarities
        positive_similarities = FaceRecognitionMetrics.cosine_similarity(
            anchor_embeddings, positive_embeddings
        )
        negative_similarities = FaceRecognitionMetrics.cosine_similarity(
            anchor_embeddings, negative_embeddings
        )

        # Convert to numpy
        pos_sim = positive_similarities.cpu().numpy()
        neg_sim = negative_similarities.cpu().numpy()

        # Create labels and predictions
        # True positives: anchor-positive pairs (label=1)
        # True negatives: anchor-negative pairs (label=0)
        y_true = np.concatenate([
            np.ones(len(pos_sim)),  # Positive pairs
            np.zeros(len(neg_sim))  # Negative pairs
        ])

        y_pred = np.concatenate([
            (pos_sim >= threshold).astype(int),
            (neg_sim >= threshold).astype(int)
        ])

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # Calculate average similarities
        avg_pos_sim = pos_sim.mean()
        avg_neg_sim = neg_sim.mean()

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'avg_positive_similarity': avg_pos_sim,
            'avg_negative_similarity': avg_neg_sim,
            'similarity_gap': avg_pos_sim - avg_neg_sim
        }