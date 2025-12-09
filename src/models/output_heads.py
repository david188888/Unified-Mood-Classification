#!/usr/bin/env python
"""Task-specific output heads module"""

import torch
import torch.nn as nn


class OutputHeads(nn.Module):
    """Task-specific output heads for multi-task learning

    Implements two output heads:
    1. Regression Head: For DEAM VA (Valence/Arousal) prediction
    2. Classification Head: For MTG-Jamendo tag classification

    Args:
        hidden_dim (int): Input hidden dimension (default: 512)
        num_class_tags (int): Number of classification tags (default: 18)
    """

    def __init__(self, hidden_dim=512, num_class_tags=18):
        super().__init__()

        # Regression head for DEAM VA prediction
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)  # Output: Valence and Arousal
        )

        # Classification head for MTG-Jamendo tag classification
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_class_tags)  # Output: Tag probabilities
        )

    def forward(self, x):
        """Forward pass for output heads

        Args:
            x (torch.Tensor): Input features with shape [B, T, D]

        Returns:
            dict: Dictionary containing regression and classification outputs
        """
        # Global Average Pooling across time dimension for static predictions
        x_pooled = x.mean(dim=1)  # shape: [B, D]

        # Regression head (DEAM VA prediction)
        regression_output = self.regression_head(x_pooled)  # shape: [B, 2]

        # Classification head (MTG-Jamendo tag prediction)
        classification_output = self.classification_head(x_pooled)  # shape: [B, num_class_tags]

        return {
            'regression': regression_output,
            'classification': classification_output
        }
