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
        deam_v_range (tuple): Valence输出范围 (min, max)，默认 (1.6, 8.4)
        deam_a_range (tuple): Arousal输出范围 (min, max)，默认 (1.6, 8.2)
    """

    def __init__(self, hidden_dim=512, num_class_tags=18,
                 deam_v_range=(1.6, 8.4), deam_a_range=(1.6, 8.2)):
        super().__init__()

        self.deam_v_min, self.deam_v_max = deam_v_range
        self.deam_a_min, self.deam_a_max = deam_a_range

        # Regression head for DEAM VA prediction
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)  # Output: Valence and Arousal (raw logits)
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
        regression_raw = self.regression_head(x_pooled)  # shape: [B, 2]

        # 使用sigmoid约束输出到指定范围
        v_scaled = self.deam_v_min + torch.sigmoid(regression_raw[:, 0]) * \
                   (self.deam_v_max - self.deam_v_min)
        a_scaled = self.deam_a_min + torch.sigmoid(regression_raw[:, 1]) * \
                   (self.deam_a_max - self.deam_a_min)
        regression_output = torch.stack([v_scaled, a_scaled], dim=1)

        # Classification head (MTG-Jamendo tag prediction)
        classification_output = self.classification_head(x_pooled)  # shape: [B, num_class_tags]

        return {
            'regression': regression_output,
            'classification': classification_output
        }
