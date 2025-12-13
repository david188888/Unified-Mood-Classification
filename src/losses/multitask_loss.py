#!/usr/bin/env python
"""Multitask loss functions for music mood classification"""

import torch
import torch.nn as nn


def concordance_correlation_coefficient(preds, targets):
    """
    Computes Concordance Correlation Coefficient (CCC) for two tensors.
    CCC measures the agreement between two variables.

    Args:
        preds (torch.Tensor): Predictions tensor of shape [B, D]
        targets (torch.Tensor): Targets tensor of shape [B, D]

    Returns:
        torch.Tensor: CCC value
    """
    # On Apple MPS (MPSGraph), some elementwise ops (e.g., subtract) require
    # identical dtypes for all operands/results. Under autocast, model outputs
    # may become fp16 while labels remain fp32, which can crash.
    preds = preds.float()
    targets = targets.float()

    B, D = preds.shape

    # Mean of predictions and targets
    pred_mean = preds.mean(dim=0, keepdim=True)
    target_mean = targets.mean(dim=0, keepdim=True)

    # Variance
    pred_var = ((preds - pred_mean) ** 2).mean(dim=0)
    target_var = ((targets - target_mean) ** 2).mean(dim=0)

    # Covariance
    covariance = ((preds - pred_mean) * (targets - target_mean)).mean(dim=0)

    # CCC formula
    ccc = (2.0 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)

    # Average over dimensions
    return ccc.mean()


class MultitaskLoss(nn.Module):
    """Multitask loss function with Uncertainty Weighting for music mood classification"""

    def __init__(self):
        super().__init__()
        # Task-specific losses
        self.deam_loss = nn.MSELoss()  # DEAM: Regression task
        self.mtg_loss = nn.BCEWithLogitsLoss()  # MTG-Jamendo: Multi-label classification

        # Learnable uncertainty parameters (log σ) for numerical stability
        # Using log σ guarantees σ = exp(log σ) > 0.
        self.deam_log_sigma = nn.Parameter(torch.zeros(()))
        self.mtg_log_sigma = nn.Parameter(torch.zeros(()))

    def forward(self, model_outputs, true_labels, task_type):
        """
        Forward pass for multitask loss calculation.

        Args:
            model_outputs (dict): Model outputs with 'regression' and 'classification' keys
            true_labels (torch.Tensor): Ground truth labels
            task_type (str): Task type ('deam' or 'mtg')

        Returns:
            tuple: (loss, metric) where metric is CCC for DEAM and None for MTG
        """
        if task_type == 'deam':
            # DEAM task: Valence/Arousal regression
            regression_output = model_outputs['regression']
            # Compute loss/metrics in fp32 to avoid MPS dtype mismatch under AMP.
            regression_output_fp32 = regression_output.float()
            true_labels_fp32 = true_labels.float()
            loss = self.deam_loss(regression_output_fp32, true_labels_fp32)

            # Uncertainty weighting for regression task
            weighted_loss = 0.5 * torch.exp(-2.0 * self.deam_log_sigma) * loss + self.deam_log_sigma

            metric = concordance_correlation_coefficient(regression_output_fp32, true_labels_fp32)
            return weighted_loss, metric
        elif task_type == 'mtg':
            # MTG-Jamendo task: Mood tag classification
            classification_output = model_outputs['classification']
            classification_output_fp32 = classification_output.float()
            true_labels_fp32 = true_labels.float()
            loss = self.mtg_loss(classification_output_fp32, true_labels_fp32)

            # Uncertainty weighting for classification task
            weighted_loss = 0.5 * torch.exp(-2.0 * self.mtg_log_sigma) * loss + self.mtg_log_sigma

            metric = None  # CCC not applicable for classification
            return weighted_loss, metric
        else:
            raise ValueError(f"Unknown task type: {task_type}")
