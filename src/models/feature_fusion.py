#!/usr/bin/env python
"""Feature Fusion module for music mood classification"""

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Feature fusion module to combine low, mid, and high level features

    This module implements two fusion strategies:
    1. Early Fusion (Cross-Attention): Uses cross-attention with MERT as query, others as key/value
    2. Late Fusion: Keeps features separate for independent processing

    Args:
        fusion_type (str): Fusion strategy to use ('early', 'late')
        hidden_dim (int): Common latent space dimension (default: 512)
    """

    def __init__(self, fusion_type='early', hidden_dim=512):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'early':
            # Early Fusion with Cross-Attention: MERT as Query, others as Key/Value
            self.other_features_proj = nn.Linear(hidden_dim * 3, hidden_dim)  # Project 3 features to D
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # For late fusion, no additional layers needed here

    def forward(self, projected_features):
        """Forward pass for feature fusion

        Args:
            projected_features (dict): Dictionary of projected features with keys
                                      'mel', 'chroma', 'tempogram', 'mert'

        Returns:
            torch.Tensor: Fused features (shape depends on fusion type)
        """
        mel = projected_features['mel']  # shape: [B, T, D]
        chroma = projected_features['chroma']  # shape: [B, T, D]
        tempogram = projected_features['tempogram']  # shape: [B, T, D]
        mert = projected_features['mert']  # shape: [B, T, D]

        if self.fusion_type == 'early':
            # Early Fusion with Cross-Attention: MERT as Query, others as Key/Value
            # Concatenate mel, chroma, tempogram along channel dim to form Key/Value
            other_features = torch.cat([mel, chroma, tempogram], dim=-1)  # [B, T, 3D]

            # Project other features to match MERT's dimension
            key = self.other_features_proj(other_features)  # [B, T, D]
            value = self.other_features_proj(other_features)  # [B, T, D]

            # Apply Cross-Attention
            fused, _ = self.cross_attn(query=mert, key=key, value=value)  # [B, T, D]

            return fused


        elif self.fusion_type == 'late':
            # Return all features separately for late fusion
            return {
                'mel': mel,
                'chroma': chroma,
                'tempogram': tempogram,
                'mert': mert
            }

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
