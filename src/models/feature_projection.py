#!/usr/bin/env python
"""Feature Projection module for music mood classification"""

import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    """Feature projection module to align different feature dimensions

    This module projects each type of feature (mel, chroma, tempogram, mert)
    into a common latent space using appropriate layers (CNN for mel,
    linear layers for others).

    Args:
        hidden_dim (int): Common latent space dimension (default: 512)
        mel_in_dim (int): Mel-spectrogram input dimension (default: 128)
        chroma_in_dim (int): Chroma input dimension (default: 12)
        tempogram_in_dim (int): Tempogram input dimension (default: 384)
        mert_in_dim (int): MERT embedding dimension (default: 1024)
    """

    def __init__(self,
                 hidden_dim=512,
                 mel_in_dim=128,
                 chroma_in_dim=12,
                 tempogram_in_dim=384,
                 mert_in_dim=1024):  # MERT layer 11 outputs 1024 features
        super().__init__()

        # CNN for mel-spectrogram (local feature extraction)
        self.mel_proj = nn.Sequential(
            nn.Conv1d(in_channels=mel_in_dim,
                      out_channels=hidden_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim,
                      out_channels=hidden_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Linear projection for chroma (mid-level feature)
        self.chroma_proj = nn.Sequential(
            nn.Linear(chroma_in_dim, hidden_dim),
            nn.ReLU()
        )

        # Linear projection for tempogram (mid-level feature)
        self.tempogram_proj = nn.Sequential(
            nn.Linear(tempogram_in_dim, hidden_dim),
            nn.ReLU()
        )

        # Linear projection for MERT (high-level feature)
        self.mert_proj = nn.Sequential(
            nn.Linear(mert_in_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, features):
        """Forward pass for feature projection

        Args:
            features (dict): Dictionary containing features with keys
                            'mel', 'chroma', 'tempogram', 'mert'

        Returns:
            dict: Projected features in common latent space
        """
        mel = features['mel']  # shape: [B, T, mel_in_dim]
        chroma = features['chroma']  # shape: [B, T, chroma_in_dim]
        tempogram = features['tempogram']  # shape: [B, T, tempogram_in_dim]
        mert = features['mert']  # shape: [B, T, mert_in_dim]

        # Mel: need to permute for CNN (batch, channel, time)
        mel_proj = self.mel_proj(mel.permute(0, 2, 1)).permute(0, 2, 1)  # shape: [B, T, hidden_dim]

        # Chroma: linear projection
        chroma_proj = self.chroma_proj(chroma)  # shape: [B, T, hidden_dim]

        # Tempogram: linear projection
        tempogram_proj = self.tempogram_proj(tempogram)  # shape: [B, T, hidden_dim]

        # MERT: linear projection
        mert_proj = self.mert_proj(mert)  # shape: [B, T, hidden_dim]

        return {
            'mel': mel_proj,
            'chroma': chroma_proj,
            'tempogram': tempogram_proj,
            'mert': mert_proj
        }
