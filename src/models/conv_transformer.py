#!/usr/bin/env python
"""Conformer encoder module for music mood classification"""

import torch
import torch.nn as nn
from torchaudio.models import Conformer


class ConvTransformerEncoder(nn.Module):
    """Conformer encoder for temporal sequence modeling

    Replaces the hybrid Conv-Transformer with a pure Conformer architecture.

    Args:
        in_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for conformer
        num_layers (int): Number of conformer layers
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate (default: 0.1)
    """
    def __init__(self, in_dim, hidden_dim=512, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()

        # Linear projection to match conformer input dimension
        if in_dim != hidden_dim:
            self.proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.proj = nn.Identity()

        # Conformer encoder
        self.conformer = Conformer(
            input_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=hidden_dim * 4,
            num_layers=num_layers,
            depthwise_conv_kernel_size=31,  # Default kernel size for Conformer
            dropout=dropout
        )

    def forward(self, x):
        """Forward pass for Conformer encoder

        Args:
            x (torch.Tensor): Input features with shape (B, T, D) where B=batch, T=time steps, D=dimension

        Returns:
            torch.Tensor: Encoded features with shape (B, T, D)
        """
        # Project input to conformer hidden dimension
        x_proj = self.proj(x)  # (B, T, D)

        # Apply Conformer
        # Conformer expects input shape (B, T, D) when time_first=False
        # For fixed-length input, pass lengths as tensor of T
        batch_size, seq_len, _ = x_proj.shape
        lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device=x_proj.device)
        x_encoded, _ = self.conformer(x_proj, lengths=lengths)

        return x_encoded
