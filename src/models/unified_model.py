#!/usr/bin/env python
"""Unified Mood Classification Model with Conv-Transformer architecture"""

import torch
import torch.nn as nn
from .feature_projection import FeatureProjection
from .feature_fusion import FeatureFusion
from .conv_transformer import ConvTransformerEncoder
from .output_heads import OutputHeads


class UnifiedMoodModel(nn.Module):
    """Unified multitask learning framework for music mood classification

    Combines feature projection, fusion, Conv-Transformer encoder, and task-specific heads.

    Args:
        fusion_type (str): Fusion strategy to use ('early', 'late')
        hidden_dim (int): Common latent space dimension (default: 512)
        num_transformer_layers (int): Number of Transformer layers (default: 4)
        num_heads (int): Number of attention heads (default: 8)
        num_class_tags (int): Number of classification tags (default: 18)
        deam_v_range (tuple): Valence output range (min, max), default (1.6, 8.4)
        deam_a_range (tuple): Arousal output range (min, max), default (1.6, 8.2)
    """

    def __init__(self,
                 fusion_type='early',
                 hidden_dim=512,
                 num_transformer_layers=4,
                 num_heads=8,
                 num_class_tags=18,
                 deam_v_range=(1.6, 8.4),
                 deam_a_range=(1.6, 8.2)):
        super().__init__()

        self.fusion_type = fusion_type

        # Feature projection module
        self.feature_proj = FeatureProjection(hidden_dim=hidden_dim)

        # Feature fusion module
        self.feature_fusion = FeatureFusion(fusion_type=fusion_type, hidden_dim=hidden_dim)

        # Determine input dimension for Conv-Transformer based on fusion type
        if fusion_type == 'early':
            conv_transformer_in_dim = hidden_dim  # Early fusion now outputs D dimension instead of 4D
        elif fusion_type == 'late':
            conv_transformer_in_dim = hidden_dim  # All features projected to same dimension
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Conv-Transformer encoder module
        if fusion_type == 'late':
            # For late fusion, create separate encoders for each feature
            self.mel_encoder = ConvTransformerEncoder(in_dim=conv_transformer_in_dim, hidden_dim=hidden_dim, num_layers=num_transformer_layers, num_heads=num_heads)
            self.chroma_encoder = ConvTransformerEncoder(in_dim=conv_transformer_in_dim, hidden_dim=hidden_dim, num_layers=num_transformer_layers, num_heads=num_heads)
            self.tempogram_encoder = ConvTransformerEncoder(in_dim=conv_transformer_in_dim, hidden_dim=hidden_dim, num_layers=num_transformer_layers, num_heads=num_heads)
            self.mert_encoder = ConvTransformerEncoder(in_dim=conv_transformer_in_dim, hidden_dim=hidden_dim, num_layers=num_transformer_layers, num_heads=num_heads)
        else:
            # Single encoder for fused features
            self.encoder = ConvTransformerEncoder(in_dim=conv_transformer_in_dim, hidden_dim=hidden_dim, num_layers=num_transformer_layers, num_heads=num_heads)

        # Task-specific output heads
        self.output_heads = OutputHeads(
            hidden_dim=hidden_dim,
            num_class_tags=num_class_tags,
            deam_v_range=deam_v_range,
            deam_a_range=deam_a_range
        )

    def forward(self, features):
        """Forward pass for the unified mood model

        Args:
            features (dict): Dictionary of input features with keys
                            'mel', 'chroma', 'tempogram', 'mert'

        Returns:
            dict: Model outputs with 'regression' and 'classification' keys
        """
        # Step 1: Feature Projection
        projected = self.feature_proj(features)

        # Step 2: Feature Fusion
        fused = self.feature_fusion(projected)

        # Step 3: Encoding
        if self.fusion_type == 'late':
            # Late fusion: encode each feature separately
            mel_encoded = self.mel_encoder(fused['mel'])
            chroma_encoded = self.chroma_encoder(fused['chroma'])
            tempogram_encoded = self.tempogram_encoder(fused['tempogram'])
            mert_encoded = self.mert_encoder(fused['mert'])

            # Average the encoded features for late fusion
            encoded = (mel_encoded + chroma_encoded + tempogram_encoded + mert_encoded) / 4
        else:
            # Single encoder for fused features
            encoded = self.encoder(fused)

        # Step 4: Task-specific outputs
        outputs = self.output_heads(encoded)

        return outputs


if __name__ == "__main__":
    # Test the model with dummy data
    dummy_features = {
        'mel': torch.randn(4, 100, 128),  # Batch size 4, 100 time steps, 128 mel features
        'chroma': torch.randn(4, 100, 12),  # Batch size 4, 100 time steps, 12 chroma features
        'tempogram': torch.randn(4, 100, 384),  # Batch size 4, 100 time steps, 384 tempogram features
        'mert': torch.randn(4, 100, 2048)  # Batch size 4, 100 time steps, 2048 mert features
    }

    print("Testing model with early fusion...")
    model_early = UnifiedMoodModel(fusion_type='early')
    outputs_early = model_early(dummy_features)
    print(f"Regression output shape: {outputs_early['regression'].shape}")
    print(f"Classification output shape: {outputs_early['classification'].shape}")
    print()

    print("Testing model with late fusion...")
    model_late = UnifiedMoodModel(fusion_type='late')
    outputs_late = model_late(dummy_features)
    print(f"Regression output shape: {outputs_late['regression'].shape}")
    print(f"Classification output shape: {outputs_late['classification'].shape}")
    print()

    print("Model test completed successfully!")
