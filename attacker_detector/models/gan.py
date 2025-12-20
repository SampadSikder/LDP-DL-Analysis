"""GAN-style discriminator for attacker detection."""

import torch
import torch.nn as nn


class AttackerDiscriminator(nn.Module):
    """
    GAN-style discriminator that learns the boundary between
    genuine benign behavior and attacker imitations.
    
    Architecture:
        - Feature Extractor: Linear(input_dim, 128) -> LayerNorm -> LeakyReLU -> Dropout
                            -> Linear(128, 64) -> LayerNorm -> LeakyReLU -> Dropout
        - Discriminator: Linear(64, 32) -> LeakyReLU -> Dropout -> Linear(32, 1)
    
    Args:
        input_dim: Number of input features
        dropout_rate: Dropout probability (default: 0.2)
    """
    
    def __init__(self, input_dim: int, dropout_rate: float = 0.2):
        super(AttackerDiscriminator, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate)
        )
        
        self.discriminator = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.feature_extractor(x)
        return self.discriminator(features)
