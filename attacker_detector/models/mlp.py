"""MLP-based attacker detector model."""

import torch
import torch.nn as nn


class RobustAttackerDetector(nn.Module):
    """
    Multi-layer perceptron for attacker detection.
    
    Architecture:
        - Layer 1: Linear(input_dim, 256) -> BatchNorm -> LeakyReLU -> Dropout
        - Layer 2: Linear(256, 128) -> BatchNorm -> LeakyReLU -> Dropout
        - Layer 3: Linear(128, 64) -> BatchNorm -> LeakyReLU -> Dropout
        - Output: Linear(64, 1)
    """
    
    def __init__(self, input_dim: int, dropout_rate: float = 0.2):
        super(RobustAttackerDetector, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate / 2)
        )
        
        self.output = nn.Linear(64, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)

