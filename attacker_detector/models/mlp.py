"""MLP-based attacker detector model."""

import math
from typing import List, Optional

import torch
import torch.nn as nn


def pyramid_hidden_sizes(
    input_dim: int,
    n_layers: int = 3,
    width_mult: int = 4,
) -> List[int]:
    seed = max(8, 2 ** round(math.log2(max(input_dim, 1) * width_mult)))
    return [max(4, seed >> i) for i in range(n_layers)]


class RobustAttackerDetector(nn.Module):
    """
    Multi-layer perceptron for attacker detection.
    """

    def __init__(
        self,
        input_dim: int,
        dropout_rate: float = 0.2,
        hidden_sizes: Optional[List[int]] = None,
    ):
        super(RobustAttackerDetector, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = pyramid_hidden_sizes(input_dim)
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer width")

        self.hidden_sizes = list(hidden_sizes)

        blocks = []
        prev = input_dim
        for i, width in enumerate(self.hidden_sizes):
            is_last = i == len(self.hidden_sizes) - 1
            blocks.append(nn.Sequential(
                nn.Linear(prev, width),
                nn.BatchNorm1d(width),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate / 2 if is_last else dropout_rate),
            ))
            prev = width

        self.hidden = nn.Sequential(*blocks)
        self.output = nn.Linear(prev, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # He init matched to the LeakyReLU(0.1) negative slope used above.
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(
                module.weight, a=0.1, nonlinearity='leaky_relu'
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.output(self.hidden(x))
