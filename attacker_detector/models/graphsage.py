"""GraphSAGE model for LDP attacker detection.

Uses mean-aggregation neighbourhood sampling to learn node
representations. Unlike GAT, GraphSAGE does not produce
attention weights, so the aggregation loss term is zero.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGEAttackerDetector(nn.Module):

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ):
        super(GraphSAGEAttackerDetector, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Layer 1
        self.conv1 = SAGEConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            aggr='mean',
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Layer 2
        self.conv2 = SAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggr='mean',
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Layer 3
        self.conv3 = SAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggr='mean',
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU()

        # Classification head
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple:
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Classify
        logits = self.classifier(x)

        return logits, None
