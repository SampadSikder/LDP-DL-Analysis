import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GATAttackerDetector(nn.Module):

    def __init__(
        self,
        input_dim: int = 24,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.2,
    ):
        super(GATAttackerDetector, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # Layer 1: multi-head, concat
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout_rate,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)

        # Layer 2: multi-head, concat
        self.conv2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout_rate,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)

        # Layer 3: single-head, no concat (reduce to hidden_dim)
        self.conv3 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            dropout=dropout_rate,
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.elu = nn.ELU() # Since edges can have negatives, used this instead of ReLU

        # Classification head
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple:
        attention_weights = []

        # Layer 1
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        attention_weights.append(attn1)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Layer 2
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        attention_weights.append(attn2)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout(x)

        # Layer 3
        x, attn3 = self.conv3(x, edge_index, return_attention_weights=True)
        attention_weights.append(attn3)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout(x)

        logits = self.classifier(x)

        return logits, attention_weights
