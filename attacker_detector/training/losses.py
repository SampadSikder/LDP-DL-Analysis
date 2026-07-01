"""Composite loss function for GNN attacker detection.

Implements: L = L_classification + λ * L_aggregation
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict


class CompositeLoss(nn.Module):
    """
    Args:
        lambda_agg: Weight for the aggregation loss term.
        pos_weight: Positive class weight for BCEWithLogitsLoss (handles class imbalance).
    """

    def __init__(
        self,
        lambda_agg: float = 0.1,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(CompositeLoss, self).__init__()
        self.lambda_agg = lambda_agg
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _compute_attention_entropy(
        self,
        attention_weights: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        layer_entropies = []

        for edge_index, alpha in attention_weights:
            if alpha.dim() == 1:
                alpha = alpha.unsqueeze(-1)

            num_heads = alpha.shape[1]
            alpha_clamped = alpha.clamp(min=1e-8)
            edge_entropy = -alpha_clamped * torch.log(alpha_clamped)

            mean_entropy = edge_entropy.mean()
            layer_entropies.append(mean_entropy)

        if len(layer_entropies) == 0:
            return torch.tensor(0.0, requires_grad=True)

        return torch.stack(layer_entropies).mean()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_weights: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_cls = self.classification_loss(logits, labels)

        if attention_weights is not None and self.lambda_agg > 0:
            l_agg = self._compute_attention_entropy(attention_weights)
        else:
            l_agg = torch.tensor(0.0, device=logits.device)

        total_loss = l_cls + self.lambda_agg * l_agg

        loss_dict = {
            'cls': l_cls.item(),
            'agg': l_agg.item(),
            'total': total_loss.item(),
        }

        return total_loss, loss_dict
