"""Composite loss function for GNN attacker detection.

Implements: L = L_classification + λ_agg * L_aggregation + λ_utility * L_utility
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict
from .utility_loss import UtilityLoss


class CompositeLoss(nn.Module):
    """
    Args:
        lambda_agg: Weight for the aggregation loss term (attention entropy).
        lambda_utility: Weight for the utility-aware loss term.
        utility_metric: Metric for utility loss ('js' or 'wasserstein').
        pos_weight: Positive class weight for BCEWithLogitsLoss (handles class imbalance).
    """

    def __init__(
        self,
        lambda_agg: float = 0.1,
        lambda_utility: float = 0.0,
        utility_metric: str = 'js',
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super(CompositeLoss, self).__init__()
        self.lambda_agg = lambda_agg
        self.lambda_utility = lambda_utility
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.utility_loss = UtilityLoss(metric=utility_metric)

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
        batch_idx: Optional[torch.Tensor] = None,
        utility_all: Optional[torch.Tensor] = None,
        utility_benign: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_cls = self.classification_loss(logits, labels)

        if attention_weights is not None and self.lambda_agg > 0:
            l_agg = self._compute_attention_entropy(attention_weights)
        else:
            l_agg = torch.tensor(0.0, device=logits.device)

        if utility_all is not None and utility_benign is not None and self.lambda_utility > 0:
            l_util = self.utility_loss(
                logits=logits,
                labels=labels,
                batch_idx=batch_idx,
                utility_all=utility_all,
                utility_benign=utility_benign,
            )
        else:
            l_util = torch.tensor(0.0, device=logits.device)

        total_loss = l_cls + self.lambda_agg * l_agg + self.lambda_utility * l_util

        loss_dict = {
            'cls': l_cls.item(),
            'agg': l_agg.item(),
            'utility': l_util.item(),
            'total': total_loss.item(),
        }

        return total_loss, loss_dict
