"""Utility-aware loss for GNN attacker detection.
"""

import torch
import torch.nn as nn
from typing import Optional


class UtilityLoss(nn.Module):
    """
    Args:
        metric: 'js' (Jensen-Shannon divergence) or 'wasserstein' (Wasserstein distance).
    """

    def __init__(self, metric: str = 'js'):
        super(UtilityLoss, self).__init__()
        if metric not in ('js', 'wasserstein'):
            raise ValueError(f"Unknown utility metric: '{metric}'. Must be 'js' or 'wasserstein'.")
        self.metric = metric

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        batch_idx: Optional[torch.Tensor],
        utility_all: torch.Tensor,
        utility_benign: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits, shape [N, 1] or [N].
            labels: Binary ground truth labels (0=benign, 1=attacker), shape [N, 1] or [N].
            batch_idx: Graph assignment vector for nodes in batch, shape [N]. If None, assumes single graph.
            utility_all: Pre-computed baseline utility with attackers present, shape [num_graphs].
            utility_benign: Pre-computed target optimal utility with benign only, shape [num_graphs].
        """
        probs = torch.sigmoid(logits).squeeze(-1)
        labels = labels.squeeze(-1)

        if batch_idx is None:
            batch_idx = torch.zeros_like(labels, dtype=torch.long)

        num_graphs = utility_all.size(0) if utility_all.dim() > 0 else 1

        if utility_all.dim() == 0:
            utility_all = utility_all.unsqueeze(0)
            utility_benign = utility_benign.unsqueeze(0)

        graph_losses = []

        for g in range(num_graphs):
            mask_g = (batch_idx == g)
            if not mask_g.any():
                continue

            labels_g = labels[mask_g]
            probs_g = probs[mask_g]

            att_mask = (labels_g == 1.0)
            ben_mask = (labels_g == 0.0)

            num_att = att_mask.sum().float()
            num_ben = ben_mask.sum().float()

            if num_att > 0:
                soft_recall = probs_g[att_mask].sum() / num_att
            else:
                soft_recall = torch.tensor(1.0, device=logits.device)

            if num_ben > 0:
                soft_fp_rate = probs_g[ben_mask].sum() / num_ben
            else:
                soft_fp_rate = torch.tensor(0.0, device=logits.device)

            quality = torch.clamp(soft_recall - soft_fp_rate, min=0.0, max=1.0)

            u_all_g = utility_all[g]
            u_ben_g = utility_benign[g]

            predicted_utility = u_all_g + quality * (u_ben_g - u_all_g)

            loss_g = (predicted_utility - u_ben_g) ** 2
            graph_losses.append(loss_g)

        if len(graph_losses) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return torch.stack(graph_losses).mean()
