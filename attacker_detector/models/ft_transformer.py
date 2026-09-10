"""FT-Transformer (Feature Tokenizer + Transformer) for tabular attacker detection.

Reference: Gorishniy et al., "Revisiting Deep Learning Models for Tabular
Data" (NeurIPS 2021). Each continuous feature is projected into its own
d_token embedding, then pre-norm transformer blocks let self-attention learn
per-example which features matter, instead of the single fixed weight matrix
a plain MLP applies to every input alike.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

# Hyperparameter names understood by FTTransformer's constructor (beyond the
# generic `dropout_rate` every model accepts) -- used by the CV/HP-search
# code to know which grid keys to forward as model kwargs.
FT_TRANSFORMER_HP_KEYS = (
    'd_token', 'n_heads', 'n_layers', 'ffn_d_multiplier',
    'attention_dropout', 'residual_dropout',
)


class FeatureTokenizer(nn.Module):
    """Embeds each scalar (continuous) feature into its own d_token vector.

    token_i = x_i * weight_i + bias_i, computed for all features at once via
    a single (num_features, d_token) weight/bias pair rather than a Linear
    per feature.
    """

    def __init__(self, num_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias = nn.Parameter(torch.empty(num_features, d_token))
        d_sqrt_inv = 1 / math.sqrt(d_token)
        nn.init.uniform_(self.weight, -d_sqrt_inv, d_sqrt_inv)
        nn.init.uniform_(self.bias, -d_sqrt_inv, d_sqrt_inv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_features) -> (batch, num_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class ReGLU(nn.Module):
    """Gated activation: splits the hidden projection in half and gates one
    half through GELU. Expects an even-sized last dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return a * nn.functional.gelu(b)


class FeedForward(nn.Module):
    """ReGLU feed-forward block, sized so its param count roughly matches a
    plain 4x GELU-MLP when d_multiplier ~= 4/3 (the paper's convention)."""

    def __init__(self, d_token: int, d_multiplier: float, dropout: float):
        super().__init__()
        inner_dim = max(1, int(d_token * d_multiplier))
        self.linear_in = nn.Linear(d_token, inner_dim * 2)
        self.activation = ReGLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(inner_dim, d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.linear_out(x)


class FTTransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> MHSA -> residual, LN -> ReGLU FFN -> residual.

    Pre-norm (vs. the post-norm used in attacker_detector.models.attention)
    is what lets several of these stack without training instability.
    """

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        ffn_d_multiplier: float,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_token)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(residual_dropout)

        self.norm_ffn = nn.LayerNorm(d_token)
        self.ffn = FeedForward(d_token, ffn_d_multiplier, attention_dropout)
        self.dropout_ffn = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attn(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout_attn(attn_out)
        x = x + self.dropout_ffn(self.ffn(self.norm_ffn(x)))
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular attacker detection.

    Args:
        input_dim: Number of input features.
        dropout_rate: Fallback used for attention_dropout / residual_dropout
            when they aren't given explicitly -- keeps this model
            constructible via the same `dropout_rate` kwarg the generic
            training code (Trainer, run_k_fold_cv) passes to every model.
        d_token: Per-feature embedding width. Must be divisible by n_heads.
        n_heads: Number of self-attention heads.
        n_layers: Number of transformer blocks.
        ffn_d_multiplier: ReGLU hidden-width multiplier (paper default ~4/3).
        attention_dropout: Dropout on attention weights and inside the FFN.
        residual_dropout: Dropout applied to each sublayer's output before
            it is added back to the residual stream.
    """

    def __init__(
        self,
        input_dim: int,
        dropout_rate: float = 0.2,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_d_multiplier: float = 4 / 3,
        attention_dropout: Optional[float] = None,
        residual_dropout: Optional[float] = None,
    ):
        super().__init__()

        if d_token % n_heads != 0:
            raise ValueError(
                f"d_token ({d_token}) must be divisible by n_heads ({n_heads})"
            )

        attention_dropout = dropout_rate if attention_dropout is None else attention_dropout
        residual_dropout = dropout_rate if residual_dropout is None else residual_dropout

        self.tokenizer = FeatureTokenizer(input_dim, d_token)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        d_sqrt_inv = 1 / math.sqrt(d_token)
        nn.init.uniform_(self.cls_token, -d_sqrt_inv, d_sqrt_inv)

        self.blocks = nn.ModuleList([
            FTTransformerBlock(
                d_token=d_token,
                n_heads=n_heads,
                ffn_d_multiplier=ffn_d_multiplier,
                attention_dropout=attention_dropout,
                residual_dropout=residual_dropout,
            )
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, tokens], dim=1)

        for block in self.blocks:
            x = block(x)

        return self.head(x[:, 0, :])
