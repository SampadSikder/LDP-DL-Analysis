"""Attention-based attacker detector model."""

import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention to capture different aspects of attack patterns.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
    """
    
    def __init__(self, d_model: int, num_heads: int = 4):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, self.num_heads, self.d_k)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.view(batch_size, self.d_model)
        return self.W_o(attn_output), attn_weights


class AttentionAttackerDetector(nn.Module):
    """
    Attention-based attacker detector with feature embeddings and transformer-style attention.
    
    Architecture:
        - Per-feature embeddings (Linear(1, d_model) for each feature)
        - CLS token for classification
        - Multi-head self-attention
        - Feed-forward network with GELU activation
        - Classification head
    
    Args:
        num_features: Number of input features (input_dim for compatibility)
        d_model: Model dimension (default: 128)
        num_heads: Number of attention heads (default: 4)
        dropout_rate: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        dropout_rate: float = 0.2
    ):
        super(AttentionAttackerDetector, self).__init__()
        
        self.num_features = input_dim
        self.d_model = d_model
        
        # Per-feature embeddings
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_dim)
        ])
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass through the attention model.
        
        Args:
            x: Input tensor of shape (batch_size, num_features)
            return_attention: If True, also return attention weights
        
        Returns:
            Output logits, optionally with attention weights
        """
        batch_size = x.shape[0]
        
        # Embed each feature separately
        tokens = []
        for i, layer in enumerate(self.feature_embeddings):
            col_data = x[:, i:i+1]
            token = layer(col_data)
            tokens.append(token.unsqueeze(1))  # [Batch, 1, d_model]
        
        x_emb = torch.cat(tokens, dim=1)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1)
        
        # Self-attention
        attn_out, attn_weights = self.attention(x_seq, x_seq, x_seq)
        x_seq = self.norm1(x_seq + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x_seq)
        x_seq = self.norm2(x_seq + ffn_out)
        
        # Use CLS token for classification
        cls_final = x_seq[:, 0, :]
        output = self.classifier(cls_final)
        
        if return_attention:
            return output, attn_weights
        
        return output
