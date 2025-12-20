"""Models module - Neural network architectures for attacker detection."""

from .mlp import RobustAttackerDetector
from .gan import AttackerDiscriminator
from .attention import AttentionAttackerDetector, MultiHeadAttention


def get_model(model_type: str, input_dim: int, **kwargs):
    """
    Factory function to create models by type.
    
    Args:
        model_type: Type of model ('mlp', 'gan', 'attention')
        input_dim: Number of input features
        **kwargs: Additional model-specific arguments (e.g., dropout_rate)
    
    Returns:
        Instantiated PyTorch model
    
    Raises:
        ValueError: If model_type is not supported
    """
    models = {
        'mlp': RobustAttackerDetector,
        'gan': AttackerDiscriminator,
        'attention': AttentionAttackerDetector,
    }
    
    if model_type not in models:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {list(models.keys())}"
        )
    
    return models[model_type](input_dim, **kwargs)


__all__ = ['RobustAttackerDetector', 'AttackerDiscriminator', 'AttentionAttackerDetector', 'MultiHeadAttention', 'get_model']
