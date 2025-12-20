"""Configuration constants for attacker detection."""

# High importance features from feature importance analysis
HIGH_IMPORTANCE_FEATURES = [
    'mean_item_freq_ratio',
    'num_ones',
    'overlap_anomalous_items_count',
    'wasserstein_distance_k',
    'js_divergence_k',
    'max_item_freq_ratio',
    'std_item_freq_ratio'
]

# Training features - Statistical features only
TRAINING_FEATURES = HIGH_IMPORTANCE_FEATURES + [
    'k_theoretical_frequency',
    'k_observed_frequency',
    'log_likelihood'
]

# Context features for sensitivity analysis
CONTEXT_FEATURES = ['target_set_size', 'attacker_ratio', 'epsilon']

# Parameter display names for plots
PARAM_DISPLAY_MAP = {
    'epsilon': 'Epsilon ($\\epsilon$)',
    'attacker_ratio': 'Attacker Ratio ($\\beta$)',
    'target_set_size': 'Target Set Size ($r$)'
}

# Default training hyperparameters
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT = 0.2
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42
