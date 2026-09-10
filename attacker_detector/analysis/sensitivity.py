"""Sensitivity analysis and visualization utilities."""

from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import PARAM_DISPLAY_MAP
from attacker_detector.data.dataset import (
    _CONF_TARGET_SET_SIZE,
    _CONF_ATTACKER_RATIO,
    _CONF_PROTOCOL,
    _CONF_SPLITS,
    _CONF_EPSILON,
    _CONF_DATASET_TYPE,
    _CONF_N,
)


def run_sensitivity_analysis(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    config_array: Optional[np.ndarray] = None,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """
    Evaluate model performance across parameter values.

    Args:
        model:        Trained model.
        X_test:       (N, F) float array of already-normalized features.
        y_test:       (N,) binary labels.
        device:       Torch device.
        config_array: (N, 9) object array with columns
                      [target_set_size, attacker_ratio, protocol, splits,
                       epsilon, dataset_type, n, experiment_id, row_in_experiment].
                      If None, per-parameter breakdown is skipped.
        batch_size:   Mini-batch size for inference.
    """
    model.eval()
    results = []

    print(f"Predicting on {len(X_test):,} samples (batch size: {batch_size})...")

    all_probs = []
    for i in range(0, len(X_test), batch_size):
        batch = torch.FloatTensor(X_test[i:i + batch_size]).to(device)
        with torch.no_grad():
            out = torch.sigmoid(model(batch))
            all_probs.append(out.cpu().numpy().flatten())
        del batch

    global_probs = np.concatenate(all_probs)
    global_preds = (global_probs > 0.5).astype(int)
    y_true_all   = y_test.astype(int)

    # Overall metrics
    results.append({
        'Parameter_Type':  'overall',
        'Parameter_Label': 'Overall',
        'Value':           'all',
        'Accuracy':  accuracy_score(y_true_all, global_preds),
        'Precision': precision_score(y_true_all, global_preds, zero_division=0),
        'Recall':    recall_score(y_true_all, global_preds, zero_division=0),
        'F1_Score':  f1_score(y_true_all, global_preds, zero_division=0),
        'Count':     len(y_true_all),
    })

    if config_array is None:
        return pd.DataFrame(results)

    print("Calculating sensitivity metrics...")

    # Named column indices (attacker_detector/data/dataset.py's _CONF_* constants)
    # so this stays correct if config.npy's column layout changes.
    _COL_MAP = {
        'target_set_size': (_CONF_TARGET_SET_SIZE, PARAM_DISPLAY_MAP.get('target_set_size', 'Target Set Size')),
        'attacker_ratio':  (_CONF_ATTACKER_RATIO,  PARAM_DISPLAY_MAP.get('attacker_ratio',  'Attacker Ratio')),
        'protocol':        (_CONF_PROTOCOL,        PARAM_DISPLAY_MAP.get('protocol',        'Protocol')),
        'splits':          (_CONF_SPLITS,          PARAM_DISPLAY_MAP.get('splits',          'Splits')),
        'epsilon':         (_CONF_EPSILON,         PARAM_DISPLAY_MAP.get('epsilon',         'Epsilon')),
        'dataset_type':    (_CONF_DATASET_TYPE,    PARAM_DISPLAY_MAP.get('dataset_type',    'Dataset Type')),
        'n':               (_CONF_N,               PARAM_DISPLAY_MAP.get('n',               'Number of Users')),
    }

    for col_name, (col_idx, display_name) in _COL_MAP.items():
        col_vals = config_array[:, col_idx]
        unique_vals = sorted(set(col_vals))

        for val in unique_vals:
            mask = col_vals == val
            if not mask.any():
                continue

            y_t  = y_true_all[mask]
            y_p  = global_preds[mask]

            results.append({
                'Parameter_Type':  col_name,
                'Parameter_Label': display_name,
                'Value':           val,
                'Accuracy':  accuracy_score(y_t, y_p),
                'Precision': precision_score(y_t, y_p, zero_division=0),
                'Recall':    recall_score(y_t, y_p, zero_division=0),
                'F1_Score':  f1_score(y_t, y_p, zero_division=0),
                'Count':     int(mask.sum()),
            })

    return pd.DataFrame(results)


def plot_sensitivity_metric(
    sensitivity_df: pd.DataFrame,
    metric: str = 'F1_Score',
    save_path: str = None,
) -> None:
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    params = ['epsilon', 'attacker_ratio', 'target_set_size', 'splits', 'n']
    labels = ['$\\epsilon$', '$\\beta$', '$r$', 'splits', '$n$']

    fig, axes = plt.subplots(1, len(params), figsize=(6 * len(params), 5))
    line_color = '#1f77b4'

    for i, param in enumerate(params):
        ax = axes[i]
        data = sensitivity_df[sensitivity_df['Parameter_Type'] == param].copy()

        if len(data) == 0:
            ax.set_title(f"No data for {param}")
            continue

        # Ensure Value is numeric (may be object dtype when mixed with string params)
        data['Value'] = pd.to_numeric(data['Value'])

        ax.plot(
            data['Value'], data[metric],
            marker='X', markersize=8,
            linestyle='--', linewidth=2,
            color=line_color, label='Proposed DL Model'
        )

        ax.set_xlabel(labels[i], fontsize=14, fontweight='bold')
        if i == 0:
            ax.set_ylabel(metric.replace('_', ' '), fontsize=14, fontweight='bold')

        ax.set_ylim(-0.05, 1.05)

        if param in ('target_set_size', 'splits'):
            ax.set_xticks(data['Value'].unique())

        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Impact of Parameters on {metric.replace('_', ' ')}",
        fontsize=16, y=1.05
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.savefig(f'sensitivity_{metric.lower()}.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to: sensitivity_{metric.lower()}.png")

    plt.close(fig)
