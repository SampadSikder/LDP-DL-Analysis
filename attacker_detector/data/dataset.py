"""Dataset classes and data loading utilities."""

from typing import Tuple, List, Optional, Dict
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class AttackerDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           StandardScaler, np.ndarray]:
    X = df[feature_cols].values
    y = df['label'].values
    
    # Get indices for later sensitivity analysis
    indices = np.arange(len(df))
    
    X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, scaler, test_idx


_CONF_TARGET_SET_SIZE = 0
_CONF_ATTACKER_RATIO  = 1
_CONF_PROTOCOL        = 2
_CONF_SPLITS          = 3
_CONF_EPSILON         = 4
_CONF_DATASET_TYPE    = 5


class NpyDataset:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        config: np.ndarray,
        feature_names: List[str],
        norm_stats: Optional[dict] = None,
    ):
        self.features = features          # (N, F) float32, already normalized
        self.labels = labels              # (N,) float32
        self.config = config              # (N, 6) object
        self.feature_names = feature_names
        self.norm_stats = norm_stats

    def __len__(self):
        return len(self.labels)

    @property
    def dataset_types(self) -> np.ndarray:
        return self.config[:, _CONF_DATASET_TYPE]

    @property
    def epsilons(self) -> np.ndarray:
        return self.config[:, _CONF_EPSILON].astype(np.float64)

    @property
    def attacker_ratios(self) -> np.ndarray:
        return self.config[:, _CONF_ATTACKER_RATIO].astype(np.float64)

    @property
    def target_set_sizes(self) -> np.ndarray:
        return self.config[:, _CONF_TARGET_SET_SIZE].astype(np.int32)


def load_npy_dataset(data_dir: str) -> NpyDataset:
    features_path = os.path.join(data_dir, 'features.npy')
    labels_path   = os.path.join(data_dir, 'labels.npy')
    config_path   = os.path.join(data_dir, 'config.npy')
    metadata_path = os.path.join(data_dir, 'metadata.npz')
    norm_path     = os.path.join(data_dir, 'norm_stats.json')

    for p in [features_path, labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    features = np.load(features_path)                         # float32 normalized
    labels   = np.load(labels_path).astype(np.float32)
    n_users  = len(labels)

    if os.path.exists(config_path):
        config = np.load(config_path, allow_pickle=True)      # object array (N, 6)
    else:
        print(
            "  [WARN] config.npy not found in dataset directory. "
            "Sensitivity analysis by epsilon/ratio/dataset_type will be unavailable. "
            "Re-run generate_dataset.py to produce a complete dataset with config.npy."
        )
        config = np.empty((n_users, 6), dtype=object)

    norm_stats = None
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm_stats = json.load(f)

    feature_names = (
        norm_stats['feature_names']
        if norm_stats and 'feature_names' in norm_stats
        else [f'feat_{i}' for i in range(features.shape[1])]
    )

    print(  
        f"Loaded NPY dataset from '{data_dir}': "
        f"{n_users:,} samples, {features.shape[1]} features"
    )
    return NpyDataset(features, labels, config, feature_names, norm_stats)



def prepare_npy_data(
    ds: NpyDataset,
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
) -> Dict:
    """Split a NpyDataset into train(/val)/test using stratified random splits.

    Features are already normalized; returns scaler=None.
    If val_size > 0, a validation set is carved from the training portion.
    """
    indices = np.arange(len(ds))
    X_trainval, X_test, y_trainval, y_test, trainval_idx, test_idx = train_test_split(
        ds.features, ds.labels, indices,
        test_size=test_size,
        random_state=random_state,
        stratify=ds.labels.astype(int),
    )

    result = {
        'X_test':       X_test,
        'y_test':       y_test,
        'test_indices': test_idx,
        'scaler':       None,
    }

    if val_size > 0:
        val_frac = val_size / (1.0 - test_size)  # fraction of trainval
        X_train, X_val, y_train, y_val, tr_idx, va_idx = train_test_split(
            X_trainval, y_trainval, trainval_idx,
            test_size=val_frac,
            random_state=random_state,
            stratify=y_trainval.astype(int),
        )
        result['X_train']       = X_train
        result['y_train']       = y_train
        result['train_indices'] = tr_idx
        result['X_val']         = X_val
        result['y_val']         = y_val
        result['val_indices']   = va_idx
        # Also keep the combined trainval for k-fold CV
        result['X_trainval']       = X_trainval
        result['y_trainval']       = y_trainval
        result['trainval_indices'] = trainval_idx
        print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    else:
        result['X_train']       = X_trainval
        result['y_train']       = y_trainval
        result['train_indices'] = trainval_idx
        print(f"Train: {len(X_trainval):,}  Test: {len(X_test):,}")

    return result


def prepare_npy_data_by_dataset_type(
    ds: NpyDataset,
    train_type: str,
    test_type: str,
    eval_type: Optional[str] = None,
    val_size: float = 0.0,
    random_state: int = 42,
) -> Dict:
    """Split by dataset_type column for cross/three-way generalization tests.
    """
    dt = ds.dataset_types

    train_mask = dt == train_type
    test_mask  = dt == test_type

    if not train_mask.any():
        raise ValueError(f"No rows for train dataset_type='{train_type}'")
    if not test_mask.any():
        raise ValueError(f"No rows for test dataset_type='{test_type}'")

    rng = np.random.default_rng(random_state)

    def _shuffle(mask):
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        return idx

    trainval_idx = _shuffle(train_mask)
    test_idx     = _shuffle(test_mask)

    X_test  = ds.features[test_idx]
    y_test  = ds.labels[test_idx]

    result = {
        'X_test':        X_test,
        'y_test':        y_test,
        'test_indices':  test_idx,
        'scaler':        None,
    }

    if val_size > 0:
        X_tv = ds.features[trainval_idx]
        y_tv = ds.labels[trainval_idx]
        X_train, X_val, y_train, y_val, tr_local, va_local = train_test_split(
            X_tv, y_tv, np.arange(len(trainval_idx)),
            test_size=val_size,
            random_state=random_state,
            stratify=y_tv.astype(int),
        )
        tr_idx = trainval_idx[tr_local]
        va_idx = trainval_idx[va_local]
        result['X_train']       = X_train
        result['y_train']       = y_train
        result['train_indices'] = tr_idx
        result['X_val']         = X_val
        result['y_val']         = y_val
        result['val_indices']   = va_idx
        result['X_trainval']       = X_tv
        result['y_trainval']       = y_tv
        result['trainval_indices'] = trainval_idx
        print(f"Train ({train_type}): {len(X_train):,}  Val: {len(X_val):,}  Test ({test_type}): {len(X_test):,}")
    else:
        X_train = ds.features[trainval_idx]
        y_train = ds.labels[trainval_idx]
        result['X_train']       = X_train
        result['y_train']       = y_train
        result['train_indices'] = trainval_idx
        print(f"Train ({train_type}): {len(X_train):,}  Test ({test_type}): {len(X_test):,}")

    if eval_type is not None:
        eval_mask = dt == eval_type
        if not eval_mask.any():
            raise ValueError(f"No rows for eval dataset_type='{eval_type}'")
        eval_idx = _shuffle(eval_mask)
        X_eval = ds.features[eval_idx]
        y_eval = ds.labels[eval_idx]
        print(f"Eval  ({eval_type}): {len(X_eval):,}")
        result['X_eval']       = X_eval
        result['y_eval']       = y_eval
        result['eval_indices'] = eval_idx

    return result


def prepare_data_by_dataset_type(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_type: str,
    test_type: str,
    eval_type: str = None,
    random_state: int = 42
) -> dict:
    # Filter by dataset_type
    train_mask = df['dataset_type'] == train_type
    test_mask = df['dataset_type'] == test_type

    if not train_mask.any():
        raise ValueError(f"No rows found for train dataset_type='{train_type}'")
    if not test_mask.any():
        raise ValueError(f"No rows found for test dataset_type='{test_type}'")
    
    print(f"Found {df[train_mask].shape[0]} rows for train dataset_type='{train_type}'")
    print(f"Found {df[test_mask].shape[0]} rows for test dataset_type='{test_type}'")

    train_df = df[train_mask].sample(frac=1, random_state=random_state).reset_index()
    test_df = df[test_mask].sample(frac=1, random_state=random_state).reset_index()

    X_train_raw = train_df[feature_cols].values
    y_train = train_df['label'].values
    train_indices = train_df['index'].to_numpy()

    X_test_raw = test_df[feature_cols].values
    y_test = test_df['label'].values
    test_indices = test_df['index'].to_numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print(f"Train ({train_type}): {len(X_train)} samples")
    print(f"Test  ({test_type}): {len(X_test)} samples")

    result = {
        'X_train': X_train,
        'y_train': y_train,
        'train_indices': train_indices,
        'X_test': X_test,
        'y_test': y_test,
        'test_indices': test_indices,
        'scaler': scaler,
    }

    if eval_type is not None:
        eval_mask = df['dataset_type'] == eval_type
        if not eval_mask.any():
            raise ValueError(f"No rows found for eval dataset_type='{eval_type}'")

        eval_df = df[eval_mask].sample(frac=1, random_state=random_state).reset_index()
        X_eval_raw = eval_df[feature_cols].values
        y_eval = eval_df['label'].values
        eval_indices = eval_df['index'].to_numpy()

        X_eval = scaler.transform(X_eval_raw)

        print(f"Eval  ({eval_type}): {len(X_eval)} samples")

        result['X_eval'] = X_eval
        result['y_eval'] = y_eval
        result['eval_indices'] = eval_indices

    return result
