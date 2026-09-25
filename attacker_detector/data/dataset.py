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


_CONF_TARGET_SET_SIZE    = 0
_CONF_ATTACKER_RATIO     = 1
_CONF_PROTOCOL           = 2
_CONF_SPLITS             = 3
_CONF_EPSILON            = 4
_CONF_DATASET_TYPE       = 5
_CONF_N                  = 6
_CONF_EXPERIMENT_ID      = 7
_CONF_ROW_IN_EXPERIMENT  = 8
_CONF_HOLDOUT            = 9   # 1 = held-out LDP run, never trained on (absent in older datasets)


class NpyDataset:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        config: np.ndarray,
        feature_names: List[str],
        norm_stats: Optional[dict] = None,
        design: Optional[dict] = None,
    ):
        self.features = features          # (N, F) float32, already normalized
        self.labels = labels              # (N,) float32
        self.config = config              # (N, 9) or (N, 10) str
        self.feature_names = feature_names
        self.norm_stats = norm_stats
        self.design = design              # contents of design.json, if present

    @property
    def is_oat(self) -> bool:
        return bool(self.design) and self.design.get('design') == 'oat'

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

    @property
    def splits(self) -> np.ndarray:
        return self.config[:, _CONF_SPLITS].astype(np.int32)

    @property
    def protocols(self) -> np.ndarray:
        return self.config[:, _CONF_PROTOCOL]

    @property
    def n_users(self) -> np.ndarray:
        return self.config[:, _CONF_N].astype(np.int64)

    @property
    def experiment_ids(self) -> np.ndarray:
        return self.config[:, _CONF_EXPERIMENT_ID].astype(np.int64)

    @property
    def rows_in_experiment(self) -> np.ndarray:
        return self.config[:, _CONF_ROW_IN_EXPERIMENT].astype(np.int64)

    @property
    def holdout(self) -> np.ndarray:
        """1 for rows from held-out LDP runs; all zeros for older datasets."""
        if self.config.shape[1] <= _CONF_HOLDOUT:
            return np.zeros(len(self), dtype=np.int8)
        return self.config[:, _CONF_HOLDOUT].astype(np.int8)


def load_npy_dataset(data_dir: str) -> NpyDataset:
    features_path = os.path.join(data_dir, 'features.npy')
    labels_path   = os.path.join(data_dir, 'labels.npy')
    config_path   = os.path.join(data_dir, 'config.npy')
    norm_path     = os.path.join(data_dir, 'norm_stats.json')
    design_path   = os.path.join(data_dir, 'design.json')

    for p in [features_path, labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found: {p}")

    features = np.load(features_path)                         # float32 normalized
    labels   = np.load(labels_path).astype(np.float32)
    n_users  = len(labels)

    if os.path.exists(config_path):
        config = np.load(config_path, allow_pickle=True)      # str array (N, 9) or (N, 10)
    else:
        print(
            "  [WARN] config.npy not found in dataset directory. "
            "Sensitivity analysis by epsilon/ratio/dataset_type will be unavailable. "
            "Re-run generate_dataset.py to produce a complete dataset with config.npy."
        )
        config = np.empty((n_users, 10), dtype=object)

    norm_stats = None
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm_stats = json.load(f)

    design = None
    if os.path.exists(design_path):
        with open(design_path) as f:
            design = json.load(f)

    feature_names = (
        norm_stats['feature_names']
        if norm_stats and 'feature_names' in norm_stats
        else [f'feat_{i}' for i in range(features.shape[1])]
    )

    print(
        f"Loaded NPY dataset from '{data_dir}': "
        f"{n_users:,} samples, {features.shape[1]} features"
        + (f", design={design.get('design')}" if design else "")
    )
    return NpyDataset(features, labels, config, feature_names, norm_stats, design)



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


def _config_groups(ds: NpyDataset, idx: np.ndarray) -> np.ndarray:
    """Integer id per row identifying its experiment config: one sweep point on
    one dataset under one protocol. Replicates of the same config share an id."""
    c = ds.config[idx]
    cols = [_CONF_DATASET_TYPE, _CONF_PROTOCOL, _CONF_EPSILON,
            _CONF_ATTACKER_RATIO, _CONF_TARGET_SET_SIZE, _CONF_SPLITS]
    frame = pd.DataFrame({k: c[:, k] for k in cols})
    return frame.groupby(cols, sort=True).ngroup().to_numpy()


def _balanced_per_group(idx, groups, y, per_class, rng):
    """Up to `per_class` positives and as many negatives from every group.
    Each group is kept 50/50
    """
    chosen, short = [], 0
    for g in np.unique(groups):
        in_g = groups == g
        pos = idx[in_g & (y == 1)]
        neg = idx[in_g & (y == 0)]
        k = min(per_class, len(pos), len(neg))
        if k < per_class:
            short += 1
        chosen.append(rng.choice(pos, size=k, replace=False))
        chosen.append(rng.choice(neg, size=k, replace=False))
    return np.concatenate(chosen), short


def prepare_npy_data_oat(
    ds: NpyDataset,
    train_samples: int,
    test_samples: int,
    val_size: float = 0.1,
    random_state: int = 42,
    train_mask: Optional[np.ndarray] = None,
    test_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Split a one-at-a-time dataset by held-out LDP run, balancing training.
    """
    rng = np.random.default_rng(random_state)
    y = ds.labels.astype(int)
    hold = ds.holdout
    n = len(ds)
    train_mask = np.ones(n, bool) if train_mask is None else train_mask
    test_mask = np.ones(n, bool) if test_mask is None else test_mask

    pool = np.where((hold == 0) & train_mask)[0]
    held = np.where((hold == 1) & test_mask)[0]
    if len(pool) == 0:
        raise ValueError("No holdout==0 rows to train on.")
    if len(held) == 0:
        raise ValueError(
            "No holdout==1 rows to test on. Generate with --design oat "
            "--holdout-replicates 1 (or more)."
        )

    # --- balanced train + val -------------------------------------------------
    pool_groups = _config_groups(ds, pool)
    n_groups = len(np.unique(pool_groups))
    per_class = max(1, train_samples // (2 * n_groups))
    tv_idx, short = _balanced_per_group(pool, pool_groups, y[pool], per_class, rng)
    if short:
        print(f"  [WARN] {short}/{n_groups} configs had fewer than {per_class:,} rows of "
              f"one class; those configs were cut to their smaller class to stay 50/50. "
              f"Train+val is {len(tv_idx):,} rows instead of ~{train_samples:,}.")

    tv_groups = _config_groups(ds, tv_idx)
    tr_idx, va_idx = train_test_split(
        tv_idx, test_size=val_size, random_state=random_state,
        stratify=tv_groups * 2 + y[tv_idx],
    )

    held_groups = _config_groups(ds, held)
    held_strata = held_groups * 2 + y[held]
    if test_samples >= len(held):
        test_idx = held
    else:
        test_idx, _ = train_test_split(
            held, train_size=test_samples, random_state=random_state,
            stratify=held_strata,
        )

    per_class_held = int(np.bincount(held_strata).max())  # keep every attacker
    sens_bal, _ = _balanced_per_group(held, held_groups, y[held], per_class_held, rng)

    def _take(ix):
        return ds.features[ix], ds.labels[ix]

    X_train, y_train = _take(tr_idx)
    X_val, y_val = _take(va_idx)
    X_test, y_test = _take(test_idx)
    X_tv, y_tv = _take(tv_idx)

    print(f"Train: {len(tr_idx):,}  Val: {len(va_idx):,}  "
          f"({per_class:,}/class x {n_groups} configs, 50/50)  "
          f"Test: {len(test_idx):,} held-out rows at natural prevalence "
          f"({100 * y_test.mean():.2f}% attackers)")
    print(f"Sensitivity: {len(held):,} held-out rows "
          f"(balanced view: {len(sens_bal):,})")

    return {
        'X_train': X_train, 'y_train': y_train, 'train_indices': tr_idx,
        'X_val': X_val, 'y_val': y_val, 'val_indices': va_idx,
        'X_trainval': X_tv, 'y_trainval': y_tv, 'trainval_indices': tv_idx,
        'X_test': X_test, 'y_test': y_test, 'test_indices': test_idx,
        'sens_indices': held,
        'sens_balanced_indices': np.sort(sens_bal),
        'scaler': None,
    }


def prepare_npy_data_by_dataset_type(
    ds: NpyDataset,
    train_type,
    test_type: str,
    eval_type: Optional[str] = None,
    val_size: float = 0.0,
    random_state: int = 42,
) -> Dict:
    """Split by dataset_type column for cross/three-way generalization tests.

    Args:
        train_type: A single dataset_type string, or a list of dataset_type
            strings to combine into one training set (e.g. ['zipf', 'emoji']).
        test_type: Single dataset_type used for testing.
        eval_type: Optional single dataset_type used for three-way eval.
    """
    dt = ds.dataset_types

    train_types = [train_type] if isinstance(train_type, str) else list(train_type)
    train_label = '+'.join(train_types)

    train_mask = np.isin(dt, train_types)
    test_mask  = dt == test_type

    if not train_mask.any():
        raise ValueError(f"No rows for train dataset_type(s)={train_types}")
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
        print(f"Train ({train_label}): {len(X_train):,}  Val: {len(X_val):,}  Test ({test_type}): {len(X_test):,}")
    else:
        X_train = ds.features[trainval_idx]
        y_train = ds.labels[trainval_idx]
        result['X_train']       = X_train
        result['y_train']       = y_train
        result['train_indices'] = trainval_idx
        print(f"Train ({train_label}): {len(X_train):,}  Test ({test_type}): {len(X_test):,}")

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
