import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from .losses import CompositeLoss


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # Optimise conv for fixed input sizes
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using GPU: {gpu_name}")
    else:
        device = torch.device('cpu')
        print("[Device] CUDA not available — using CPU")
    return device


def init_weights(model: nn.Module, method: str = 'xavier_uniform') -> None:
    """
    Weight initialization using different method
    Args:
        model: PyTorch model.
        method: Initialization method.
            - 'xavier_uniform': Xavier/Glorot uniform
            - 'kaiming': Kaiming/He (for ReLU-family)
            - 'orthogonal': Orthogonal initialization
            - 'default': Keep PyTorch defaults
    """
    if method == 'default':
        return

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue 

        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(param)
        elif method == 'kaiming':
            nn.init.kaiming_uniform_(param, nonlinearity='leaky_relu')
        elif method == 'orthogonal':
            nn.init.orthogonal_(param)
        else:
            raise ValueError(f"Unknown init method: '{method}'")


class GNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = 0.001,
        lambda_agg: float = 0.1,
        lambda_utility: float = 0.0,
        utility_metric: str = 'js',
        patience: int = 10,
        model_type: str = 'gat',
        pos_weight: Optional[float] = None,
        threshold: float = 0.5,
    ):
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.lambda_agg = lambda_agg
        self.lambda_utility = lambda_utility
        self.utility_metric = utility_metric
        self.patience = patience
        self.model_type = model_type
        self._pos_weight_override = pos_weight  
        self.threshold = threshold

        self._use_amp = self.device.type == 'cuda'
        self._scaler = torch.amp.GradScaler(enabled=self._use_amp)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        self.criterion: Optional[CompositeLoss] = None
        self.history: List[Dict] = []

        self._best_model_state = None
        self._best_val_f1 = -1.0
        self._best_epoch = -1

    def _compute_pos_weight(self, loader: PyGDataLoader) -> torch.Tensor:
        """Compute positive class weight from the training data."""
        total_pos = 0
        total_neg = 0
        for batch in loader:
            labels = batch.y
            total_pos += (labels == 1).sum().item()
            total_neg += (labels == 0).sum().item()

        if self._pos_weight_override is not None:
            ratio = self._pos_weight_override
            print(f"Class balance: {total_neg:,} benign / {total_pos:,} attackers "
                  f"(pos_weight = {ratio:.4f} [manual])")
        else:
            ratio = total_neg / max(total_pos, 1)
            print(f"Class balance: {total_neg:,} benign / {total_pos:,} attackers "
                  f"(pos_weight = {ratio:.4f} [auto])")
        return torch.tensor([ratio], dtype=torch.float32, device=self.device)

    def _extract_utility_tensors(self, batch):
        batch_idx = getattr(batch, 'batch', None)
        u_all_attr = f'utility_{self.utility_metric}_all'
        u_benign_attr = f'utility_{self.utility_metric}_benign'
        utility_all = getattr(batch, u_all_attr, None)
        utility_benign = getattr(batch, u_benign_attr, None)
        return batch_idx, utility_all, utility_benign

    def _train_one_epoch(
        self,
        train_loader: PyGDataLoader,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_agg_loss = 0.0
        epoch_util_loss = 0.0
        num_batches = 0
        all_probs: list = []

        for batch in train_loader:
            batch = batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True) #Non recurrent

            batch_idx, utility_all, utility_benign = self._extract_utility_tensors(batch)

            with torch.amp.autocast(device_type=self.device.type, enabled=self._use_amp):
                logits, attention_weights = self.model(batch.x, batch.edge_index)
                labels = batch.y.unsqueeze(1)
                loss, loss_dict = self.criterion(
                    logits, labels, attention_weights,
                    batch_idx=batch_idx, utility_all=utility_all, utility_benign=utility_benign
                )

            # Backward with AMP scaler
            self._scaler.scale(loss).backward()

            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self._scaler.step(self.optimizer)
            self._scaler.update()

            with torch.no_grad():
                probs = torch.sigmoid(logits.detach().float()).flatten()
                all_probs.append(probs.cpu())

            epoch_loss += loss_dict['total']
            epoch_cls_loss += loss_dict['cls']
            epoch_agg_loss += loss_dict['agg']
            epoch_util_loss += loss_dict.get('utility', 0.0)
            num_batches += 1

        all_probs_t = torch.cat(all_probs)
        pred_min  = all_probs_t.min().item()
        pred_max  = all_probs_t.max().item()
        pred_mean = all_probs_t.mean().item()
        pred_pos_frac = (all_probs_t > self.threshold).float().mean().item()

        return {
            'train_loss': epoch_loss / max(num_batches, 1),
            'train_cls_loss': epoch_cls_loss / max(num_batches, 1),
            'train_agg_loss': epoch_agg_loss / max(num_batches, 1),
            'train_util_loss': epoch_util_loss / max(num_batches, 1),
            'pred_min': pred_min,
            'pred_max': pred_max,
            'pred_mean': pred_mean,
            'pred_pos_frac': pred_pos_frac,
        }

    @torch.no_grad()
    def _validate(
        self,
        val_loader: PyGDataLoader,
    ) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        val_cls_loss = 0.0
        val_agg_loss = 0.0
        val_util_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = batch.to(self.device, non_blocking=True)
            batch_idx, utility_all, utility_benign = self._extract_utility_tensors(batch)

            with torch.amp.autocast(device_type=self.device.type, enabled=self._use_amp):
                logits, attention_weights = self.model(batch.x, batch.edge_index)
                labels = batch.y.unsqueeze(1)
                loss, loss_dict = self.criterion(
                    logits, labels, attention_weights,
                    batch_idx=batch_idx, utility_all=utility_all, utility_benign=utility_benign
                )

            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > self.threshold).astype(int)

            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy().flatten())

            val_loss += loss_dict['total']
            val_cls_loss += loss_dict['cls']
            val_agg_loss += loss_dict['agg']
            val_util_loss += loss_dict.get('utility', 0.0)
            num_batches += 1

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        return {
            'val_loss': val_loss / max(num_batches, 1),
            'val_cls_loss': val_cls_loss / max(num_batches, 1),
            'val_agg_loss': val_agg_loss / max(num_batches, 1),
            'val_util_loss': val_util_loss / max(num_batches, 1),
            'val_accuracy': accuracy_score(all_labels, all_preds),
            'val_precision': precision_score(all_labels, all_preds, zero_division=0),
            'val_recall': recall_score(all_labels, all_preds, zero_division=0),
            'val_f1': f1_score(all_labels, all_preds, zero_division=0),
            'val_samples': len(all_labels),
        }

    def fit(
        self,
        train_loader: PyGDataLoader,
        val_loader: PyGDataLoader,
        epochs: int = 50,
        scheduler_type: str = 'cosine',
        verbose: bool = True,
    ) -> List[Dict]:
        """
        Train the model with validation and early stopping
        """
        pos_weight = self._compute_pos_weight(train_loader)
        self.criterion = CompositeLoss(
            lambda_agg=self.lambda_agg,
            lambda_utility=self.lambda_utility,
            utility_metric=self.utility_metric,
            pos_weight=pos_weight,
        )
        scheduler = None
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )

        self.history = []
        self._best_val_f1 = -1.0
        self._best_epoch = -1
        self._best_model_state = None
        patience_counter = 0

        if verbose:
            print(f"\nStarting Training ({epochs} epochs, patience={self.patience})...")
            print(f"  Model type: {self.model_type}")
            print(f"  Lambda (aggregation): {self.lambda_agg}")
            print(f"  Lambda (utility): {self.lambda_utility} ({self.utility_metric})")

        for epoch in range(epochs):
            train_metrics = self._train_one_epoch(train_loader)

            val_metrics = self._validate(val_loader)

            current_lr = self.optimizer.param_groups[0]['lr']
            if scheduler is not None:
                scheduler.step()

            # Combine metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'lr': current_lr,
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_metrics)

            # Early stopping check
            if val_metrics['val_f1'] > self._best_val_f1:
                self._best_val_f1 = val_metrics['val_f1']
                self._best_epoch = epoch + 1
                self._best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose:
                print(
                    f"  Epoch {epoch + 1:3d}/{epochs} | "
                    f"Train Loss: {train_metrics['train_loss']:.4f} "
                    f"(cls={train_metrics['train_cls_loss']:.4f}, "
                    f"agg={train_metrics['train_agg_loss']:.4f}, "
                    f"util={train_metrics['train_util_loss']:.4f}) | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val F1: {val_metrics['val_f1']:.4f} | "
                    f"LR: {current_lr:.6f}"
                    f"{' *' if patience_counter == 0 else ''}"
                )
                print(
                    f"             Pred dist → "
                    f"min={train_metrics['pred_min']:.3f}  "
                    f"max={train_metrics['pred_max']:.3f}  "
                    f"mean={train_metrics['pred_mean']:.3f}  "
                    f"frac>{self.threshold:.2f}={train_metrics['pred_pos_frac']:.4f}"
                )

            if patience_counter >= self.patience:
                if verbose:
                    print(
                        f"\n  Early stopping at epoch {epoch + 1} "
                        f"(best epoch: {self._best_epoch}, "
                        f"best val F1: {self._best_val_f1:.4f})"
                    )
                break

        # Restore best model
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            if verbose:
                print(f"\n  Restored best model from epoch {self._best_epoch}")

        if verbose:
            print("Training complete!")

        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: PyGDataLoader,
        label: str = "Test",
    ) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_labels = []

        for batch in test_loader:
            batch = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self._use_amp):
                logits, _ = self.model(batch.x, batch.edge_index)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > self.threshold).astype(int)

            all_preds.extend(preds)
            all_labels.extend(batch.y.cpu().numpy().flatten())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        metrics = {
            'Accuracy': accuracy_score(all_labels, all_preds),
            'Precision': precision_score(all_labels, all_preds, zero_division=0),
            'Recall': recall_score(all_labels, all_preds, zero_division=0),
            'F1_Score': f1_score(all_labels, all_preds, zero_division=0),
        }

        total = len(all_labels)
        pos = int(all_labels.sum())
        neg = total - pos

        print(f"\n{label} Metrics ({total:,} nodes: {neg:,} benign, {pos:,} attackers):")
        for name, val in metrics.items():
            print(f"  {name.replace('_', ' ')}: {val:.4f}")

        return metrics

    @torch.no_grad()
    def evaluate_per_graph(
        self,
        test_loader: PyGDataLoader,
    ) -> List[Dict]:
        """
        Evaluate per-graph metrics for detailed analysis.

        Returns:
            List of dicts, one per graph, with metadata and metrics.
        """
        self.model.eval()
        results = []

        def _unwrap(val):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return val.item() if val.numel() == 1 else val.tolist()
            if isinstance(val, (list, tuple)):
                return val[0] if len(val) == 1 else val
            return val

        # We need individual graphs, so iterate with batch_size=1
        for batch in test_loader:
            batch = batch.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self._use_amp):
                logits, _ = self.model(batch.x, batch.edge_index)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > self.threshold).astype(int)
            labels = batch.y.cpu().numpy().flatten()

            result = {
                'epsilon':      _unwrap(getattr(batch, 'epsilon', None)),
                'dataset_type': _unwrap(getattr(batch, 'dataset_type', None)),
                'ratio':        _unwrap(getattr(batch, 'ratio', None)),
                'protocol':     _unwrap(getattr(batch, 'protocol', None)),
                'num_nodes':    len(labels),
                'num_attackers': int(labels.sum()),
                'Accuracy':  accuracy_score(labels, preds),
                'Precision': precision_score(labels, preds, zero_division=0),
                'Recall':    recall_score(labels, preds, zero_division=0),
                'F1_Score':  f1_score(labels, preds, zero_division=0),
            }
            results.append(result)

        return results

    def save(self, path: str) -> None:
        """Save model checkpoint with training metadata."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'lambda_agg': self.lambda_agg,
            'best_val_f1': self._best_val_f1,
            'best_epoch': self._best_epoch,
            'history': self.history,
        }, path)
        print(f"Model saved to: {path}")

    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', [])
        print(f"Model loaded from: {path}")


def run_k_fold_cv(
    model_class,
    model_kwargs: Dict,
    dataset_loader,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    lambda_agg: float = 0.1,
    lambda_utility: float = 0.0,
    utility_metric: str = 'js',
    patience: int = 10,
    init_method: str = 'default',
    model_type: str = 'gat',
    device: torch.device = None,
    seed: int = 42,
    pos_weight: Optional[float] = None,
    threshold: float = 0.5,
) -> Dict:
    """
    Args:
        model_class: Model class (GATAttackerDetector or GraphSAGEAttackerDetector).
        model_kwargs: Dict of kwargs for model constructor (e.g., input_dim, hidden_dim).
        dataset_loader: GraphDatasetLoader instance.
        n_folds: Number of folds.
        epochs: Max epochs per fold.
        batch_size: Graphs per batch.
        learning_rate: Optimizer LR.
        lambda_agg: Aggregation loss weight.
        lambda_utility: Utility loss weight.
        utility_metric: Metric for utility loss ('js' or 'wasserstein').
        patience: Early stopping patience.
        init_method: Weight initialization method.
        model_type: 'gat' or 'graphsage'.
        device: Torch device.
        seed: Random seed.
        pos_weight: Positive class weight (None = auto from data).
        threshold: Classification threshold for predictions.
    """
    if device is None:
        device = get_device()

    fold_results = []

    print(f"\n{'='*70}")
    print(f"K-Fold Cross-Validation ({n_folds} folds)")
    print(f"  Model: {model_type}")
    print(f"  Init: {init_method}")
    print(f"  LR: {learning_rate}, Lambda (agg): {lambda_agg}, Lambda (utility): {lambda_utility} ({utility_metric})")
    print(f"  Patience: {patience}, Max epochs: {epochs}")
    print(f"{'='*70}")

    for fold_i, (train_idx, val_idx) in enumerate(
        dataset_loader.k_fold_split(n_folds=n_folds, seed=seed)
    ):
        print(f"\n--- Fold {fold_i + 1}/{n_folds} ---")
        print(f"  Train: {len(train_idx)} graphs, Val: {len(val_idx)} graphs")
        model = model_class(**model_kwargs)
        init_weights(model, method=init_method)

        pin = device.type == 'cuda'
        train_loader = dataset_loader.get_dataloader(
            train_idx, batch_size=batch_size, shuffle=True, pin_memory=pin
        )
        val_loader = dataset_loader.get_dataloader(
            val_idx, batch_size=batch_size, shuffle=False, pin_memory=pin
        )

        # Train
        trainer = GNNTrainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            lambda_agg=lambda_agg,
            lambda_utility=lambda_utility,
            utility_metric=utility_metric,
            patience=patience,
            model_type=model_type,
            pos_weight=pos_weight,
            threshold=threshold,
        )
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            verbose=True,
        )

        val_metrics = trainer.evaluate(val_loader, label=f"Fold {fold_i + 1} Val")
        val_metrics['best_epoch'] = trainer._best_epoch
        fold_results.append(val_metrics)

        del model, trainer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    mean_metrics = {}
    std_metrics = {}

    for metric in metric_names:
        values = [r[metric] for r in fold_results]
        mean_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)

    print(f"\n{'='*70}")
    print(f"K-Fold CV Results ({n_folds} folds)")
    print(f"{'='*70}")
    for metric in metric_names:
        print(f"  {metric.replace('_', ' ')}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
    print(f"  Avg best epoch: {np.mean([r['best_epoch'] for r in fold_results]):.1f}")
    print(f"{'='*70}")

    return {
        'fold_results': fold_results,
        'mean': mean_metrics,
        'std': std_metrics,
    }


def run_hp_search_cv(
    model_class,
    base_model_kwargs: Dict,
    dataset_loader,
    hp_grid: Dict[str, list],
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    init_method: str = 'default',
    model_type: str = 'gat',
    device: torch.device = None,
    seed: int = 42,
    base_learning_rate: float = 0.001,
    base_lambda_agg: float = 0.1,
    base_lambda_utility: float = 0.0,
    utility_metric: str = 'js',
    pos_weight: Optional[float] = None,
    threshold: float = 0.5,
) -> Dict:
    """

    Args:
        model_class: Model class (e.g. GATAttackerDetector).
        base_model_kwargs: Base kwargs for model constructor.
        dataset_loader: GraphDatasetLoader instance.
        hp_grid: Dict mapping param names to lists of values to search.
        n_folds: Number of CV folds.
        epochs: Max epochs per fold.
        batch_size: Graphs per batch.
        patience: Early stopping patience.
        init_method: Fallback weight initialization if not in grid.
        model_type: 'gat' or 'graphsage'.
        device: Torch device.
        seed: Random seed.
        base_learning_rate: LR used for all configs.
        base_lambda_agg: Fallback lambda_agg if not in grid.
        base_lambda_utility: Fallback lambda_utility if not in grid.
        utility_metric: Metric for utility loss ('js' or 'wasserstein').
        pos_weight: Positive class weight (None = auto from data).
        threshold: Classification threshold for predictions.
    """
    import itertools

    if device is None:
        device = get_device()

    # Build the search space
    param_names = sorted(hp_grid.keys())
    param_values = [hp_grid[k] for k in param_names]
    all_combos = list(itertools.product(*param_values))
    total_combos = len(all_combos)

    print(f"\n{'='*70}")
    print(f"Hyperparameter Search ({total_combos} configs × {n_folds} folds)")
    print(f"{'='*70}")
    for name in param_names:
        print(f"  {name}: {hp_grid[name]}")
    print(f"{'='*70}")

    best_mean_f1 = -1.0
    best_config = None
    best_cv_results = None
    all_results = []

    for combo_i, values in enumerate(all_combos):
        config = dict(zip(param_names, values))

        print(f"\n{'─'*70}")
        print(f"Config {combo_i + 1}/{total_combos}: {config}")
        print(f"{'─'*70}")

        # Build model kwargs for this config
        model_kwargs = dict(base_model_kwargs)
        if 'num_heads' in config:
            model_kwargs['num_heads'] = config['num_heads']

        # Training params for this config
        lr = base_learning_rate
        lam = config.get('lambda_agg', base_lambda_agg)
        lam_util = config.get('lambda_utility', base_lambda_utility)
        init = config.get('init_method', init_method)

        cv_results = run_k_fold_cv(
            model_class=model_class,
            model_kwargs=model_kwargs,
            dataset_loader=dataset_loader,
            n_folds=n_folds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            lambda_agg=lam,
            lambda_utility=lam_util,
            utility_metric=utility_metric,
            patience=patience,
            init_method=init,
            model_type=model_type,
            device=device,
            seed=seed,
            pos_weight=pos_weight,
            threshold=threshold,
        )

        mean_f1 = cv_results['mean']['F1_Score']

        result_entry = {
            **config,
            'mean_f1': mean_f1,
            'mean_accuracy': cv_results['mean']['Accuracy'],
            'mean_precision': cv_results['mean']['Precision'],
            'mean_recall': cv_results['mean']['Recall'],
            'std_f1': cv_results['std']['F1_Score'],
        }
        all_results.append(result_entry)

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_config = dict(config)
            best_cv_results = cv_results
            print(f"  ★ New best config! Mean F1 = {mean_f1:.4f}")

    print(f"\n{'='*70}")
    print(f"Hyperparameter Search Complete")
    print(f"{'='*70}")
    print(f"  Best config: {best_config}")
    print(f"  Best mean F1: {best_mean_f1:.4f}")
    print(f"  Searched {total_combos} configurations × {n_folds} folds")
    print(f"{'='*70}")

    return {
        'best_config': best_config,
        'best_cv_results': best_cv_results,
        'all_results': all_results,
    }

