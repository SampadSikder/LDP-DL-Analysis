"""Training loop and utilities."""

from typing import Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..data import AttackerDataset


def _ft_transformer_param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        name_lower = name.lower()
        if (
            param.ndim <= 1
            or 'norm' in name_lower
            or 'tokenizer' in name_lower
            or 'cls_token' in name_lower
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


class Trainer:
    """
    Trainer class for attacker detection models.
    
    Args:
        model: PyTorch model to train
        device: Device to train on ('cpu' or 'cuda')
        learning_rate: Optimizer learning rate
        model_type: Type of model ('mlp', 'gan', 'attention') - affects optimizer config
        epochs: Number of epochs (needed for attention scheduler)
        pos_weight: Positive class weight for BCEWithLogitsLoss. None = auto-compute
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        model_type: str = 'mlp',
        epochs: int = 5,
        pos_weight: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.scheduler = None
        self._pos_weight_override = pos_weight

        if model_type == 'gan':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        elif model_type == 'attention':
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        elif model_type == 'ft_transformer':
            param_groups = _ft_transformer_param_groups(model, weight_decay=0.01)
            self.optimizer = optim.AdamW(param_groups, lr=learning_rate)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.criterion = None

    def _resolve_pos_weight(self, y_train: np.ndarray) -> float:
        if self._pos_weight_override is not None:
            return self._pos_weight_override
        num_benign = (y_train == 0).sum()
        num_attackers = (y_train == 1).sum()
        return float(num_benign) / max(float(num_attackers), 1.0)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 5,
        batch_size: int = 256
    ) -> dict:
        """
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Dict with 'history' (list of per-epoch {'epoch', 'train_loss'} dicts).
        """
        # Calculate class imbalance
        num_benign = (y_train == 0).sum()
        num_attackers = (y_train == 1).sum()
        ratio = self._resolve_pos_weight(y_train)
        print(f"Class Imbalance: {num_benign:,} benign / {num_attackers:,} attackers (pos_weight = {ratio:.4f})")

        # Set up weighted loss
        pos_weight = torch.tensor([ratio]).float().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Create data loader
        train_dataset = AttackerDataset(X_train, y_train)
        pin_memory = self.device.type == 'cuda'
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
        )

        # Training loop
        print(f"\nStarting Training ({epochs} epochs)...")
        self.model.train()
        history = []

        for epoch in range(epochs):
            epoch_loss = torch.zeros((), device=self.device)
            num_batches = 0

            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                features = features.to(self.device, non_blocking=pin_memory)
                labels = labels.to(self.device, non_blocking=pin_memory).unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # Gradient clipping for attention-based models
                if self.model_type in ('attention', 'ft_transformer'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Accumulate on-device; sync to a Python float once per epoch
                # (below) rather than once per batch.
                epoch_loss += loss.detach()
                num_batches += 1

            # Step scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = (epoch_loss / num_batches).item()
            epoch_entry = {'epoch': epoch + 1, 'train_loss': avg_loss}
            if self.scheduler is not None:
                epoch_entry['lr'] = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, LR: {epoch_entry['lr']:.6f}")
            else:
                print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")
            history.append(epoch_entry)

        print("Training complete!")
        return {'history': history}

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to: {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from: {path}")

    def fit_with_validation(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Train with early stopping based on validation F1 score.

        Returns:
            Dict with 'best_epoch', 'best_val_f1', 'history' (list of per-epoch dicts).
        """
        # Class imbalance weight
        num_benign = (y_train == 0).sum()
        num_attackers = (y_train == 1).sum()
        ratio = self._resolve_pos_weight(y_train)
        if verbose:
            print(f"Class Imbalance: {num_benign:,} benign / {num_attackers:,} attackers (pos_weight = {ratio:.4f})")

        pos_weight = torch.tensor([ratio]).float().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_dataset = AttackerDataset(X_train, y_train)
        pin_memory = self.device.type == 'cuda'
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory,
        )

        best_val_f1 = -1.0
        best_epoch = 0
        best_state = None
        epochs_no_improve = 0
        history = []

        if verbose:
            print(f"\nTraining with early stopping (patience={patience}, max_epochs={epochs})...")

        for epoch in range(epochs):
            # --- Train ---
            self.model.train()
            epoch_loss = torch.zeros((), device=self.device)
            num_batches = 0

            for features, labels in train_loader:
                features = features.to(self.device, non_blocking=pin_memory)
                labels = labels.to(self.device, non_blocking=pin_memory).unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.model_type in ('attention', 'ft_transformer'):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                epoch_loss += loss.detach()
                num_batches += 1

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = (epoch_loss / num_batches).item()

            # --- Validate ---
            val_metrics = self.evaluate(X_val, y_val, batch_size=batch_size, label=None)
            val_f1 = val_metrics['F1_Score']

            history.append({
                'epoch':     epoch + 1,
                'train_loss': avg_loss,
                'val_f1':    val_f1,
                'val_acc':   val_metrics['Accuracy'],
            })

            improved = val_f1 > best_val_f1
            if improved:
                best_val_f1 = val_f1
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if verbose:
                marker = " *" if improved else ""
                print(
                    f"  Epoch {epoch+1:3d}/{epochs} — "
                    f"loss={avg_loss:.4f}  val_f1={val_f1:.4f}  val_acc={val_metrics['Accuracy']:.4f}"
                    f"{marker}"
                )

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if verbose:
            print(f"  Best epoch: {best_epoch}, best val F1: {best_val_f1:.4f}")

        return {
            'best_epoch':  best_epoch,
            'best_val_f1': best_val_f1,
            'history':     history,
        }

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 4096,
        label: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate the model and return aggregate metrics.

        Args:
            X: Scaled feature array.
            y: Ground-truth labels.
            batch_size: Batch size for inference.
            label: Label for print output (e.g. 'Test', 'Eval'). None = silent.

        Returns:
            Dict with Accuracy, Precision, Recall, F1_Score.
        """
        self.model.eval()
        all_probs = []

        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                probs = torch.sigmoid(self.model(batch))
                all_probs.append(probs.cpu().numpy().flatten())
            del batch

        probs = np.concatenate(all_probs)
        preds = (probs > 0.5).astype(int)

        metrics = {
            'Accuracy': accuracy_score(y, preds),
            'Precision': precision_score(y, preds, zero_division=0),
            'Recall': recall_score(y, preds, zero_division=0),
            'F1_Score': f1_score(y, preds, zero_division=0),
        }

        if label is not None:
            print(f"\n{label} Metrics ({len(y)} samples):")
            for name, val in metrics.items():
                print(f"  {name.replace('_', ' ')}: {val:.4f}")

        return metrics


def run_k_fold_cv(
    model_type: str,
    input_dim: int,
    dropout_rate: float,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    patience: int = 10,
    device: torch.device = None,
    seed: int = 42,
    pos_weight: Optional[float] = None,
    hidden_sizes: Optional[list] = None,
    ft_config: Optional[dict] = None,
) -> dict:
    """
    Stratified k-fold cross-validation for tabular MLP / attention / GAN / FT-Transformer models.

    Args:
        model_type:   One of 'mlp', 'attention', 'gan', 'ft_transformer'.
        input_dim:    Number of input features.
        dropout_rate: Dropout rate for the model.
        hidden_sizes: Hidden layer widths, 'mlp' only. None uses the model default.
        X_trainval:   Combined train+val feature array (N, F).
        y_trainval:   Combined train+val label array (N,).
        n_folds:      Number of CV folds.
        epochs:       Max epochs per fold.
        batch_size:   Training batch size.
        learning_rate: Optimizer LR.
        patience:     Early stopping patience.
        device:       Torch device.
        seed:         Random seed.
        pos_weight:   Positive class weight for BCE loss. None = auto-compute
            neg/pos ratio per fold; a float pins it to that fixed value.
        ft_config:    FT-Transformer hyperparameters (d_token, n_heads,
            n_layers, ffn_d_multiplier, attention_dropout, residual_dropout),
            'ft_transformer' only. None uses the model defaults.

    Returns:
        Dict with 'fold_results' (list of per-fold metric dicts),
        'mean' and 'std' (summary dicts).
    """
    from sklearn.model_selection import StratifiedKFold
    from attacker_detector.models import get_model

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    print(f"\n{'='*70}")
    print(f"K-Fold Cross-Validation ({n_folds} folds)")
    print(f"  Model: {model_type}")
    if model_type == 'mlp' and hidden_sizes is not None:
        print(f"  Hidden sizes: {hidden_sizes}")
    if model_type == 'ft_transformer' and ft_config:
        print(f"  FT-Transformer config: {ft_config}")
    print(f"  LR: {learning_rate}, Patience: {patience}, Max epochs: {epochs}")
    print(f"{'='*70}")

    for fold_i, (train_idx, val_idx) in enumerate(
        skf.split(X_trainval, y_trainval.astype(int))
    ):
        print(f"\n--- Fold {fold_i + 1}/{n_folds} ---")
        print(f"  Train: {len(train_idx):,} samples, Val: {len(val_idx):,} samples")

        X_tr, y_tr = X_trainval[train_idx], y_trainval[train_idx]
        X_va, y_va = X_trainval[val_idx],   y_trainval[val_idx]

        model_kwargs = {'dropout_rate': dropout_rate}
        if model_type == 'mlp' and hidden_sizes is not None:
            model_kwargs['hidden_sizes'] = hidden_sizes
        if model_type == 'ft_transformer' and ft_config:
            model_kwargs.update(ft_config)
        model = get_model(model_type, input_dim=input_dim, **model_kwargs)
        trainer = Trainer(
            model, device,
            learning_rate=learning_rate,
            model_type=model_type,
            epochs=epochs,
            pos_weight=pos_weight,
        )

        result = trainer.fit_with_validation(
            X_tr, y_tr, X_va, y_va,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=True,
        )

        val_metrics = trainer.evaluate(X_va, y_va, label=f"Fold {fold_i + 1} Val")
        val_metrics['best_epoch'] = result['best_epoch']
        fold_results.append(val_metrics)

        # Drop references so CUDA memory is reclaimed by the allocator's cache
        # (and reused by the next fold) without forcing a driver-level free.
        del model, trainer

    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    mean_metrics = {}
    std_metrics = {}

    for metric in metric_names:
        values = [r[metric] for r in fold_results]
        mean_metrics[metric] = np.mean(values)
        std_metrics[metric]  = np.std(values)

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
    model_type: str,
    input_dim: int,
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    hp_grid: Dict[str, list],
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 256,
    patience: int = 10,
    device: torch.device = None,
    seed: int = 42,
    base_learning_rate: float = 0.001,
    base_dropout_rate: float = 0.2,
    base_pos_weight: Optional[float] = None,
    base_hidden_sizes: Optional[list] = None,
    base_ft_config: Optional[dict] = None,
) -> dict:

    import itertools
    from attacker_detector.models.ft_transformer import FT_TRANSFORMER_HP_KEYS

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    param_names = sorted(hp_grid.keys())
    param_values = [hp_grid[k] for k in param_names]
    all_combos = [dict(zip(param_names, values)) for values in itertools.product(*param_values)]

    if model_type == 'ft_transformer':
        valid_combos = []
        n_skipped = 0
        for config in all_combos:
            d_token = config.get('d_token')
            n_heads = config.get('n_heads')
            if d_token is not None and n_heads is not None and d_token % n_heads != 0:
                n_skipped += 1
                continue
            valid_combos.append(config)
        if n_skipped:
            print(
                f"Skipping {n_skipped} invalid config(s) "
                f"where d_token is not divisible by n_heads"
            )
        all_combos = valid_combos

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

    for combo_i, config in enumerate(all_combos):
        print(f"\n{'─'*70}")
        print(f"Config {combo_i + 1}/{total_combos}: {config}")
        print(f"{'─'*70}")

        lr = config.get('lr', base_learning_rate)
        dropout = config.get('dropout', base_dropout_rate)
        pos_weight = config.get('pos_weight', base_pos_weight)
        hidden_sizes = config.get('hidden_sizes', base_hidden_sizes)

        ft_config = None
        if model_type == 'ft_transformer':
            ft_config = dict(base_ft_config) if base_ft_config else {}
            ft_config.update({k: v for k, v in config.items() if k in FT_TRANSFORMER_HP_KEYS})

        cv_results = run_k_fold_cv(
            model_type=model_type,
            input_dim=input_dim,
            dropout_rate=dropout,
            X_trainval=X_trainval,
            y_trainval=y_trainval,
            n_folds=n_folds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            patience=patience,
            device=device,
            seed=seed,
            pos_weight=pos_weight,
            hidden_sizes=hidden_sizes,
            ft_config=ft_config,
        )

        mean_f1 = cv_results['mean']['F1_Score']

        result_entry = {
            **config,
            'pos_weight': 'auto' if pos_weight is None else pos_weight,
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
