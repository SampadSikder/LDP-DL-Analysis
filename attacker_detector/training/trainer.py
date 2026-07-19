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


class Trainer:
    """
    Trainer class for attacker detection models.
    
    Args:
        model: PyTorch model to train
        device: Device to train on ('cpu' or 'cuda')
        learning_rate: Optimizer learning rate
        model_type: Type of model ('mlp', 'gan', 'attention') - affects optimizer config
        epochs: Number of epochs (needed for attention scheduler)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        model_type: str = 'mlp',
        epochs: int = 5
    ):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.scheduler = None
        
        if model_type == 'gan':
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        elif model_type == 'attention':
            self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.criterion = None  
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 5,
        batch_size: int = 256
    ) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        # Calculate class imbalance
        num_benign = (y_train == 0).sum()
        num_attackers = (y_train == 1).sum()
        # Currently using 1 as severe imbalance 5.77:1 Benign:Attacker
        ratio = 1.0
        print(f"Class Imbalance: {num_benign:,} benign / {num_attackers:,} attackers (pos_weight = {ratio:.4f})")
        
        # Set up weighted loss
        pos_weight = torch.tensor([ratio]).float().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Create data loader
        train_dataset = AttackerDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Training loop
        print(f"\nStarting Training ({epochs} epochs)...")
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping for attention model
                if self.model_type == 'attention':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Step scheduler if exists
            if self.scheduler is not None:
                self.scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            if self.scheduler is not None:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            else:
                print(f"  Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")
        
        print("Training complete!")
    
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
        ratio = 1.0
        if verbose:
            print(f"Class Imbalance: {num_benign:,} benign / {num_attackers:,} attackers (pos_weight = {ratio:.4f})")

        pos_weight = torch.tensor([ratio]).float().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_dataset = AttackerDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
            epoch_loss = 0.0
            num_batches = 0

            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                loss.backward()

                if self.model_type == 'attention':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            if self.scheduler is not None:
                self.scheduler.step()

            avg_loss = epoch_loss / num_batches

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
) -> dict:
    """
    Stratified k-fold cross-validation for tabular MLP / attention / GAN models.

    Args:
        model_type:   One of 'mlp', 'attention', 'gan'.
        input_dim:    Number of input features.
        dropout_rate: Dropout rate for the model.
        X_trainval:   Combined train+val feature array (N, F).
        y_trainval:   Combined train+val label array (N,).
        n_folds:      Number of CV folds.
        epochs:       Max epochs per fold.
        batch_size:   Training batch size.
        learning_rate: Optimizer LR.
        patience:     Early stopping patience.
        device:       Torch device.
        seed:         Random seed.

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
    print(f"  LR: {learning_rate}, Patience: {patience}, Max epochs: {epochs}")
    print(f"{'='*70}")

    for fold_i, (train_idx, val_idx) in enumerate(
        skf.split(X_trainval, y_trainval.astype(int))
    ):
        print(f"\n--- Fold {fold_i + 1}/{n_folds} ---")
        print(f"  Train: {len(train_idx):,} samples, Val: {len(val_idx):,} samples")

        X_tr, y_tr = X_trainval[train_idx], y_trainval[train_idx]
        X_va, y_va = X_trainval[val_idx],   y_trainval[val_idx]

        model = get_model(model_type, input_dim=input_dim, dropout_rate=dropout_rate)
        trainer = Trainer(
            model, device,
            learning_rate=learning_rate,
            model_type=model_type,
            epochs=epochs,
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

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
