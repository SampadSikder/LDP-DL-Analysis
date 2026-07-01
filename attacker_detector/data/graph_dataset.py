from typing import List, Tuple, Dict, Optional, Generator
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import StratifiedKFold


class GraphDatasetLoader:
    """
    Supports:
        - Lazy loading of graphs from disk
        - Stratified train/val/test splits at graph level
        - K-fold cross-validation at graph level
        - PyG DataLoader creation for batched trainin
    """

    def __init__(self, pt_path: str, max_graphs: Optional[int] = None):
        self.pt_path = pt_path
        self._graphs: List[Data] = []
        self._loaded = False
        self.max_graphs = max_graphs

    def _ensure_loaded(self):
        if not self._loaded:
            print(f"Loading graph dataset from: {self.pt_path}")
            raw = torch.load(self.pt_path, weights_only=False)
            if not isinstance(raw, list):
                raise ValueError(
                    f"Expected .pt file to contain a list of Data objects, "
                    f"got {type(raw)}"
                )
            if self.max_graphs is not None:
                raw = raw[:self.max_graphs]
            self._graphs = raw
            self._loaded = True
            print(f"  Loaded {len(self._graphs)} graphs")
            self._print_summary()

    def _print_summary(self):
        if not self._graphs:
            return

        total_nodes = sum(g.x.shape[0] for g in self._graphs)
        total_edges = sum(g.edge_index.shape[1] for g in self._graphs)
        total_attackers = sum(int(g.y.sum().item()) for g in self._graphs)
        total_benign = total_nodes - total_attackers

        dataset_types = set()
        epsilons = set()
        ratios = set()
        protocols = set()
        for g in self._graphs:
            if hasattr(g, 'dataset_type'):
                dataset_types.add(g.dataset_type)
            if hasattr(g, 'epsilon'):
                epsilons.add(g.epsilon)
            if hasattr(g, 'ratio'):
                ratios.add(g.ratio)
            if hasattr(g, 'protocol'):
                protocols.add(g.protocol)

        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Total edges: {total_edges:,}")
        print(f"  Benign: {total_benign:,} ({100.0 * total_benign / total_nodes:.1f}%)")
        print(f"  Attackers: {total_attackers:,} ({100.0 * total_attackers / total_nodes:.1f}%)")
        print(f"  Feature dim: {self._graphs[0].x.shape[1]}")
        print(f"  Dataset types: {sorted(dataset_types)}")
        print(f"  Protocols: {sorted(protocols)}")
        print(f"  Epsilons: {sorted(epsilons)}")
        print(f"  Ratios: {sorted(ratios)}")

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._graphs)

    def __getitem__(self, idx) -> Data:
        self._ensure_loaded()
        return self._graphs[idx]

    @property
    def graphs(self) -> List[Data]:
        """Access the full list of graphs."""
        self._ensure_loaded()
        return self._graphs

    @property
    def input_dim(self) -> int:
        """Number of node features."""
        self._ensure_loaded()
        return self._graphs[0].x.shape[1]

    def _get_stratification_labels(self) -> np.ndarray:
        """
        Build a stratification key per graph from metadata.

        Combines dataset_type and ratio to ensure balanced splits
        across experimental conditions.
        """
        labels = []
        for g in self._graphs:
            dt = getattr(g, 'dataset_type', 'unknown')
            ratio = getattr(g, 'ratio', 0.0)
            labels.append(f"{dt}_{ratio}")
        return np.array(labels)

    def stratified_split(
        self,
        test_ratio: float = 0.15,
        val_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[int]]:
        """
        Returns:
            Dict with 'train', 'val', 'test' keys mapping to lists of graph indices.
        """
        self._ensure_loaded()
        n = len(self._graphs)
        strat_labels = self._get_stratification_labels()

        rng = np.random.RandomState(seed)
        indices = np.arange(n)

        # Fall back to simple random split when only 1 unique class exists
        # (e.g. small homogeneous subsets where StratifiedKFold would crash)
        if len(set(strat_labels)) < 2:
            rng.shuffle(indices)
            n_test = max(1, int(n * test_ratio))
            n_val  = max(1, int(n * val_ratio))
            n_train = max(1, n - n_test - n_val)
            result = {
                'train': indices[:n_train].tolist(),
                'val':   indices[n_train:n_train + n_val].tolist(),
                'test':  indices[n_train + n_val:].tolist(),
            }
            print(f"\nGraph-level split (random, seed={seed}):")
            print(f"  Train: {len(result['train'])} graphs")
            print(f"  Val:   {len(result['val'])} graphs")
            print(f"  Test:  {len(result['test'])} graphs")
            return result

        # First split: separate test set
        n_test = max(1, int(n * test_ratio))
        n_remaining = n - n_test

        skf_test = StratifiedKFold(
            n_splits=max(2, int(round(1.0 / test_ratio))),
            shuffle=True,
            random_state=seed
        )
        train_val_idx, test_idx = next(skf_test.split(indices, strat_labels))

        # Second split: separate val from train
        val_fraction_of_remaining = val_ratio / (1.0 - test_ratio)
        n_splits_val = max(2, int(round(1.0 / val_fraction_of_remaining)))

        remaining_labels = strat_labels[train_val_idx]
        skf_val = StratifiedKFold(
            n_splits=n_splits_val,
            shuffle=True,
            random_state=seed + 1
        )
        train_rel_idx, val_rel_idx = next(
            skf_val.split(train_val_idx, remaining_labels)
        )

        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]

        result = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist(),
        }

        print(f"\nGraph-level split (seed={seed}):")
        print(f"  Train: {len(result['train'])} graphs")
        print(f"  Val:   {len(result['val'])} graphs")
        print(f"  Test:  {len(result['test'])} graphs")

        return result

    def k_fold_split(
        self,
        n_folds: int = 5,
        seed: int = 42
    ) -> Generator[Tuple[List[int], List[int]], None, None]:
        """        Args:
            n_folds: Number of folds.
            seed: Random seed.        """
        self._ensure_loaded()
        n = len(self._graphs)
        strat_labels = self._get_stratification_labels()
        indices = np.arange(n)

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed
        )

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(indices, strat_labels)):
            yield train_idx.tolist(), val_idx.tolist()

    def get_dataloader(
        self,
        indices: List[int],
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = False,
    ) -> PyGDataLoader:
        """
        Create a PyG DataLoader for a subset of graphs.

        Args:
            indices: Graph indices to include.
            batch_size: Number of graphs per batch.
            shuffle: Whether to shuffle.

        Returns:
            PyG DataLoader that batches graphs into Batch objects.
        """
        self._ensure_loaded()
        subset = [self._graphs[i] for i in indices]
        return PyGDataLoader(subset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)
