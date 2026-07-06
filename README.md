# LDP Attacker Detection

A deep learning-based attacker detection system for Local Differential Privacy (LDP) protocols.

## Models

| Model | Description |
|-------|-------------|
| `mlp` | Multi-layer perceptron with BatchNorm and Dropout |
| `gan` | GAN-style discriminator with LayerNorm |
| `attention` | Transformer-style with per-feature embeddings and multi-head attention |
| `gat` | 3-layer Graph Attention Network with multi-head attention and composite loss |
| `graphsage` | 3-layer GraphSAGE with mean aggregation |

## Installation

```bash
conda env create -f environment.yml
conda activate attacker-detector
```

## Required Datasets

To generate LDP distributions, the real dataset files must be placed in a `datasets` folder in the project root:
- `datasets/zipf.npy` — Zipf dataset
- `datasets/emoji.npy` — Emoji dataset
- `datasets/fire.csv` — Fire dataset (must contain `Unit_ID` column)

## Dataset Generation


Generate LDP attack detection training data:

```bash
# Generate with defaults
python generate_dataset.py --output dataset.csv

# Custom configuration
python generate_dataset.py --output custom.csv \
    --protocols OUE OLH_Server OLH_User HST_User HST_Server \
    --epsilons 0.5 1.0 2.0 \
    --datasets zipf emoji fire \
    --ratios 0.10 0.15 0.20 \
    --experiments 5
```

### Dataset Generation CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output`, `-o` | Output CSV file path | *required* |
| `--protocols` | LDP protocols: `OUE`, `OLH` | `['OUE', 'OLH']` |
| `--epsilons` | Privacy parameters | `[0.5, 0.7, 1.0, 1.5]` |
| `--datasets` | Dataset types: `zipf`, `emoji`, `fire` | all three |
| `--ratios` | Attacker ratios | `[0.10, 0.15, 0.20]` |
| `--target-sizes` | Target set sizes | `[2, 4, 6, 8]` |
| `--splits` | Split values | `[2, 4, 6, 8]` |
| `--experiments` | Experiments per config | 5 |
| `--full-scale` | Use full-scale dataset sizes | False |
| `--n` | Override number of users | None |
| `--processors` | Parallel processes | 4 |

## PCA Graph Dataset Generation

Generate LDP attack detection training data using PCA and kNN graph extraction, outputting a list of PyTorch Geometric `Data` objects.

The script applies PCA to the per-user perturbed `support_list` matrix (shape `n × domain`) and retains a fixed number of components (default: 16). A kNN graph is then built using these PCA features, and density and influence graph features are extracted for each user node.

```bash
# Generate with defaults
python generate_dataset_pca.py --output dataset_pca.pt

# Custom configuration
python generate_dataset_pca.py --output pca.pt \
    --protocols OUE OLH_Server HST_User HST_Server \
    --epsilons 0.5 1.0 2.0 \
    --datasets zipf emoji fire \
    --ratios 0.10 0.15 0.20 \
    --experiments 5 \
    --workers 4 \
    --inner-processors 4
```

### PCA Graph Dataset CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output`, `-o` | Output `.pt` file path | *required* |
| `--protocols` | LDP protocols: `OUE`, `OLH`, `OLH_User`, `OLH_Server`, `HST_User`, `HST_Server` | `['OUE', 'OLH']` |
| `--epsilons` | Privacy parameters | `[0.5, 0.7, 1.0, 1.5]` |
| `--datasets` | Dataset types: `zipf`, `emoji`, `fire` | all three |
| `--ratios` | Attacker ratios | `[0.10, 0.15, 0.20]` |
| `--target-sizes` | Target set sizes | `[2, 4, 6, 8]` |
| `--splits` | Split values | `[2, 4, 6, 8]` |
| `--experiments` | Experiments per config | 5 |
| `--full-scale` | Use full-scale dataset sizes | False |
| `--n` | Override number of users | None |
| `--domain` | Override domain size | None |
| `--pca-dim` | Number of PCA components to retain | 16 |
| `--knn-k` | Number of neighbors for kNN graph | 10 |
| `--workers` | Outer ProcessPoolExecutor workers (OUE/HST) | 4 |
| `--inner-processors` | Inner parallel processes per task | 4 |
| `--append` | Append to existing output file | False |

### Output `.pt` Format

The output file is a list of PyTorch Geometric `torch_geometric.data.Data` objects. Each object represents an experiment run:
```python
Data(
    x=FloatTensor[n, 24],        # Node features
    edge_index=LongTensor[2, E],   # Symmetric kNN edges
    y=FloatTensor[n],            # Labels: 0 = benign, 1 = attacker
    # Graph-level metadata:
    epsilon=float,
    protocol=str,
    dataset_type=str,
    ratio=float,
    target_set_size=int,
    splits=int,
    pca_variance_explained=float
)
```

### Node Feature Composition (40 Dims)

| Feature Block | Dims | Description |
|--------------|------|-------------|
| PCA coordinates | 32 (or `pca_dim`) | `pc_0` ... `pc_31` |
| Density features | 4 | `avg_knn_dist`, `local_density`, `relative_density`, `knn_dist_std` |
| Influence features | 3 | `in_degree`, `degree_centrality`, `hub_score` |
| Epsilon | 1 | Epsilon value (broadcasted to all nodes) |

### Parallelism

- **OUE, HST_User, HST_Server** tasks are executed in parallel via `ProcessPoolExecutor` with `--workers` controlling concurrency.
- **OLH, OLH_User, OLH_Server** tasks run **sequentially** in the main process because OLH protocols already use inner `multiprocessing.Pool` for hash-function search and user-seed processing. Nesting process pools would cause deadlocks or excessive resource contention.




## Usage

```bash
# Train MLP model
python main.py --data-path dataset.csv --model mlp --epochs 5

# Train GAN model
python main.py --data-path dataset.csv --model gan --epochs 10

# Train Attention model
python main.py --data-path dataset.csv --model attention --epochs 15

# Save outputs to directory
python main.py --data-path dataset.csv --model mlp --output-dir ./results

# Cross-dataset: train on zipf, test on emoji
python main.py --data-path dataset.csv --model mlp \
    --training-method cross --train-dataset zipf --test-dataset emoji

# Three-way: train on zipf, test on emoji, evaluate on fire
python main.py --data-path dataset.csv --model mlp \
    --training-method three-way \
    --train-dataset zipf --test-dataset emoji --eval-dataset fire
```
### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-path`, `-d` | Path to CSV dataset | *required* |
| `--model`, `-m` | Model type: `mlp`, `gan`, `attention` | *required* |
| `--epochs`, `-e` | Training epochs | 5 |
| `--batch-size`, `-b` | Batch size | 256 |
| `--lr` | Learning rate | 0.001 |
| `--dropout` | Dropout rate | 0.2 |
| `--test-size` | Test split ratio (only for `none` mode) | 0.2 |
| `--seed` | Random seed | 42 |
| `--output-dir`, `-o` | Save model/plots here | None |
| `--no-plot` | Skip sensitivity plots | False |

### Generalizability Training

| Argument | Description | Default |
|----------|-------------|---------|
| `--training-method` | `none` (conventional random split), `cross` (train on one dataset type, test on another), `three-way` (train/test/eval on separate types) | `none` |
| `--train-dataset` | Dataset type for training: `zipf`, `emoji`, `fire` (required for `cross` / `three-way`) | None |
| `--test-dataset` | Dataset type for testing: `zipf`, `emoji`, `fire` (required for `cross` / `three-way`) | None |
| `--eval-dataset` | Dataset type for evaluation: `zipf`, `emoji`, `fire` (required for `three-way`) | None |

**Training methods:**

- **`none`** — Conventional training. All dataset types are mixed together and split randomly into train/test sets using `--test-size`.
- **`cross`** — Cross-dataset generalizability. The model is trained entirely on one dataset type and tested on another (e.g., train on `zipf`, test on `emoji`).
- **`three-way`** — Full generalizability evaluation. Train on one type, test on a second, and run an additional evaluation + sensitivity analysis on a third (e.g., train `zipf`, test `emoji`, evaluate `fire`).

## GNN Training (`main_gnn.py`)

Train and evaluate Graph Attention Network (GAT) or GraphSAGE models on the PCA graph dataset (`.pt` file generated by `generate_dataset_pca.py`). Each graph represents one experiment run (~5,000 nodes, one node per user).

### GNN Models

| Model | Description |
|-------|-------------|
| `gat` | 3-layer Graph Attention Network with multi-head attention. Returns attention weights used in composite loss. |
| `graphsage` | 3-layer GraphSAGE with mean aggregation. No attention weights; aggregation loss term is zero. |

### Usage

```bash
# Quick run — GAT on 10 graphs (50k nodes), save to ./results
python main_gnn.py \
    --data-path dataset_pca.pt \
    --model gat \
    --max-graphs 10 \
    --epochs 10 \
    --patience 5 \
    --batch-size 4 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --output-dir ./results

# Full dataset run — GraphSAGE
python main_gnn.py \
    --data-path dataset_pca.pt \
    --model graphsage \
    --epochs 50 \
    --output-dir ./results_sage

# With k-fold cross-validation (5 folds)
python main_gnn.py \
    --data-path dataset_pca.pt \
    --model gat \
    --k-folds 5 \
    --epochs 50 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --output-dir ./results_cv

# Custom HP search grid
python main_gnn.py \
    --data-path dataset_pca.pt \
    --model gat \
    --k-folds 5 \
    --hp-lambda-agg 0.05 0.1 0.2 \
    --hp-num-heads 2 4 8 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --output-dir ./results_hp

# Direct train + test (no CV, no HP search)
python main_gnn.py \
    --data-path dataset_pca.pt \
    --model gat \
    --epochs 50 \
    --batch-size 4 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --output-dir ./results

# k-fold CV for validation only (no HP search)
python main_gnn.py \
    --data-path dataset_pca.pt \
    --model gat \
    --k-folds 5 \
    --no-hp-search \
    --batch-size 4 \
    --test-ratio 0.2 \
    --val-ratio 0.2 \
    --output-dir ./results_cv_only
```

When `--k-folds > 0`, a grid search runs over `lambda_agg` × `num_heads` (GAT) × `init_method` combinations by default. The best config by mean validation F1 is selected and used for final training. Use `--no-hp-search` to skip the grid search and use CLI params directly. Use `--cv-only` to skip final training and only output CV results.

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-path`, `-d` | Path to `.pt` graph dataset | `dataset_pca.pt` |
| `--model`, `-m` | Model type: `gat`, `graphsage` | *required* |
| `--max-graphs` | Limit graphs loaded (for debugging/quick runs) | None (load all) |
| `--hidden-dim` | Hidden dimension per attention head | 64 |
| `--num-heads` | Attention heads in GAT layers 1 & 2 | 4 |
| `--dropout` | Dropout rate | 0.2 |
| `--epochs`, `-e` | Maximum training epochs | 5 |
| `--lr` | Learning rate (AdamW) | 0.001 |
| `--lambda-agg` | Attention entropy regularization weight λ_agg | 0.1 |
| `--lambda-utility` | Utility-aware loss weight λ_utility | 0.0 |
| `--utility-metric` | Utility metric: `js` (Jensen-Shannon divergence) or `wasserstein` (Wasserstein distance) | `js` |
| `--pos-weight` | Positive class weight for BCE loss: `auto` (neg/pos ratio from data) or a float | `auto` |
| `--threshold` | Classification threshold (lower → more attacker predictions, higher recall) | 0.5 |
| `--batch-size`, `-b` | Graphs per batch | 32 |
| `--patience` | Early stopping patience (epochs without val F1 improvement) | 10 |
| `--init-method` | Weight initialization: `xavier_uniform`, `kaiming`, `orthogonal`, `default` | `xavier_uniform` |
| `--k-folds` | Number of k-fold CV folds (0 = skip CV) | 0 |
| `--hp-lambda-agg` | Lambda agg values to search | `[0.05, 0.1, 0.2]` |
| `--hp-lambda-utility` | Lambda utility values to search | None |
| `--hp-num-heads` | Attention heads to search (GAT only) | `[2, 4, 8]` |
| `--hp-init-method` | Weight init methods to search | `[xavier_uniform, kaiming, orthogonal]` |
| `--cv-only` | Run only CV / HP search, skip final training + test | False |
| `--no-hp-search` | Skip HP grid search; use CLI params directly | False |
| `--test-ratio` | Fraction of graphs for test set | 0.15 |
| `--val-ratio` | Fraction of graphs for validation set | 0.15 |
| `--seed` | Random seed | 42 |
| `--output-dir`, `-o` | Directory to save model, metrics, and plots | None |

### Loss Function

The training uses a **composite loss**:

```
L = L_classification  +  λ_agg × L_aggregation  +  λ_utility × L_utility
```

- **L_classification** — `BCEWithLogitsLoss` with positive class weighting. `--pos-weight auto` computes `neg_count / pos_count` from training data (e.g. ~5.77 for a 85/15 split). A manual float can be supplied to override (e.g. `--pos-weight 1.0` for uniform weighting).
- **L_aggregation** — Attention entropy regularizer (GAT only). Penalises uniform attention distributions across neighbours, encouraging peaked/discriminative attention. Zero for GraphSAGE.
- **L_utility** — Utility-aware loss. Rewards attacker predictions that bring the LDP frequency distribution closer to the ground truth distribution (`REAL_DIST`). Measures JS divergence or Wasserstein distance improvements toward the optimal benign-only baseline. Controlled by `--lambda-utility` and `--utility-metric`.

### Output Files

When `--output-dir` is specified:

| File | Contents |
|------|----------|
| `gat_model.pt` / `graphsage_model.pt` | Model checkpoint (weights + optimizer state + history) |
| `training_history.csv` | Per-epoch: total loss, cls loss, agg loss, val F1, val accuracy, LR |
| `hp_search_results.csv` | Per-config HP search summary: config values, mean F1, accuracy, precision, recall |
| `cv_results.csv` | Fold-level results for the best HP config |
| `test_metrics.csv` | Overall test: Accuracy, Precision, Recall, F1_Score |
| `per_graph_results.csv` | Per-graph metrics with ε, ratio, protocol, dataset_type |
| `sensitivity_test_results.csv` | Sensitivity table grouped by parameter (matches `main.py` format) |
| `sensitivity_f1_score.png` | F1 Score vs. ε, ratio, protocol, dataset_type |
| `sensitivity_accuracy.png` | Accuracy sensitivity plot |
| `sensitivity_precision.png` | Precision sensitivity plot |
| `sensitivity_recall.png` | Recall sensitivity plot |

### GPU Setup

The trainer auto-detects CUDA via `get_device()`. To enable GPU:

1. Update `environment.yml` — replace `cpuonly` with `pytorch-cuda=12.1` (or match your driver version).
2. Reinstall PyTorch:
   ```bash
   pip install --force-reinstall torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/cu121
   ```
3. Install CUDA runtime into the conda env:
   ```bash
   conda install -c nvidia cuda-toolkit=12.1 cudnn
   ```
4. Verify:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

GPU optimizations enabled automatically when CUDA is available:
- Mixed precision (AMP) via `torch.amp.autocast`
- `pin_memory=True` DataLoaders for faster CPU→GPU transfers
- `non_blocking=True` batch transfers
- `cudnn.benchmark=True` for fixed-size inputs

## License

MIT
