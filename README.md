# LDP Attacker Detection

A deep learning-based attacker detection system for Local Differential Privacy (LDP) protocols.

## Models

| Model | Description |
|-------|-------------|
| `mlp` | Multi-layer perceptron with BatchNorm and Dropout |
| `gan` | GAN-style discriminator with LayerNorm |
| `attention` | Transformer-style with per-feature embeddings and multi-head attention |

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

### Node Feature Composition (24 Dims)

| Feature Block | Dims | Description |
|--------------|------|-------------|
| PCA coordinates | 16 (or `pca_dim`) | `pc_0` ... `pc_15` |
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

## License

MIT
