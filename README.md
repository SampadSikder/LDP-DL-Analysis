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

## PCA Dataset Generation

Generate LDP attack detection training data using PCA on the raw perturbed support vectors instead of hand-crafted statistical features.

The script applies PCA to the per-user perturbed `support_list` matrix (shape `n × domain`) and retains components using a two-part heuristic:

1. **Kaiser criterion**: keep components whose eigenvalue > 1 (meaningful on standardized data)
2. **Variance floor**: keep enough components to explain ≥ the specified cumulative variance threshold (default 90%)

The final number of components is `max(kaiser_k, variance_k)`.

```bash
# Generate with defaults
python generate_dataset_pca.py --output dataset_pca.csv

# Custom configuration
python generate_dataset_pca.py --output pca.csv \
    --protocols OUE OLH_Server HST_User HST_Server \
    --epsilons 0.5 1.0 2.0 \
    --datasets zipf emoji fire \
    --ratios 0.10 0.15 0.20 \
    --experiments 5 \
    --workers 4 \
    --inner-processors 4

# Higher variance threshold with dimension cap
python generate_dataset_pca.py --output pca_strict.csv \
    --variance-threshold 0.95 \
    --max-pca-dim 100
```

### PCA Dataset CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output`, `-o` | Output CSV file path | *required* |
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
| `--variance-threshold` | Minimum cumulative explained variance for PCA | 0.90 |
| `--max-pca-dim` | Hard cap on PCA components (None = no cap) | None |
| `--workers` | Outer ProcessPoolExecutor workers (OUE/HST) | 4 |
| `--inner-processors` | Inner parallel processes per task | 4 |
| `--append` | Append to existing output file | False |

### Output CSV Format

| Column | Description |
|--------|-------------|
| `pc_0`, `pc_1`, ... | PCA-projected feature columns |
| `pca_dim` | Number of PCA components retained for this row |
| `pca_variance_explained` | Cumulative variance explained |
| `target_set_size` | Config metadata |
| `attacker_ratio` | Config metadata |
| `protocol` | Config metadata |
| `splits` | Config metadata |
| `epsilon` | Config metadata |
| `dataset_type` | Config metadata |
| `label` | 0 = benign, 1 = attacker |

> **Note:** Different experiments may produce different numbers of PCA columns depending on the domain size and data characteristics. Each chunk is written independently, so the CSV may have varying column widths across rows. The `pca_dim` column records the actual dimensionality for each row.

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
