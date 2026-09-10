# LDP Attacker Detection

A deep learning-based attacker detection system for Local Differential Privacy (LDP) protocols.

## Models

| Model | Description |
|-------|-------------|
| `mlp` | Multi-layer perceptron with BatchNorm and Dropout |
| `gan` | GAN-style discriminator with LayerNorm |
| `attention` | Transformer-style with per-feature embeddings and multi-head attention |
| `ft_transformer` | Feature Tokenizer + pre-norm Transformer (Gorishniy et al. 2021) with ReGLU FFN |
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

Generate LDP attack detection training data. Output is a **directory** of `.npy` files with pre-normalized (z-score) features.

```bash
# Generate with defaults
python generate_dataset.py -o output/my_dataset

# Custom configuration
python generate_dataset.py -o output/custom \
    --protocols OUE OLH HST_Server \
    --epsilons 0.5 1.0 2.0 \
    --datasets zipf emoji fire \
    --ratios 0.10 0.15 0.20 \
    --experiments 5 \
    --workers 4
```

### Output Directory Format

| File | Description |
|------|-------------|
| `features.npy` | `(N, 16)` float32 — z-score normalized feature matrix |
| `labels.npy` | `(N,)` float32 — binary labels (0 = benign, 1 = attacker) |
| `config.npy` | `(N, 9)` object — per-row config: `[target_set_size, attacker_ratio, protocol, splits, epsilon, dataset_type, n, experiment_id, row_in_experiment]` |
| `norm_stats.json` | Feature means, stds, and names used for normalization |
| `metadata.npz` | Per-experiment metadata (π̂, π_true, p, q, support tensors), keyed by `experiment_id` |

`n`, `experiment_id`, and `row_in_experiment` exist purely for post-training analysis and for indexing
back into `metadata.npz` — never fed to the model (`main.py` builds `X_train`/`X_val`/`X_test` from
`features.npy` only). `experiment_id` matches the integer used in `metadata.npz`'s `graph_{id:06d}_*` keys
(that on-disk naming is an unrelated storage detail); `row_in_experiment` is that row's position (0..n-1)
within its own experiment, letting you recover its raw support vector from `metadata.npz`.

### Dataset Generation CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output`, `-o` | Output directory path | *required* |
| `--protocols` | LDP protocols: `OUE`, `OLH`, `HST_User`, `HST_Server` | `['OUE', 'OLH']` |
| `--epsilons` | Privacy parameters | `[0.5, 0.7, 1.0, 1.5]` |
| `--datasets` | Dataset types: `zipf`, `emoji`, `fire` | all three |
| `--ratios` | Attacker ratios | `[0.10, 0.15, 0.20]` |
| `--target-sizes` | Target set sizes | `[2, 4, 6, 8]` |
| `--splits` | Split values | `[2, 4, 6, 8]` |
| `--experiments` | Experiments per config | 5 |
| `--full-scale` | Use full-scale dataset sizes | False |
| `--n` | Override number of users | None |
| `--workers` | Outer ProcessPoolExecutor workers (OUE/HST) | 4 |
| `--inner-processors` | Inner parallel processes per task | 4 |
| `--save-every` | Flush to disk every N completed tasks | 50 |

### Feature Set

The 16 per-user features follow the DiffStats methodology, extended so every baseline they're scored
against is derivable purely from public information — `(ε, protocol, domain)` alone — never from the
true un-perturbed distribution or a simulated "ideal" reference, since a deployed server never observes
either of those (that's what LDP protects). Two baselines are used, both purely analytic:

- **k-value baseline**: `binom.pmf(k, domain, p_binomial)`, where `p_binomial` is derived exactly the
  way `construct_omega` (`attacker_detector/data/generators/protocols.py`) derives it per protocol.
- **Item-level baseline**: every item is equally likely to be reported with probability `p_binomial`
  under complete ignorance of true item popularity — the only baseline computable without data that
  isn't available at inference time. This carries no true item-popularity signal by itself, but is
  still discriminative *per user*: all attackers in an experiment target the same small `target_set`,
  so their individual deviations from this shared baseline are correlated across attacker rows, while a
  benign user's deviation reflects only their own independently-drawn item.

An empirically-estimated baseline (computed from the attacked reports themselves) doesn't have this
property — at high attacker ratios it shifts toward the attack, and anomaly scores can invert (attackers
end up looking *less* anomalous than honest users). The analytic baseline can't be contaminated that way.

**HST caveat.** `HST_Server`/`HST_User`'s `one_list` (attacks.py) is a count of `+1`s in each user's
*public random sign vector* — unrelated to their actual report for `HST_Server`, and deliberately padded
by attackers to mimic the benign count for `HST_User`. The 9 features below derived from `one_list` /
the k-value baseline carry near-zero signal for HST specifically (measured AUC ≈ 0.50–0.57, vs. ≈ 0.6–0.9
for the same features under OUE) — a property of those two protocols' generators, not a bug in feature
extraction. Only the item-level features (`overlap_anomalous_items_ratio`, `max_item_freq_ratio`,
`mean_item_freq_ratio`, `support_entropy_scaled`, `user_theoretical_deviation`) retain any signal there.

### Domain-Invariant Scaling

Every feature is **dimensionless by construction**. Each scale-dependent quantity is divided by its
analytically known scale, derived from `(d, p, q)` alone — nothing is fitted from data. A feature
therefore means the same thing at any domain size, including one never seen during training.

| Feature | Scaling | Rationale |
|---------|---------|-----------|
| `num_ones_scaled` | ÷ `expected_ones` = `p + (d−1)q` | centres on 1.0 for any `d` |
| `one_deviation` | ÷ `σ_k` | unsigned magnitude counterpart to `num_ones_scaled`; see below |
| `k_discrepancy_scaled`, `k_observed_frequency_scaled`, `k_theoretical_frequency_scaled` | × `σ_k` | see below |
| `max_item_freq_ratio`, `mean_item_freq_ratio` | none needed | `item_frequency_ratio` is now `item_counts / (n · p_binomial)` — a ratio of two counts against an analytically-known scale, dimensionless at its source |
| `user_theoretical_deviation` | none needed | mean absolute deviation of `reported` from `p_binomial` over `domain` items — already bounded in [0, 1] |
| `support_entropy_scaled` | ÷ `log(d)` | a user reporting `k` items uniformly has entropy `log(k) ≤ log(d)`; bounds it to [0, 1] |
| `wasserstein_distance_scaled` | ÷ `σ_k` | see below |
| `freq_ratio`, `is_anomalous_k`, `overlap_anomalous_items_ratio`, `max_support_value`, `js_divergence_k` | none | already ratios, bounded, or indicator |

where `σ_k = sqrt(d · p_binomial · (1 − p_binomial))` is the standard deviation of the k-histogram.

**Why `σ_k` and not `d`.** The k-histogram is `Binomial(d, p_binomial)`, so its mass spreads over a
width of `σ_k ≈ √d` bins, not `d` bins. Per-bin probability shrinks as `1/√d`, and the typical
earth-mover distance from a point mass to that histogram is `O(σ_k)`, not `O(d)`. Dividing by `d`
overcorrects — measured across d = 296/1024/1496, `wasserstein ÷ d` left a 0.88× spread while
`÷ σ_k` leaves 0.03×.

`is_anomalous_k` flags a fixed *fraction* of distinct k-values (`TOP_DISCREPANT_FRACTION`, 10%)
rather than a fixed count. A fixed count would flag a shrinking share as `d` grows, since the number
of distinct k scales with `σ_k`.

**No simulated "ideal" reference is generated any more.** Earlier versions of this extractor sampled a
synthetic un-attacked population (`build_normal_lists_from_mechanism_stochastic`) purely to estimate
`item_probabilities` empirically. That sampling assigns each user's reported items *uniformly at random*
among the domain — so, in expectation, it was always estimating the same `p_binomial` this version
computes in closed form, just noisily and at the cost of an `O(n·domain)` simulation per experiment.
Computing it directly is strictly better: exact, free, and (unlike a simulated reference) something a
real server could reproduce from public parameters alone.

`std_item_freq_ratio` and `overlap_anomalous_items_count` are deliberately excluded — the former still
drifts across domains even after scaling, the latter's ratio form carries the same signal, scale-free.
`theoretical_probability_k` is also dropped — identical to `k_theoretical_frequency` by construction.

Features that depend only on the user's `k` are computed once per distinct `k` and indexed per user,
rather than recomputed for every user in the experiment.

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




## Training (`main.py`)

Train and evaluate tabular attacker detection models on the NPY dataset directory (output of `generate_dataset.py`). Supports k-fold cross-validation, early stopping, and cross-dataset generalization.

```bash
# Train MLP with k-fold CV + final training
python main.py -d output/my_dataset -m mlp --epochs 20 --k-folds 5 --patience 10 -o results/

# CV only (no final training)
python main.py -d output/my_dataset -m mlp --k-folds 5 --cv-only

# Standard training (no CV, no early stopping)
python main.py -d output/my_dataset -m mlp --epochs 10 --k-folds 0 --val-size 0

# Cross-dataset: train on zipf, test on emoji, with CV (HP grid search runs by default)
python main.py -d output/my_dataset -m mlp \
    --training-method cross --train-dataset zipf --test-dataset emoji --k-folds 5

# Cross-dataset with combined training types: train on zipf+emoji, test on fire
python main.py -d output/my_dataset -m mlp \
    --training-method cross --train-dataset zipf emoji --test-dataset fire --k-folds 5

# Three-way: train on zipf, test on emoji, evaluate on fire
python main.py -d output/my_dataset -m mlp \
    --training-method three-way \
    --train-dataset zipf --test-dataset emoji --eval-dataset fire

# Skip HP grid search, use --lr/--dropout directly for CV + final training
python main.py -d output/my_dataset -m mlp --k-folds 5 --no-hp-search
```

### Training Pipeline

1. **Load NPY dataset** from directory (features are pre-normalized)
2. **Split data** into train / val / test (stratified)
3. **K-fold CV** on train+val, with HP grid search by default (`--k-folds`, `--no-hp-search`)
4. **Final training** with early stopping on val F1 (`--patience`), using the best HP config found by CV
5. **Test evaluation** + sensitivity analysis with config metadata

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-path`, `-d` | Path to NPY dataset directory | *required* |
| `--model`, `-m` | Model type: `mlp`, `gan`, `attention` | *required* |
| `--epochs`, `-e` | Max training epochs | 5 |
| `--batch-size`, `-b` | Batch size | 256 |
| `--lr` | Learning rate (fallback when HP search is skipped, or for values not in the grid) | 0.001 |
| `--dropout` | Dropout rate (fallback when HP search is skipped, or for values not in the grid) | 0.2 |
| `--hidden-sizes` | MLP hidden layer widths, comma-separated, e.g. `--hidden-sizes 64,32,16` | pyramid rule (see below) |
| `--test-size` | Test split ratio (only for `none` mode) | 0.2 |
| `--val-size` | Validation fraction (carved from train) | 0.15 |
| `--k-folds` | K-fold CV folds (0 to skip CV entirely) | 5 |
| `--patience` | Early stopping patience (epochs) | 10 |
| `--cv-only` | Run only k-fold CV, skip final training | False |
| `--no-hp-search` | Skip HP grid search; use `--lr`/`--dropout`/`--pos-weight` directly for CV and final training | False |
| `--hp-lr` | Learning rate values to grid-search, e.g. `--hp-lr 0.0005 0.001 0.003` | `config.DEFAULT_TABULAR_HP_GRID['lr']` |
| `--hp-dropout` | Dropout values to grid-search, e.g. `--hp-dropout 0.1 0.2 0.3` | `config.DEFAULT_TABULAR_HP_GRID['dropout']` |
| `--pos-weight` | Positive class weight for BCE loss. `'auto'` computes `neg_count / pos_count` from training data; or provide a float | `auto` |
| `--hp-pos-weight` | pos_weight values to grid-search, e.g. `--hp-pos-weight auto 1.0 5.77` (each is `'auto'` or a float). Not searched unless given | None (not searched) |
| `--hp-hidden-sizes` | MLP shapes to grid-search, e.g. `--hp-hidden-sizes 256,128,64 64,32,16`, or `default` for `config.DEFAULT_HIDDEN_SIZE_GRID`. `mlp` only | None (not searched) |
| `--seed` | Random seed | 42 |
| `--output-dir`, `-o` | Save model/plots/results here | None |
| `--no-plot` | Skip sensitivity plots | False |

### Hyperparameter Grid Search

When `--k-folds > 0` (the default), `main.py` runs a **grid search over `--hp-lr` × `--hp-dropout`** via k-fold CV before final training — mirroring the HP search already used in `main_gnn.py`. Each combination is evaluated with `n_folds`-fold CV; the config with the highest mean validation F1 is used for the final training run. Pass `--no-hp-search` to skip this and train directly with `--lr`/`--dropout`/`--pos-weight`.

`--pos-weight` is **not searched by default** — pass `--hp-pos-weight` explicitly (e.g. `--hp-pos-weight auto 1.0 3.0`) to add it as a third grid axis; it multiplies the total number of configs by however many values you give. `'auto'` recomputes `neg_count / pos_count` fresh per fold (so it can vary slightly fold-to-fold), while a fixed float pins the same weight everywhere. Note: `Trainer`'s previous default silently used a fixed `pos_weight = 1.0` regardless of actual class imbalance; the default is now `'auto'`, which for a typical ~85/15 benign/attacker split resolves to roughly 5.7 instead of 1.0 — expect final-training behavior to shift accordingly unless you pass `--pos-weight 1.0` to match the old behavior.

### MLP Layer Sizing

`RobustAttackerDetector` takes a `hidden_sizes` list instead of a fixed `256 → 128 → 64` stack. Each
entry becomes a `Linear → BatchNorm → LeakyReLU(0.1) → Dropout` block, with the last block using half
the dropout rate. Weights use He/Kaiming init matched to the LeakyReLU slope.

The default applies the **geometric pyramid rule** (Masters, 1993): seed the first hidden layer near
the input width, then halve on each subsequent layer. For a 17-feature input this gives
`[64, 32, 16]` (~4K parameters) versus the previous `[256, 128, 64]` (~47K).

```python
pyramid_hidden_sizes(input_dim=17)  # -> [64, 32, 16]
```

Sizing heuristics do not transfer reliably between problems, so treat the default as a starting point
and let CV decide — pass `--hp-hidden-sizes default` to search `config.DEFAULT_HIDDEN_SIZE_GRID`:

| `hidden_sizes` | Params | Rationale |
|----------------|--------|-----------|
| `[256, 128, 64]` | ~47K | previous hardcoded default |
| `[128, 64, 32]` | ~13K | moderate taper |
| `[64, 32, 16]` | ~4K | geometric pyramid rule |
| `[64, 64, 32, 16]` | ~8K | deeper, gentler taper |

One caveat worth knowing when interpreting results: MLPs are
[uniquely sensitive to uninformative features](https://arxiv.org/pdf/2207.08815) because their
rotational invariance prevents them from ignoring a feature the way a decision tree can. If a shape
search produces little separation between configs, suspect the feature set before adding capacity.

Grid search is **not yet applied to `--training-method cross`/`three-way`'s val/test splits** beyond CV itself — the selected config is still used for the actual cross-dataset final training and evaluation, so `--train-dataset zipf emoji --test-dataset fire --k-folds 5` grid-searches over the combined zipf+emoji train+val portion, then trains/evaluates the winning config on the real train→test split.

### Generalizability Training

| Argument | Description | Default |
|----------|-------------|---------|
| `--training-method` | `none`, `cross`, or `three-way` | `none` |
| `--train-dataset` | One or more dataset types for training, e.g. `zipf` or `zipf emoji` | None |
| `--test-dataset` | Dataset type for testing | None |
| `--eval-dataset` | Dataset type for evaluation (three-way only) | None |

**Training methods:**

- **`none`** — Conventional training. All dataset types are mixed together and split randomly into train/val/test.
- **`cross`** — Cross-dataset generalizability. Train on one *or more* dataset types (combined), test on another. Val is carved from train.
- **`three-way`** — Full generalizability evaluation. Train on one type (or combination), test on a second, evaluate on a third.

### Output Files

| File | Contents |
|------|----------|
| `model.pt` | Model checkpoint |
| `hp_search_results.csv` | Per-config mean/std CV metrics from HP grid search (omitted with `--no-hp-search`) |
| `cv_results.csv` | Per-fold metrics from k-fold CV (best config, if HP search ran) |
| `training_history.csv` | Per-epoch train loss + val F1/accuracy |
| `sensitivity_test_results.csv` | Sensitivity by ε, ratio, target size, dataset type |
| `sensitivity_eval_results.csv` | Sensitivity for eval set (three-way only) |
| `sensitivity_*_*.png` | Sensitivity plots |

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
