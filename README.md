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
# Create conda environment
conda env create -f environment.yml
conda activate attacker-detector
```

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
| `--test-size` | Test split ratio | 0.2 |
| `--seed` | Random seed | 42 |
| `--output-dir`, `-o` | Save model/plots here | None |
| `--no-plot` | Skip sensitivity plots | False |

## License

MIT
