"""
Analyze clipping ratio of neg_log_p values in knn_dataset.pt.

In pca_graph.py the raw computation is:
    log_p = chi2.logsf(chisq, df=1)
    neg_log_p = -log_p
    neg_log_p = np.nan_to_num(neg_log_p, nan=1000.0, posinf=1000.0, neginf=0.0)

Values that were posinf or nan are clipped to 1000.0.
Since the dataset stores the already-clipped values, we identify clipped entries as
those >= 1000.0 (the clip sentinel).
"""

import os
import sys
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from attacker_detector.data.graph_dataset import GraphDatasetLoader

CLIP_VALUE = 1000.0
NEG_LOG_P_IDX = 41   # column index in x feature matrix


def analyze(data_path: str, sample_every: int = 1):
    print(f"Loading dataset from {data_path}...")
    loader = GraphDatasetLoader(data_path)
    graphs = loader.graphs
    print(f"Loaded {len(graphs)} graphs.")

    sampled = graphs[::sample_every]
    print(f"Sampling every {sample_every} graph → {len(sampled)} graphs analysed.\n")

    all_neg_log_p = []
    all_labels = []

    for g in sampled:
        all_neg_log_p.append(g.x[:, NEG_LOG_P_IDX].numpy())
        all_labels.append(g.y.numpy())

    neg_log_p = np.concatenate(all_neg_log_p)
    labels    = np.concatenate(all_labels)

    # Clipped = values that hit the 1000.0 sentinel (posinf / nan before nan_to_num)
    clipped_mask = neg_log_p >= CLIP_VALUE
    finite_mask  = ~clipped_mask

    n_total   = len(neg_log_p)
    n_clipped = clipped_mask.sum()
    n_finite  = finite_mask.sum()

    # Per-class breakdown
    benign_mask   = (labels == 0)
    attacker_mask = (labels == 1)

    n_clipped_benign   = (clipped_mask & benign_mask).sum()
    n_clipped_attacker = (clipped_mask & attacker_mask).sum()

    n_benign   = benign_mask.sum()
    n_attacker = attacker_mask.sum()

    print("=" * 60)
    print("neg_log_p Clipping Analysis")
    print("=" * 60)
    print(f"Total users analysed : {n_total:>12,}")
    print(f"  Benign             : {n_benign:>12,}")
    print(f"  Attacker           : {n_attacker:>12,}")
    print()
    print(f"Clipped to {CLIP_VALUE:.0f} (posinf/nan→sentinel):")
    print(f"  Overall : {n_clipped:>10,} / {n_total:,}  "
          f"({100*n_clipped/n_total:.2f}%)")
    print(f"  Benign  : {n_clipped_benign:>10,} / {n_benign:,}  "
          f"({100*n_clipped_benign/n_benign:.2f}%)")
    print(f"  Attacker: {n_clipped_attacker:>10,} / {n_attacker:,}  "
          f"({100*n_clipped_attacker/n_attacker:.2f}%)")
    print()
    if n_finite > 0:
        print(f"Finite (non-clipped) neg_log_p range:")
        print(f"  min : {neg_log_p[finite_mask].min():.4f}")
        print(f"  max : {neg_log_p[finite_mask].max():.4f}")
        print(f"  mean: {neg_log_p[finite_mask].mean():.4f}")
        print(f"  std : {neg_log_p[finite_mask].std():.4f}")
        print()
        print(f"Finite neg_log_p — per class:")
        for label_val, name in [(0, 'Benign'), (1, 'Attacker')]:
            mask = finite_mask & (labels == label_val)
            if mask.sum() > 0:
                vals = neg_log_p[mask]
                print(f"  {name}: min={vals.min():.4f}, max={vals.max():.4f}, "
                      f"mean={vals.mean():.4f}, std={vals.std():.4f}")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze neg_log_p clipping ratio in dataset')
    parser.add_argument('--data-path', default='knn_dataset.pt',
                        help='Path to .pt dataset file (default: knn_dataset.pt)')
    parser.add_argument('--sample-every', type=int, default=50,
                        help='Sample every N-th graph (default: 50, use 1 for full scan)')
    args = parser.parse_args()

    analyze(args.data_path, sample_every=args.sample_every)
