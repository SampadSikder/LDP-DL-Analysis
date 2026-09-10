import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from attacker_detector.data.graph_dataset import GraphDatasetLoader

def main():
    data_path = os.path.join(parent_dir, 'knn_dataset.pt')
    
    print(f"Loading dataset from {data_path}...")
    loader = GraphDatasetLoader(data_path)
    graphs = loader.graphs
    print(f"Loaded {len(graphs)} graphs.")

    sample_interval = max(1, len(graphs) // 100)
    sampled_graphs = graphs[::sample_interval]
    print(f"Sampled {len(sampled_graphs)} graphs for plotting.")

    all_deviations = []
    all_neg_log_p = []
    all_labels = []

    for idx, g in enumerate(sampled_graphs):
        x = g.x.numpy()
        y = g.y.numpy()
        
        # In knn_dataset.pt, deviations_feat is at index 40, neg_log_p is at index 41
        deviations = x[:, 40]
        neg_log_p = x[:, 41]
        
        all_deviations.append(deviations)
        all_neg_log_p.append(neg_log_p)
        all_labels.append(y)

    all_deviations = np.concatenate(all_deviations)
    all_neg_log_p = np.concatenate(all_neg_log_p)
    all_labels = np.concatenate(all_labels)

    print(f"Total nodes extracted: {len(all_labels):,}")
    print(f"Benign nodes: {np.sum(all_labels == 0):,}")
    print(f"Attacker nodes: {np.sum(all_labels == 1):,}")

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 18
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {0: '#2b5c8f', 1: '#d95f02'}
    labels_map = {0: 'Benign (Label 0)', 1: 'Attacker (Label 1)'}

    print("Plotting Absolute Deviation Feature...")
    for label in [0, 1]:
        mask = (all_labels == label)
        sns.kdeplot(
            all_deviations[mask], 
            ax=axes[0], 
            label=labels_map[label], 
            color=colors[label], 
            fill=True, 
            alpha=0.4, 
            linewidth=2
        )
    axes[0].set_title('Distribution of Deviations Feature')
    axes[0].set_xlabel('Deviations Feature')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    print("Plotting Negative Log Probability Feature...")
    for label in [0, 1]:
        mask = (all_labels == label)
        sns.kdeplot(
            all_neg_log_p[mask], 
            ax=axes[1], 
            label=labels_map[label], 
            color=colors[label], 
            fill=True, 
            alpha=0.4, 
            linewidth=2
        )
    axes[1].set_title('Distribution of neg_log_p Feature')
    axes[1].set_xlabel('neg_log_p')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.suptitle('Comparison of Deviation Features: Benign vs Attacker (knn_dataset)', y=0.98)
    plt.tight_layout()

    output_filename = 'deviation_knn_distribution.png'
    workspace_output = os.path.join(parent_dir, 'scripts', output_filename)

    plt.savefig(workspace_output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to workspace: {workspace_output}")

    plt.close()
    print("Plotting complete!")

if __name__ == '__main__':
    main()
