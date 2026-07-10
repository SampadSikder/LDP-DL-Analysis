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
    data_path = os.path.join(parent_dir, 'dataset_pca_v3.pt')
    
    print(f"Loading dataset from {data_path}...")
    loader = GraphDatasetLoader(data_path)
    graphs = loader.graphs
    print(f"Loaded {len(graphs)} graphs.")

    sample_interval = max(1, len(graphs) // 100)
    sampled_graphs = graphs[::sample_interval]
    print(f"Sampleed {len(sampled_graphs)} graphs for plotting.")

    all_avg_knn = []
    all_rel_density = []
    all_labels = []

    for idx, g in enumerate(sampled_graphs):
        x = g.x.numpy()
        y = g.y.numpy()
        
        input_dim = x.shape[1]
        pca_dim = input_dim - 8
        
        avg_knn_idx = pca_dim
        rel_density_idx = pca_dim + 2
        
        avg_knn = x[:, avg_knn_idx]
        rel_density = x[:, rel_density_idx]
        
        all_avg_knn.append(avg_knn)
        all_rel_density.append(rel_density)
        all_labels.append(y)

    all_avg_knn = np.concatenate(all_avg_knn)
    all_rel_density = np.concatenate(all_rel_density)
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

    print("Plotting Average kNN Distance...")
    for label in [0, 1]:
        mask = (all_labels == label)
        sns.kdeplot(
            all_avg_knn[mask], 
            ax=axes[0], 
            label=labels_map[label], 
            color=colors[label], 
            fill=True, 
            alpha=0.4, 
            linewidth=2
        )
    axes[0].set_title('Distribution of Average kNN Distance')
    axes[0].set_xlabel('Average kNN Distance')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    print("Plotting Relative Density...")
    for label in [0, 1]:
        mask = (all_labels == label)
        sns.kdeplot(
            all_rel_density[mask], 
            ax=axes[1], 
            label=labels_map[label], 
            color=colors[label], 
            fill=True, 
            alpha=0.4, 
            linewidth=2
        )
    axes[1].set_title('Distribution of Relative Density')
    axes[1].set_xlabel('Relative Density')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.suptitle('Comparison of Deviation/Density Features: Benign vs Attacker', y=0.98)
    plt.tight_layout()

    output_filename = 'deviation_features_distribution.png'
    workspace_output = os.path.join(parent_dir, 'scripts', output_filename)
    artifact_dir = '/home/sampad/.gemini/antigravity-ide/brain/a8a4cd0c-cba3-4ad7-b9d5-8c2d06f6e435'
    artifact_output = os.path.join(artifact_dir, output_filename)

    plt.savefig(workspace_output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to workspace: {workspace_output}")

    if os.path.exists(artifact_dir):
        plt.savefig(artifact_output, dpi=150, bbox_inches='tight')
        print(f"Saved plot to artifacts: {artifact_output}")

    plt.close()
    print("Plotting complete!")

if __name__ == '__main__':
    main()
