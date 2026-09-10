import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from attacker_detector.data.graph_dataset import GraphDatasetLoader
data_path = os.path.join(parent_dir, 'dataset_pca_v3.pt')

loader = GraphDatasetLoader(data_path)
for idx, graph in enumerate(loader.graphs):
    print(f"Graph {idx} Explained Variance: {graph.pca_variance_explained:.4f}")