import os
import sys
import torch

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    test_file = os.path.join(parent_dir, 'scripts', 'test_dataset_pca_new.pt')
    if not os.path.exists(test_file):
        print(f"ERROR: {test_file} does not exist. Please generate the test dataset first:")
        print("python generate_dataset_pca.py --output scripts/test_dataset_pca_new.pt --protocols OUE --epsilons 0.5 --experiments 1 --datasets zipf --ratios 0.1")
        sys.exit(1)

    print(f"Loading generated test dataset from: {test_file}")
    graphs = torch.load(test_file, weights_only=False)
    if not isinstance(graphs, list) or len(graphs) == 0:
        print("ERROR: Loaded data is not a non-empty list of graphs.")
        sys.exit(1)

    graph = graphs[0]
    print(f"Successfully loaded graph.")
    print(f"  Dataset Type: {getattr(graph, 'dataset_type', 'N/A')}")
    print(f"  Protocol:     {getattr(graph, 'protocol', 'N/A')}")
    print(f"  Epsilon:      {getattr(graph, 'epsilon', 'N/A')}")
    print(f"  Ratio:        {getattr(graph, 'ratio', 'N/A')}")
    print(f"  Feature shape: {graph.x.shape}")
    print(f"  Label shape:   {graph.y.shape}")
    print(f"  PCA Variance Explained: {getattr(graph, 'pca_variance_explained', 'N/A'):.6f}")

    expected_dim = 44
    if graph.x.shape[1] == expected_dim:
        print(f"SUCCESS: Feature dimensions match expected size of {expected_dim}!")
    else:
        print(f"FAILURE: Feature dimension is {graph.x.shape[1]}, but expected {expected_dim}.")
        sys.exit(1)

if __name__ == '__main__':
    main()
