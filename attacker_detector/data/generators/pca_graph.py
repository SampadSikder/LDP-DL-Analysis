"""PCA and Graph features generator for LDP Attacker Detection GAT dataset."""

import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data

from .distributions import (
    generate_zipf_dist,
    generate_emoji_dist,
    generate_fire_dist,
)
from .attacks import (
    perturb_OUE_multi,
    HST_Server,
    HST_Users,
    build_support_list_1_OUE,
    build_support_list_1_OLH,
    build_support_list_1_OLH_Server,
)


def get_distribution_generator(dataset_type: str):
    generators = {
        'zipf': generate_zipf_dist,
        'emoji': generate_emoji_dist,
        'fire': generate_fire_dist,
    }
    return generators[dataset_type]


def generate_perturbed_data(
    epsilon: float,
    domain: int,
    n: int,
    protocol: str,
    ratio: float,
    target_set_size: int,
    splits: int,
    dataset_type: str = 'zipf',
    h_ao: int = 1,
    seed: int = None,
    processors: int = 4,
    olh_setting: str = 'server'
) -> tuple:
    if seed is not None:
        np.random.seed(seed)

    generator = get_distribution_generator(dataset_type)
    X, REAL_DIST = generator(n, domain, seed=seed)

    target_set_size = min(target_set_size, domain)
    target_set = set(np.random.choice(domain, size=target_set_size, replace=False))

    if protocol == "OLH":
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        User_Seed = np.arange(n)
        Y = np.zeros(n)

        if olh_setting == 'user':
            support_list, _, _, _ = build_support_list_1_OLH(
                domain, Y, n, User_Seed, ratio, g, target_set,
                p, splits, h_ao, epsilon, processor=processors
            )
        else:
            support_list, _, _, _ = build_support_list_1_OLH_Server(
                domain, Y, n, User_Seed, ratio, g, target_set,
                p, splits, h_ao, epsilon, processor=processors
            )

    elif protocol == "OUE":
        Y_data = perturb_OUE_multi(
            X=X,
            epsilon=epsilon,
            domain=domain,
            n=n,
            target_set=target_set,
            ratio=ratio,
            h_ao=h_ao,
            splits=splits,
            num_processes=processors
        )
        support_list, _, _, _ = build_support_list_1_OUE(Y_data, n, epsilon)

    elif protocol == "HST_User":
        support_list, _, _, _ = HST_Users(
            X=X,
            ratio=ratio,
            domain=domain,
            epsilon=epsilon,
            n=n,
            target_set=target_set,
            h_ao=h_ao,
            splits=splits
        )

    elif protocol == "HST_Server":
        support_list, _, _, _ = HST_Server(
            X=X,
            ratio=ratio,
            domain=domain,
            epsilon=epsilon,
            n=n,
            target_set=target_set,
            splits=splits
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    num_benign = int(n * (1 - ratio))
    labels = np.zeros(n)
    labels[num_benign:] = 1

    return support_list, labels


def apply_pca_fixed(support_list: np.ndarray, n_components: int = 16) -> tuple:
    # Returns PCA with default 16 dimensions
    n_samples, n_features = support_list.shape
    max_components = min(n_samples, n_features)
    k = min(n_components, max_components)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(support_list.astype(np.float64))

    pca = PCA(n_components=k)
    pca_features = pca.fit_transform(X_scaled)

    explained_variance = float(np.sum(pca.explained_variance_ratio_))

    if k < n_components:
        padding = np.zeros((n_samples, n_components - k))
        pca_features = np.hstack([pca_features, padding])

    return pca_features, explained_variance


def build_knn_graph(pca_features: np.ndarray, k: int = 10) -> tuple:
    """
    Default 10
    Returns:
        edge_index: (2, E) COO array (symmetric, no self-loops)
        knn_distances: (n, k)
        knn_indices: (n, k)
    """
    n = pca_features.shape[0]
    actual_k = min(k, n - 1)

    nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='auto').fit(pca_features) # Default metric L2 distance
    distances, indices = nbrs.kneighbors(pca_features)

    # Filter out self-loops for each node
    filtered_indices = []
    filtered_distances = []
    for i in range(n):
        idx = indices[i]
        dist = distances[i]
        mask = idx != i
        if np.sum(mask) == len(idx):
            idx_filtered = idx[:actual_k]
            dist_filtered = dist[:actual_k]
        else:
            idx_filtered = idx[mask][:actual_k]
            dist_filtered = dist[mask][:actual_k]
        filtered_indices.append(idx_filtered)
        filtered_distances.append(dist_filtered)

    knn_indices = np.array(filtered_indices)
    knn_distances = np.array(filtered_distances)

    # Construct symmetric edge list (undirected, no self-loops)
    edges = set()
    for i in range(n):
        for j in knn_indices[i]:
            if i != j:
                edges.add((i, int(j)))
                edges.add((int(j), i))

    if len(edges) > 0:
        edge_index = np.array(list(edges)).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    if actual_k < k:
        #Padding
        pad_size = k - actual_k
        knn_indices = np.pad(knn_indices, ((0, 0), (0, pad_size)), mode='edge')
        knn_distances = np.pad(knn_distances, ((0, 0), (0, pad_size)), mode='edge')

    return edge_index, knn_distances, knn_indices


def compute_density_features(knn_distances: np.ndarray, knn_indices: np.ndarray, n: int) -> np.ndarray:

    avg_knn_distance = np.mean(knn_distances, axis=1)
    local_density = 1.0 / (avg_knn_distance + 1e-5)

    neighbor_densities = local_density[knn_indices]
    mean_neighbor_density = np.mean(neighbor_densities, axis=1)
    relative_density = local_density / (mean_neighbor_density + 1e-5)

    knn_distance_std = np.std(knn_distances, axis=1)

    return np.column_stack([
        avg_knn_distance,
        local_density,
        relative_density,
        knn_distance_std
    ])


def compute_influence_features(edge_index: np.ndarray, knn_indices: np.ndarray, n: int) -> np.ndarray:
    in_degree = np.zeros(n)
    unique, counts = np.unique(knn_indices, return_counts=True)
    for u, c in zip(unique, counts):
        if 0 <= u < n:
            in_degree[int(u)] = c

    degree_centrality = in_degree / (n - 1) if n > 1 else np.zeros(n)

    hub_score = np.zeros(n)
    directed_edges = set()
    for i in range(n):
        for j in knn_indices[i]:
            if i != j:
                directed_edges.add((i, int(j)))

    for (i, j) in directed_edges:
        if (j, i) in directed_edges:
            hub_score[i] += 1

    return np.column_stack([
        in_degree,
        degree_centrality,
        hub_score
    ])


def build_graph_data(
    support_list: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    pca_dim: int = 16,
    knn_k: int = 10,
    metadata: dict = None
) -> Data:
    if metadata is None:
        metadata = {}

    pca_features, explained_variance = apply_pca_fixed(support_list, n_components=pca_dim)
    edge_index, knn_distances, knn_indices = build_knn_graph(pca_features, k=knn_k)

    n = len(labels)
    density_feats = compute_density_features(knn_distances, knn_indices, n)
    influence_feats = compute_influence_features(edge_index, knn_indices, n)

    eps_feat = np.full((n, 1), float(epsilon))

    # Concat all node features: pca_features (pca_dim), density (4), influence (3), epsilon (1)
    x_numpy = np.hstack([pca_features, density_feats, influence_feats, eps_feat])

    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    # Final feature vector
    data = Data(
        x=x_tensor,
        edge_index=edge_index_tensor,
        y=y_tensor,
        epsilon=float(epsilon),
        protocol=metadata.get('protocol', ''),
        dataset_type=metadata.get('dataset_type', ''),
        ratio=float(metadata.get('ratio', 0.0)),
        target_set_size=int(metadata.get('target_set_size', 0)),
        splits=int(metadata.get('splits', 0)),
        pca_variance_explained=explained_variance
    )

    return data
