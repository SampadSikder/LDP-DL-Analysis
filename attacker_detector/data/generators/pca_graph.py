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
            support_list, _, estimate_dist, _ = build_support_list_1_OLH(
                domain, Y, n, User_Seed, ratio, g, target_set,
                p, splits, h_ao, epsilon, processor=processors
            )
        else:
            support_list, _, estimate_dist, _ = build_support_list_1_OLH_Server(
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
        support_list, _, estimate_dist, _ = build_support_list_1_OUE(Y_data, n, epsilon)

    elif protocol == "HST_User":
        support_list, _, estimate_dist, _ = HST_Users(
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
        support_list, _, estimate_dist, _ = HST_Server(
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

    return support_list, labels, REAL_DIST, estimate_dist


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

    # After removing self
    knn_indices = np.array(filtered_indices)
    knn_distances = np.array(filtered_distances)

    # Construct symmetric edge list
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


def compute_utility_metrics(
    support_list: np.ndarray,
    labels: np.ndarray,
    real_dist: np.ndarray,
    estimate_dist_all: np.ndarray,
    epsilon: float,
    protocol: str,
) -> dict:
    """
    Compute JS divergence and Wasserstein distance between estimated distributions
    and the ground truth distribution Attacker + Benign and then seperate benign.
    """
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import wasserstein_distance

    n = len(labels)
    domain = len(real_dist)
    num_benign = int((labels == 0).sum())

    benign_mask = (labels == 0)
    obs_counts_benign = np.sum(support_list[benign_mask], axis=0)

    base_proto = 'OLH' if protocol in ('OLH', 'OLH_User', 'OLH_Server') else protocol

    if base_proto == 'OUE':
        p_OUE = 0.5
        q_OUE = 1.0 / (math.exp(epsilon) + 1.0)
        estimate_benign = (obs_counts_benign - num_benign * q_OUE) / max(p_OUE - q_OUE, 1e-12)
    elif base_proto == 'OLH':
        g = int(round(math.exp(epsilon))) + 1
        p_olh = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        a = 1.0 * g / (p_olh * g - 1)
        b_benign = 1.0 * num_benign / (p_olh * g - 1)
        estimate_benign = a * obs_counts_benign - b_benign
    else:
        estimate_benign = obs_counts_benign

    # Construct dist
    def _to_prob(arr):
        clipped = np.maximum(arr, 0.0)
        s = np.sum(clipped)
        if s > 0:
            return clipped / s
        return np.full(len(arr), 1.0 / len(arr))

    p_real = _to_prob(real_dist)
    p_all = _to_prob(estimate_dist_all) if estimate_dist_all is not None else p_real
    p_benign = _to_prob(estimate_benign)

    domain_idx = np.arange(domain)

    js_all = float(jensenshannon(p_all, p_real)) if estimate_dist_all is not None else 0.0
    js_benign = float(jensenshannon(p_benign, p_real))
    ws_all = float(wasserstein_distance(domain_idx, domain_idx, p_all, p_real)) if estimate_dist_all is not None else 0.0
    ws_benign = float(wasserstein_distance(domain_idx, domain_idx, p_benign, p_real))

    return {
        'utility_js_all': js_all,
        'utility_js_benign': js_benign,
        'utility_ws_all': ws_all,
        'utility_ws_benign': ws_benign,
    }


def build_graph_data(
    support_list: np.ndarray,
    labels: np.ndarray,
    epsilon: float,
    pca_dim: int = 16,
    knn_k: int = 10,
    metadata: dict = None,
    real_dist: np.ndarray = None,
    estimate_dist_all: np.ndarray = None,
) -> Data:
    if metadata is None:
        metadata = {}

    pca_features, explained_variance = apply_pca_fixed(support_list, n_components=pca_dim)
    edge_index, knn_distances, knn_indices = build_knn_graph(pca_features, k=knn_k)

    n = len(labels)
    density_feats = compute_density_features(knn_distances, knn_indices, n)
    influence_feats = compute_influence_features(edge_index, knn_indices, n)

    eps_feat = np.full((n, 1), float(epsilon))

    x_numpy = np.hstack([pca_features, density_feats, influence_feats, eps_feat])

    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    utility_metrics = {'utility_js_all': 0.0, 'utility_js_benign': 0.0, 'utility_ws_all': 0.0, 'utility_ws_benign': 0.0}
    if real_dist is not None:
        protocol_name = metadata.get('protocol', 'OUE')
        utility_metrics = compute_utility_metrics(
            support_list=support_list,
            labels=labels,
            real_dist=real_dist,
            estimate_dist_all=estimate_dist_all,
            epsilon=epsilon,
            protocol=protocol_name,
        )

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
        pca_variance_explained=explained_variance,
        utility_js_all=float(utility_metrics['utility_js_all']),
        utility_js_benign=float(utility_metrics['utility_js_benign']),
        utility_ws_all=float(utility_metrics['utility_ws_all']),
        utility_ws_benign=float(utility_metrics['utility_ws_benign']),
    )

    return data
