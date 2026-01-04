"""Attack simulation functions for OUE and OLH protocols."""

import math
import random
from functools import partial
from multiprocessing import Pool
from typing import Set, Tuple

import numpy as np
import xxhash
from scipy import stats
from tqdm import tqdm

from .protocols import construct_omega, _olh_params


# Global worker state for multiprocessing
_worker_X = None
_worker_domain = None
_worker_q_OUE = None


def _init_worker(X, domain, q_OUE):
    """Initialize worker process with shared data."""
    global _worker_X, _worker_domain, _worker_q_OUE
    _worker_X = X
    _worker_domain = domain
    _worker_q_OUE = q_OUE


def _perturb_oue_process(args):
    """Worker function for OUE perturbation."""
    start, end, ratio, target_set, h_ao, splits, average_1_num_list = args
    
    n = _worker_X.shape[0]
    domain = _worker_domain
    q_OUE = _worker_q_OUE
    
    local_user_data = np.zeros((end - start, domain), dtype=int)
    h_ao_local = int(h_ao * 10)

    for idx, i in enumerate(range(start, end)):
        v = int(_worker_X[i])

        if i < n * (1 - ratio):
            # Benign user
            random_flip = (np.random.rand(domain) < q_OUE).astype(int)
            local_user_data[idx, :] = random_flip
            local_user_data[idx, v] = 1 if np.random.rand() < 0.5 else 0
            continue

        # Attacker
        avg1 = int(average_1_num_list[i])
        if splits < avg1:
            splits_k = min(int(splits), len(target_set))
            if splits_k > 0:
                splits_list = random.sample(list(target_set), splits_k)
                local_user_data[idx, splits_list] = 1
            remaining_set = list(set(range(domain)) - set(splits_list if splits_k > 0 else []))
            diff = avg1 - len(splits_list if splits_k > 0 else [])
            diff_AO = random.randint(max(0, diff - h_ao_local), diff + h_ao_local) if diff > 0 else 0
            if diff_AO > 0 and len(remaining_set) >= diff_AO:
                random_numbers = random.sample(remaining_set, diff_AO)
                local_user_data[idx, random_numbers] = 1
        else:
            k = min(avg1, len(target_set))
            if k > 0:
                splits_list = random.sample(list(target_set), k)
                local_user_data[idx, splits_list] = 1
            remaining_set = list(set(range(domain)) - set(splits_list if k > 0 else []))
            diff = avg1 - len(splits_list if k > 0 else [])
            diff_AO = random.randint(max(0, diff - h_ao_local), diff + h_ao_local) if diff > 0 else 0
            if diff_AO > 0 and len(remaining_set) >= diff_AO:
                random_numbers = random.sample(remaining_set, diff_AO)
                local_user_data[idx, random_numbers] = 1

    return local_user_data


def perturb_OUE_multi(
    X: np.ndarray,
    epsilon: float,
    domain: int,
    n: int,
    target_set: Set[int],
    ratio: float,
    h_ao: int,
    splits: int,
    num_processes: int = 4
) -> np.ndarray:
    """
    Perturb data using OUE protocol with attack simulation.
    
    Args:
        X: User data array
        epsilon: Privacy parameter
        domain: Domain size
        n: Number of users
        target_set: Set of target items for attack
        ratio: Attacker ratio
        h_ao: Attack optimization parameter
        splits: Number of splits
        num_processes: Parallel processes
    
    Returns:
        Perturbed user data matrix
    """
    q_OUE = 1 / (math.exp(epsilon) + 1)
    
    # Prepare average_1_num_list
    omega_probs = construct_omega(epsilon, domain, 'OUE')
    if h_ao == 1:
        average_1_num_list = np.random.choice(np.arange(domain), size=n, p=omega_probs)
    else:
        average_1_num_list = np.full(n, int(0.5 + (domain - 1) * q_OUE), dtype=int)

    # Prepare ranges for parallel processing
    ranges = []
    for i in range(num_processes):
        start = (i * n) // num_processes
        end = ((i + 1) * n) // num_processes if i < num_processes - 1 else n
        if end <= start:
            continue
        ranges.append((start, end, ratio, target_set, h_ao, splits, average_1_num_list))

    # Create pool with initializer
    with Pool(processes=len(ranges), initializer=_init_worker, initargs=(X, domain, q_OUE)) as pool:
        results = pool.map(_perturb_oue_process, ranges)

    return np.vstack(results)


def build_support_list_1_OUE(
    estimates: np.ndarray,
    n: int,
    epsilon: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """
    Build support list from OUE perturbed data.
    
    Args:
        estimates: Perturbed user data matrix
        n: Number of users
        epsilon: Privacy parameter
    
    Returns:
        Tuple of (support, one_list, ESTIMATE_DIST, None)
    """
    q_OUE = 1 / (math.exp(epsilon) + 1)
    p = 0.5
    
    Results_support = np.array(estimates)
    Estimations = np.sum(Results_support, axis=0)
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = [(i - n * q_OUE) / (p - q_OUE) for i in Estimations]
    
    return Results_support, Results_support_one_list, np.array(Estimations), None


def _process_user_seeds(i, User_Seed, Y, domain, g):
    """Process user seed for OLH."""
    local_estimate = np.zeros(domain)
    user_seed = User_Seed[i]
    for v in range(domain):
        if Y[i] == (xxhash.xxh3_64(str(v), seed=int(user_seed)).intdigest() % g):
            local_estimate[v] += 1
    return local_estimate


def _calculate_prob_according_sample_size(num_samples, domain, g, h, target_set, splits):
    """Calculate probability distribution for OLH attack."""
    splits_list = random.sample(list(target_set), min(splits, len(target_set)))
    p = 1 / g
    
    mu = domain * p
    binom_dist = stats.binom(domain, p)
    lower_bound = max(0, mu - h)
    upper_bound = min(domain, mu + h)
    ratio = (binom_dist.cdf(upper_bound) - binom_dist.cdf(lower_bound - 1))
    ratio = ratio / (2 * h + 1) if h > 0 else ratio
    
    N_effective = num_samples * ratio

    K_min = 1
    K_max = len(splits_list)
    for K in range(K_max, K_min - 1, -1):
        prob = (p) ** K * N_effective
        if prob < 1:
            K_max = K
        else:
            break
    K_min = max(K_max, 1)

    K_values = np.arange(K_min, len(splits_list) + 1)
    K_probs = np.array([(p) ** K * N_effective for K in K_values])
    K_probs = K_probs / np.sum(K_probs)

    return K_values, K_probs


def _process_attacker_olh(i, n, ratio, target_set, g, domain, splits, h_ao, epsilon, K_values, K_probs):
    """Process single OLH attacker."""
    k = np.random.choice(K_values, p=K_probs)
    average_project_hash = int(domain / g)
    
    if splits < average_project_hash:
        splits_list = random.sample(list(target_set), min(splits, len(target_set)))
        num_map = average_project_hash
        
        if h_ao == 0:
            num_map_AO = random.randint(num_map - int(h_ao), num_map + int(h_ao))
        else:
            omega_probs = construct_omega(epsilon, domain, 'OLH_User')
            num_map_AO = np.random.choice(range(domain), p=omega_probs)
        
        non_target_ones = max(0, num_map_AO - k)
        
        target_indices = np.random.choice(list(splits_list), size=min(k, len(splits_list)), replace=False)
        non_target_indices = list(set(range(domain)) - set(splits_list))
        non_target_selected = np.random.choice(
            non_target_indices, 
            size=min(non_target_ones, len(non_target_indices)), 
            replace=False
        ) if non_target_ones > 0 and len(non_target_indices) > 0 else []
        
        vector = np.zeros(domain, dtype=int)
        vector[target_indices] = 1
        if len(non_target_selected) > 0:
            vector[non_target_selected] = 1
    else:
        vector = np.zeros(domain, dtype=int)

    index = int(n * (1 - ratio) + i)
    return index, vector


def build_support_list_1_OLH(
    domain: int,
    Y: np.ndarray,
    n: int,
    User_Seed: np.ndarray,
    ratio: float,
    g: int,
    target_set: Set[int],
    p: float,
    splits: int,
    h_ao: int,
    epsilon: float,
    processor: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """
    Build support list for OLH protocol with attack simulation.
    """
    num_samples = 1000000
    
    K_values, K_probs = _calculate_prob_according_sample_size(
        num_samples, domain, g, h_ao, target_set, splits
    )

    process_attacker_partial = partial(
        _process_attacker_olh,
        n=n,
        ratio=ratio,
        target_set=target_set,
        g=g,
        domain=domain,
        splits=splits,
        h_ao=10 * h_ao,
        epsilon=epsilon,
        K_values=K_values,
        K_probs=K_probs
    )

    num_attackers = int(round(n * ratio))

    with Pool(processes=processor) as pool:
        results = list(tqdm(
            pool.imap(process_attacker_partial, range(num_attackers)),
            total=num_attackers,
            desc='Processing OLH attackers'
        ))

    vector_matrix = np.zeros((num_attackers, domain))
    for i, (index, best_vector) in enumerate(results):
        vector_matrix[i, :] = best_vector

    # Process normal users
    process_partial = partial(
        _process_user_seeds,
        User_Seed=User_Seed,
        Y=Y,
        domain=domain,
        g=g
    )

    with Pool(processes=processor) as pool:
        estimates = pool.map(process_partial, range(n - num_attackers))

    estimates = np.array(estimates)
    estimates = np.vstack((estimates, vector_matrix))
    estimates = estimates.reshape(int(n), domain)
    
    Results_support = estimates
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = np.sum(Results_support, axis=0)
    
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    Estimations = a * Estimations - b
    
    return Results_support, Results_support_one_list, Estimations, None
