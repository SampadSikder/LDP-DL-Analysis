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

from .protocols import construct_omega

# Global worker state for multiprocessing
_worker_X = None
_worker_domain = None
_worker_q_OUE = None


def apa_quota(m: int, omega_probs: np.ndarray) -> np.ndarray:
    """How many of m fake users report each count k under optimal APA.

    omega[k] = floor(m * P(X = k)); the few users left over by flooring go to
    the k values with the largest fractional parts, so the quota sums to m.
    """
    expected = m * np.asarray(omega_probs, dtype=np.float64)
    counts = np.floor(expected).astype(np.int64)
    leftover = int(m - counts.sum())
    if leftover > 0:
        counts[np.argsort(-(expected - counts))[:leftover]] += 1
    return counts


def apa_counts(m: int, omega_probs: np.ndarray) -> np.ndarray:
    """Per-fake-user support counts for the optimal APA attack (paper Sec. 4.1.4).

    Exactly apa_quota(m, omega_probs)[k] fake users get count k, so the fake
    users' count histogram matches the genuine one as closely as integers
    allow. Returned in random order (seeded by the caller's np.random state).
    """
    counts = apa_quota(m, omega_probs)
    ks = np.repeat(np.arange(len(counts)), counts)
    np.random.shuffle(ks)
    return ks


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
    # +-10 jitter on the count only for the legacy h_ao=1 setting; MGA-A (0) and
    # exact APA (2) use each fake user's assigned count as is.
    h_ao_local = 10 if h_ao == 1 else 0

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
    Perturb data using OUE protocol with attack simulation (With parallel processing).
    
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
    if h_ao == 2:
        # exact APA: fake users' counts follow omega[k] = floor(m * P(X=k))
        is_fake = np.arange(n) >= n * (1 - ratio)
        average_1_num_list = np.zeros(n, dtype=int)
        average_1_num_list[is_fake] = apa_counts(int(is_fake.sum()), omega_probs)
    elif h_ao == 1:
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


num_samples = 1000000


def uniform_sampling_best_vector(target_set, g, d, m, num_samples):
    best_vector = None
    closest_ones_diff = float('inf')
    max_target_count = 0
    best_score = -float('inf')
    current_target = None
    current_diff = None

    for _ in range(num_samples):
        vector = np.random.binomial(1, 1 / g, size=d)

        # Count the number of 1's in the vector
        ones_count = np.sum(vector)
        ones_diff = abs(ones_count - m)

        target_count = sum(1 for item in target_set if vector[item % d] == 1)

        # Calculate the score: target_count - ones_diff
        score = target_count - ones_diff

        if score > best_score:
            best_score = score
            best_vector = vector
            current_target = target_count
            current_diff = ones_diff

    return best_vector, current_target, current_diff


def calculate_prob_according_sample_size(num_samples, d, g, h, target_set, splits):
    splits_list = random.sample(list(target_set), splits)
    target_set = splits_list
    user_vectors = []
    p = 1 / g

    mu = d * p
    sigma = np.sqrt(d * p * (1 - p))

    lower_bound = max(0, mu - h)
    upper_bound = min(d, mu + h)

    binom_dist = stats.binom(d, p)
    ratio = (binom_dist.cdf(upper_bound) - binom_dist.cdf(lower_bound - 1))
    ratio = ratio / (2 * h + 1)
    # ratio = 1

    N_effective = num_samples * ratio
    print('N_effective: ', N_effective)

    K_min = 1
    K_max = len(target_set)
    for K in range(K_max, K_min - 1, -1):
        prob = (p) ** K * N_effective
        if prob < 1:
            K_max = K
        else:
            break
    K_min = max(K_max, 1)

    K_values = np.arange(K_min, len(target_set) + 1)
    K_probs = []
    for K in K_values:
        prob = (p) ** K * N_effective
        K_probs.append(prob)
    K_probs = np.array(K_probs)

    K_probs = K_probs / np.sum(K_probs)

    return K_values, K_probs


def process_attacker(i, n, ratio, target_set, g, domain, splits, h_ao, e, K_values, K_probs):
    k = np.random.choice(K_values, p=K_probs)
    random.seed()
    averge_project_hash = int(domain / g)
    if splits < averge_project_hash:
        # Split the target set for each user
        splits_list = random.sample(list(target_set), splits)
        # Gap between average mapping
        num_map = averge_project_hash
        # Remaining set (unused in this snippet but kept for completeness)
        remaining_set = set(range(domain)) - set(target_set)
        #h_ao = 0
        if h_ao == 0:
          num_map_AO = random.randint(num_map - int(h_ao), num_map + int(h_ao))
        # num_map_AO num_map_AO = np.random.choice([i for i in range(domain)], construct_omega(e, domain, 'OLH_User'))
        else:
          omega_probs = construct_omega(e, domain, 'OLH_User')
          num_map_AO = np.random.choice(range(domain), p=omega_probs)
        non_target_ones = num_map_AO - k
        non_target_ones = max(0, min(non_target_ones, len(non_target_indices)))
        # Each attacker finds their optimal hash function
        '''best_vector, target_map, diff  = uniform_sampling_best_vector(
            splits_list, g, domain, num_map_AO, num_samples)'''
        k = min(k, len(splits_list))
        target_indices = np.random.choice(list(splits_list), size=k, replace=False)
        non_target_indices = list(set(range(domain)) - set(splits_list))
        non_target_selected = np.random.choice(non_target_indices, size=non_target_ones, replace=False)
        vector = np.zeros(domain, dtype=int)
        vector[target_indices] = 1
        vector[non_target_selected] = 1
    else:
        print('splits > averge_project_hash')
        exit(0)
    # Calculate the index in User_Seed to update
    index = int(n * (1 - ratio) + i)
    #print(f'attacker:{i}, target_map:{k}, diff:{num_map_AO - sum(vector)}, h_ao:{h_ao}, splits:{splits}')
    return index, vector


def process_user_seeds(i, User_Seed_noattack, Y_Nattack, domain, g):
    local_estimate = np.zeros(domain)
    user_seed = User_Seed_noattack[i]
    for v in range(domain):
        if Y_Nattack[i] == (xxhash.xxh3_64(str(v).encode(), seed=int(user_seed)).intdigest() % g):
            local_estimate[v] += 1
    # Apply the correction factor
    local_estimate = local_estimate
    return local_estimate


def find_hash_function(seed_list, target_set, domain_eliminate, g, num_map_AO):
    # log the max projection number
    best_score = -np.inf
    # log the best projection seed
    best_seed = -1
    # log the target mapped
    best_target_mapped = None
    # log the best hash value
    best_hash_value = None
    # log the min gap
    best_gap = None
    for seed in seed_list:
        hash_projection_list = np.zeros(g)
        hash_other_projection_list = np.zeros(g)
        hash_result = None
        for item in target_set:
            hash_result = xxhash.xxh3_64(str(item).encode(), seed=seed).intdigest() % g
            hash_projection_list[hash_result] += 1
        for item in domain_eliminate:
            hash_result = xxhash.xxh3_64(str(item).encode(), seed=seed).intdigest() % g
            hash_other_projection_list[hash_result] += 1
        score = hash_projection_list - np.abs(num_map_AO - hash_projection_list - hash_other_projection_list)
        current_best_score = np.max(score)
        max_indices = np.where(score == current_best_score)[0]
        current_max_target_mapped = hash_projection_list[max_indices]
        current_untarget_mapped = hash_other_projection_list[max_indices]
        current_hash_value = max_indices
        current_gap = np.abs(num_map_AO - current_max_target_mapped - current_untarget_mapped)
        if current_best_score > best_score:
            best_seed = seed
            best_score = current_best_score
            best_hash_value = current_hash_value
            best_gap = current_gap
            best_target_mapped = current_max_target_mapped
    if best_seed == -1:
        return -1, 0.0, None
    return best_seed, best_gap, best_target_mapped, best_hash_value


def process_attacker_User(attacker_args, n, ratio, target_set, g, domain, splits):
    '''
    Craft one OLH-User fake report.
    :param attacker_args: (i, attacker_seed, num_map_AO) where num_map_AO is the
        support count this fake user aims for (drawn from omega under APA)
    :return: (index in User_Seed, attack_vector)
    '''
    i, attacker_seed, num_map_AO = attacker_args
    # Per-attacker RNG: forked pool workers would otherwise share random state
    rng = random.Random(int(attacker_seed))
    average_project_hash = int(domain / g)
    vector = np.zeros(domain, dtype=int)
    if splits < average_project_hash:
        # Split the target set for each user
        splits_list = rng.sample(sorted(target_set), splits)
        # Every other item counts toward the bucket size, so the crafted
        # report's support is matched to num_map_AO as a whole
        remaining_set = set(range(domain)) - set(splits_list)
        seed_list = rng.sample(range(1, 10000000), num_samples)
        best_seed, best_gap, current_max_target_mapped, best_hash_value = find_hash_function(seed_list, splits_list,
                                                                                             remaining_set, g,
                                                                                             int(num_map_AO))
    else:
        print('splits > averge_project_hash')
        exit(0)
    # Calculate the index in User_Seed to update
    index = int(n * (1 - ratio) + i)
    # find_hash_function returns every tied bucket; report the first
    best_hash_value = int(best_hash_value[0])
    for v in range(domain):
        hashed_value = xxhash.xxh3_64(str(v).encode(), seed=int(best_seed)).intdigest() % g
        if hashed_value == best_hash_value:
            vector[v] = 1
   # print(f'attacker:{i}, target_map:{current_max_target_mapped}, diff:{best_gap}, h_ao:{h_ao}, splits:{splits}')
    return index, vector


def build_support_list_1_OLH(domain, Y, n, User_Seed, ratio, g, target_set, p, splits, h_ao=0, e=1.0, processor=100):
    '''
    build the support list matrix under OLH-User
    :param h_ao: 1 runs APA (support count drawn from omega); any other value
        runs MGA-A with support count d/g jittered by up to 10*h_ao
    :param e: privacy budget epsilon
    '''
    #K_values, K_probs = calculate_prob_according_sample_size(num_samples, domain, g, h_ao, target_set, splits)

    # Calculate the number of attackers
    num_attackers = int(round(n * ratio))

    h_ao = int(h_ao)
    if h_ao == 1:
        # APA: each fake user targets a support count drawn from the genuine-user distribution
        omega_probs = construct_omega(e, domain, 'OLH_User')
        num_map_list = np.random.choice(np.arange(domain), size=num_attackers, p=omega_probs)
    else:
        # MGA-A: expected support d/g with uniform jitter
        num_map = int(domain / g)
        jitter = 10 * h_ao
        num_map_list = np.random.randint(num_map - jitter, num_map + jitter + 1, size=num_attackers)
        num_map_list = np.maximum(num_map_list, 0)
    attacker_seeds = np.random.randint(0, 2**31 - 1, size=num_attackers)
    attacker_args = list(zip(range(num_attackers), attacker_seeds, num_map_list))

    # Prepare the partial function with fixed arguments for multiprocessing
    process_attacker_partial = partial(
        process_attacker_User,
        n=n,
        ratio=ratio,
        target_set=target_set,
        g=g,
        domain=domain,
        splits=splits,
    )

    # Parallel execution of process_attacker using multiprocessing
    with Pool(processes=processor) as pool:
        # Use imap to process in parallel and tqdm for progress bar
        results = list(tqdm(
            pool.imap(process_attacker_partial, attacker_args),
            total=num_attackers,
            desc='Finding optimal seeds'
        ))

    vector_matrix = np.zeros((num_attackers, domain))
    # Update User_Seed with the results from all attackers
    for i, (index, best_vector) in enumerate(results):
        vector_matrix[i, :] = best_vector

    # Create a partial function with fixed arguments for processing user seeds
    process_partial = partial(
        process_user_seeds,
        User_Seed_noattack=User_Seed,
        Y_Nattack=Y,
        domain=domain,
        g=g
    )

    # Process user seeds across multiple processes
    with Pool(processes=processor) as pool:
        estimates = pool.map(process_partial, range(n - num_attackers))
    # input attack's estimate
    '''with Pool(processes=processor) as pool:
        estimates_input = pool.map(process_partial, range(int(n)))'''

    # Combine the results from all attackers
    estimates = np.array(estimates)
    # estimates_input = np.array(estimates_input)
    # estimates_input = estimates_input.reshape(int(n), domain)
    estimates = np.vstack((estimates, vector_matrix))
    estimates = estimates.reshape(int(n), domain)
    Results_support = estimates
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = np.sum(Results_support, axis=0)
    # Estimations_input = np.sum(estimates_input, axis=0)
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    Estimations = a * Estimations - b
    # Estimations_input = a * Estimations_input - b
    Estimations_input = None
    return Results_support, Results_support_one_list, Estimations, Estimations_input


def process_attacker_server(i, n, ratio, target_set, g, domain, User_Seed, splits):
    '''
    Function to process each attacker.
    Each attacker finds the best hash value and constructs a vector based on it.

    :param i: Index of the attacker.
    :param n: Total number of users.
    :param ratio: Ratio of attackers to total users.
    :param target_set: Set of target items.
    :param g: Range of hash function outputs (modulo value).
    :param domain: Total domain size.
    :param User_Seed: List of hash seeds for users.
    :return: Tuple of (index in User_Seed, attack_vector).
    '''
    # Calculate the index in User_Seed
    index = int(n * (1 - ratio) + i)
    user_seed = User_Seed[index]

    # Compute hash values for all target items
    target_hashes = {}
    splits_list = random.sample(list(target_set), splits)
    for t in splits_list:
        hashed_value = xxhash.xxh3_64(str(t).encode(), seed=int(user_seed)).intdigest() % g
        if hashed_value in target_hashes:
            target_hashes[hashed_value] += 1
        else:
            target_hashes[hashed_value] = 1

    # Find the hash value that maps the most target items
    best_hashed_value = max(target_hashes, key=target_hashes.get)
    max_target_count = target_hashes[best_hashed_value]

    # Construct the attack vector
    attack_vector = np.zeros(domain)
    for v in range(domain):
        hashed_value = xxhash.xxh3_64(str(v).encode(), seed=int(user_seed)).intdigest() % g
        if hashed_value == best_hashed_value:
            attack_vector[v] = 1

    return index, attack_vector


def _olh_server_buckets(index_seed, g, domain):
    '''Bucket h_seed(v) of every item v under one fake user's server-assigned
    hash -- the user's whole choice set (bucket b supports h^-1(b)).'''
    index, user_seed = index_seed
    buckets = np.empty(domain, dtype=np.uint8)
    for v in range(domain):
        buckets[v] = xxhash.xxh3_64(str(v).encode(), seed=int(user_seed)).intdigest() % g
    return index, buckets


def choose_server_apa_buckets(sizes, coverage, omega_probs, mode='hist'):
    '''Pick one bucket per fake user for server-side APA (S-APA).

    In OLH-Server a fake user can only choose which of its g buckets to report,
    and that choice fixes both its support count (bucket size) and how many of
    its targets it covers. Both modes make the fake users' count histogram
    follow the genuine one, omega[k] = floor(m * P(X=k)) (apa_quota):

      'user'  each fake user draws its own count k* (apa_counts) and takes the
              bucket whose size is closest to k*; coverage only breaks ties.
      'hist'  the quota is enforced over the whole population: (user, bucket)
              pairs are taken in order of coverage while their count bin still
              has room, so coverage decides who fills which bin. Users whose
              bins are all full take the bucket whose bin has the most room left.

    :param sizes: (m, g) support count of each bucket
    :param coverage: (m, g) number of the user's targets in each bucket
    :return: (m,) chosen bucket per fake user
    '''
    m, g = sizes.shape
    rows = np.arange(m)
    if mode == 'user':
        k_star = apa_counts(m, omega_probs)
        # Lexicographic: count distance first, then more targets covered.
        score = np.abs(sizes - k_star[:, None]) * (g + 1 + coverage.max()) - coverage
        return np.argmin(score, axis=1)
    if mode != 'hist':
        raise ValueError(f"Unknown server APA mode: {mode}")

    quota = apa_quota(m, omega_probs)
    remaining = np.concatenate([quota, np.zeros(max(0, sizes.max() + 1 - len(quota)), np.int64)])
    chosen = np.full(m, -1, dtype=np.int64)
    # Random order first so equal-coverage pairs are taken in random order.
    pairs = np.random.permutation(m * g)
    pairs = pairs[np.argsort(-coverage.ravel()[pairs], kind='stable')]
    for pair in pairs:
        j, b = divmod(int(pair), g)
        k = sizes[j, b]
        if chosen[j] < 0 and remaining[k] > 0:
            chosen[j] = b
            remaining[k] -= 1
    for j in np.flatnonzero(chosen < 0):
        b = int(np.argmax(remaining[sizes[j]]))
        chosen[j] = b
        remaining[sizes[j, b]] -= 1
    return chosen


def _olh_server_apa_vectors(n, num_attackers, User_Seed, g, domain, target_set,
                            splits, epsilon, processor, mode):
    '''Fake users' support vectors under server-side APA (see
    choose_server_apa_buckets). Subsets are drawn here, from the caller's
    seeded np.random state, so workers need no randomness of their own.'''
    start = n - num_attackers
    targets = np.array(sorted(target_set))
    subsets = np.array([np.random.choice(targets, splits, replace=False)
                        for _ in range(num_attackers)]).reshape(num_attackers, splits)

    work = [(start + i, User_Seed[start + i]) for i in range(num_attackers)]
    with Pool(processes=processor) as pool:
        results = list(tqdm(
            pool.imap(partial(_olh_server_buckets, g=g, domain=domain), work, chunksize=64),
            total=num_attackers,
            desc='Hashing attackers (S-APA)'
        ))
    buckets = np.empty((num_attackers, domain), dtype=np.uint8)
    for index, row in results:
        buckets[index - start] = row

    sizes = np.stack([(buckets == b).sum(axis=1) for b in range(g)], axis=1)
    target_buckets = np.take_along_axis(buckets, subsets, axis=1)
    coverage = np.stack([(target_buckets == b).sum(axis=1) for b in range(g)], axis=1)
    chosen = choose_server_apa_buckets(
        sizes, coverage, construct_omega(epsilon, domain, 'OLH_Server'), mode)
    return (buckets == chosen[:, None].astype(np.uint8)).astype(np.float64)


def build_support_list_1_OLH_Server(domain, Y, n, User_Seed, ratio, g, target_set, p, splits, h_ao=0, epsilon=1.0, processor=100, server_apa='hist'):
    '''
    build the support list matrix
    h_ao=2 runs server-side APA (choose_server_apa_buckets, mode server_apa);
    any other h_ao runs MGA-A (best-covering bucket per fake user).
    :return:
    '''
    # Calculate the number of attackers
    num_attackers = int(round(n * ratio))
    num_normal = int(n - num_attackers)

    if h_ao == 2:
        vector_matrix = _olh_server_apa_vectors(
            n, num_attackers, User_Seed, g, domain, target_set, splits,
            epsilon, processor, server_apa)
    else:
        # Prepare the partial function with fixed arguments for multiprocessing
        process_attacker_partial = partial(
            process_attacker_server,
            n=n,
            ratio=ratio,
            target_set=target_set,
            g=g,
            domain=domain,
            User_Seed=User_Seed,
            splits=splits
        )

        # Parallel execution of process_attacker using multiprocessing
        with Pool(processes=processor) as pool:
            # Use imap to process in parallel and tqdm for progress bar
            results = list(tqdm(
                pool.imap(process_attacker_partial, range(num_attackers)),
                total=num_attackers,
                desc='Processing attackers'
            ))

        vector_matrix = np.zeros((num_attackers, domain))
        # Update User_Seed with the results from all attackers
        for i, (index, best_vector) in enumerate(results):
            vector_matrix[i, :] = best_vector

    # Create a partial function with fixed arguments for processing user seeds
    process_partial = partial(
        process_user_seeds,
        User_Seed_noattack=User_Seed,
        Y_Nattack=Y,
        domain=domain,
        g=g
    )

    # Process user seeds across multiple processes
    with Pool(processes=processor) as pool:
        estimates = pool.map(process_partial, range(int(num_normal)))
    # input attack's estimate
    '''with Pool(processes=processor) as pool:
        estimates_input = pool.map(process_partial, range(int(n)))'''

    # Combine the results from all attackers
    estimates = np.array(estimates)
    # estimates_input = np.array(estimates_input)
    # estimates_input = estimates_input.reshape(int(n), domain)
    estimates = np.vstack((estimates, vector_matrix))
    estimates = estimates.reshape(int(n), domain)
    Results_support = estimates
    Results_support_one_list = np.sum(Results_support, axis=1)
    Estimations = np.sum(Results_support, axis=0)
    # Estimations_input = np.sum(estimates_input, axis=0)
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    Estimations = a * Estimations - b
    # Estimations_input = a * Estimations_input - b
    Estimations_input = None
    return Results_support, Results_support_one_list, Estimations, Estimations_input


def HST_Server(X, ratio, domain, epsilon, n, target_set, splits):
    '''
    Perform the HST protocol
    :param X: The real values for each users
    :param ratio: fake users ratio
    :param domain: domain size
    :param epsilon: privacy budget
    :param n: number of users
    :param target_set: fake users target set
    :return: support_list, one_list, ESTIMATE_DIST, ESTIMATE_Input
    '''
    c = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
    s_vectors = np.zeros((n, domain))
    fake_user_num = int(round(n * ratio))
    normal_user_num = n - fake_user_num
    start_idx = n - fake_user_num
    y_values = np.zeros(n)
    for i in range(n):
        # Generate random public vector s_i
        s_i = np.random.choice([-1.0, 1.0], size=domain)
        s_vectors[i, :] = s_i
    for i in range(normal_user_num):
        # User's true data item
        v = X[i]  # Assuming X[i] is in the range [0, domain-1]
        # Generate random public vector s_j
        s_i = s_vectors[i, :]
        # Get s_j[v_b]
        s_i_v = s_i[v]
        # Perturbation process
        if random.random() < math.exp(epsilon) / (math.exp(epsilon) + 1):
            y = c * s_i_v
        else:
            y = -c * s_i_v
        y_values[i] = y
    for i in range(fake_user_num):
        # MGA-A: each fake user promotes its own r'-subset of the targets.
        splits_list = random.sample(list(target_set), splits)
        idx = start_idx + i
        s_i = s_vectors[idx, :]
        positive_count = 0
        negative_count = 0
        for v in splits_list:
            if s_i[v] == 1.0:
                positive_count += 1
            elif s_i[v] == -1.0:
                negative_count += 1
        if positive_count >= negative_count:
            y = c
        else:
            y = -c
        y_values[idx] = y

    support_list = y_values.reshape(-1, 1) * s_vectors
    ESTIMATE_DIST = np.sum(support_list, axis=0)
    # Count the items the report actually supports (s_i(v) * y > 0): k+ when
    # y > 0, d - k+ when y < 0 -- the same set the item features read.
    Results_support_one_list = np.sum(support_list > 0, axis=1)

    return support_list, Results_support_one_list, ESTIMATE_DIST, ESTIMATE_DIST


def HST_Users(X, ratio, domain, epsilon, n, target_set, h_ao, splits):
    '''
    Perform the HST protocol
    :param X: The real values for each users
    :param ratio: fake users ratio
    :param domain: domain size
    :param epsilon: privacy budget
    :param n: number of users
    :param target_set: fake users target set
    :param h_ao: APA's parameter
    :param splits: MGA-A's parameter
    :return: support_list, one_list, ESTIMATE_DIST, ESTIMATE_Input
    '''
    c = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
    s_vectors = np.zeros((n, domain))
    average_1_num = domain / 2
    fake_user_num = int(round(n * ratio))
    normal_user_num = n - fake_user_num
    start_idx = n - fake_user_num
    # Exact APA (h_ao=2): each fake user's count of +1 positions follows the
    # genuine B(d, 1/2) histogram, omega[k] = floor(m * P(X=k)). Otherwise every
    # fake user targets d/2, jittered by +-10*h_ao (0 = MGA-A, 1 = legacy).
    apa_k = (apa_counts(fake_user_num, construct_omega(epsilon, domain, 'HST_User'))
             if h_ao == 2 else None)
    h_ao = 0 if h_ao == 2 else h_ao * 10
    y_values = np.zeros(n)
    # theoretical APA use the omega_list to replace the averge_1_num and set h_ao = 0
    '''if h_ao != 0:
        average_1_num = np.random.choice([i for i in range(domain)], construct_omega(epsilon, domain, 'OUE'))
        h_ao = 0'''
    for i in range(normal_user_num):
        # User's true data item
        v = X[i]  # Assuming X[i] is in the range [0, domain-1]
        # Generate random public vector s_j
        s_i = np.random.choice([-1.0, 1.0], size=domain)
        s_vectors[i, :] = s_i
        # Get s_j[v_b]
        s_i_v = s_i[v]
        # Perturbation process
        if random.random() < math.exp(epsilon) / (math.exp(epsilon) + 1):
            y = c * s_i_v
        else:
            y = -c * s_i_v
        y_values[i] = y
    for i in range(fake_user_num):

        splits_list = random.sample(list(target_set), splits)
        local_user_data = np.full(domain, -1.0)
        local_user_data[list(splits_list)] = 1
        remaining_set = list(set(range(domain)) - set(splits_list))
        target_count = apa_k[i] if apa_k is not None else average_1_num
        diff = int(target_count - len(splits_list))
        diff_AO = random.randint(diff - h_ao, diff + h_ao)
        diff_AO = max(0, min(diff_AO, len(remaining_set)))
        #print(f'attacker:{i}, exp1:{average_1_num}, h_ao:{h_ao}, splits:{splits}')
        if diff_AO > 0 and len(remaining_set) >= diff_AO:
            random_numbers = random.sample(remaining_set, diff_AO)
            local_user_data[random_numbers] = 1
        idx = start_idx + i
        s_vectors[idx,:] = local_user_data
        y = c
        y_values[idx] = y

    support_list = y_values.reshape(-1, 1) * s_vectors
    ESTIMATE_DIST = np.sum(support_list, axis=0)
    Results_support_one_list = np.sum(s_vectors == 1, axis=1)

    return support_list, Results_support_one_list, ESTIMATE_DIST, ESTIMATE_DIST


#def perturb_normal_olh(X, g, p, q):
    Y_normal = np.zeros(len(X))
    for i, v in enumerate(X):
        # Generate hash value
        x = (xxhash.xxh3_64(str(v).encode(), seed=i).intdigest() % g)
        y = x
        p_sample = np.random.random_sample()

        # Apply perturbation
        if p_sample <= p:

            # perturb
            y = np.random.randint(0, g)
        Y_normal[i] = y
    return Y_normal


#def build_support_list_normal_olh(Y_normal, n, domain, g, p):
    Results_support_normal = np.zeros((n, domain))
    Estimations_normal_raw = np.zeros(domain)

    for i in range(n):
        user_seed = i  # Use the same seed as in perturbation
        for v in range(domain):
            hashed_value = (xxhash.xxh3_64(str(v).encode(), seed=user_seed).intdigest() % g)
            if Y_normal[i] == hashed_value:
                Results_support_normal[i, v] += 1
                Estimations_normal_raw[v] += 1

    Results_support_one_list_normal = np.sum(Results_support_normal, axis=1)

    # Apply the OLH correction factor
    a = 1.0 * g / (p * g - 1)
    b = 1.0 * n / (p * g - 1)
    Estimations_normal = a * Estimations_raw - b

    return Results_support_normal, Results_support_one_list_normal, Estimations_normal
