#!/usr/bin/env python
"""
Usage:
    python generate_dataset_pca.py --output dataset_pca.csv
    python generate_dataset_pca.py --output pca.csv --protocols OUE --epsilons 0.5 1.0 --experiments 3
"""

import argparse
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import (
    DEFAULT_EPSILONS,
    DEFAULT_RATIOS,
    DEFAULT_TARGET_SIZES,
    DEFAULT_SPLITS,
    DEFAULT_SEED,
    DEFAULT_VARIANCE_THRESHOLD,
    DEFAULT_MAX_PCA_DIM,
    DATASET_CONFIGS,
    DATASET_CONFIGS_FULL,
    DATASET_CONFIG_COLUMNS,
)
from attacker_detector.data.generators import (
    generate_zipf_dist,
    generate_emoji_dist,
    generate_fire_dist,
    construct_omega,
    build_normal_lists_from_mechanism_stochastic,
    build_support_list_1_OUE,
    build_support_list_1_OLH,
    build_support_list_1_OLH_Server,
)
from attacker_detector.data.generators.attacks import perturb_OUE_multi, HST_Server, HST_Users



def get_distribution_generator(dataset_type: str):
    """Get distribution generator function by type."""
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

    # Generate user level dataset without feature extraction
    
    if seed is not None:
        np.random.seed(seed)

    generator = get_distribution_generator(dataset_type)
    X, REAL_DIST = generator(n, domain, seed=seed)

    target_set = set(np.random.choice(domain, size=target_set_size, replace=False))

    base_mechanism = 'OLH' if protocol in ('OLH', 'OLH_User', 'OLH_Server') else protocol

    # Generate ideal (non-attacked) lists for reference
    # but kept for API consistency; could be commented out)
    # ideal_support_list, ideal_one_list, ideal_ESTIMATE_DIST, _ = \
    #     build_normal_lists_from_mechanism_stochastic(
    #         epsilon=epsilon,
    #         d=domain,
    #         n=n,
    #         mechanism=base_mechanism,
    #         seed=seed if seed else 42
    #     )

    if protocol == "OLH":
        g = int(round(math.exp(epsilon))) + 1
        p = math.exp(epsilon) / (math.exp(epsilon) + g - 1)
        User_Seed = np.arange(n)
        Y = np.zeros(n)

        if olh_setting == 'user':
            support_list, one_list, ESTIMATE_DIST, _ = \
                build_support_list_1_OLH(
                    domain, Y, n, User_Seed, ratio, g, target_set,
                    p, splits, h_ao, epsilon, processor=processors
                )
        else:
            support_list, one_list, ESTIMATE_DIST, _ = \
                build_support_list_1_OLH_Server(
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

        support_list, one_list, ESTIMATE_DIST, _ = \
            build_support_list_1_OUE(Y_data, n, epsilon)

    elif protocol == "HST_User":
        support_list, one_list, ESTIMATE_DIST, _ = \
            HST_Users(
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
        support_list, one_list, ESTIMATE_DIST, _ = \
            HST_Server(
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


def apply_pca_heuristic(
    support_list: np.ndarray,
    variance_threshold: float = 0.90,
    max_dim: int = None
) -> tuple:
    """
    Apply PCA with Kaiser criterion + cumulative variance floor.

    Steps:
        1. Standardize support_list (zero-mean, unit-variance per column).
        2. Fit PCA with all components.
        3. Kaiser criterion: count components whose eigenvalue > 1.
        4. Variance floor: count components to reach >= variance_threshold.
        5. k = max(kaiser_k, variance_k); optionally capped at max_dim.
        6. Project data onto the top-k components.

        pca_features: (n, k) projected data
        k: number of retained components
        explained_variance: cumulative explained variance ratio for the k components
    """
    n_samples, n_features = support_list.shape

    max_components = min(n_samples, n_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(support_list.astype(np.float64))

    pca_full = PCA(n_components=max_components)
    pca_full.fit(X_scaled)

    eigenvalues = pca_full.explained_variance_
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Step 3: Kaiser criterion — eigenvalue > 1
    kaiser_k = int(np.sum(eigenvalues > 1.0))
    kaiser_k = max(kaiser_k, 1)  # at least 1

    # # Step 4: Variance floor
    # variance_k = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    # variance_k = min(variance_k, max_components)

    #k = max(kaiser_k, variance_k)
    k = kaiser_k    
    if max_dim is not None: # Can be specified
        k = min(k, max_dim)
    k = min(k, max_components)

    pca_features = pca_full.transform(X_scaled)[:, :k]
    explained = float(cumulative_variance[k - 1])

    return pca_features, k, explained



def build_tasks(args) -> list:
    """Build list of task dicts for all experiment configurations."""
    configs = DATASET_CONFIGS_FULL if args.full_scale else DATASET_CONFIGS
    tasks = []

    for epsilon in args.epsilons:
        for dataset_type in args.datasets:
            dataset_config = configs[dataset_type]
            domain = args.domain if args.domain else dataset_config['domain']
            n = args.n if args.n else dataset_config['n']

            for protocol in args.protocols:
                for ratio in args.ratios:
                    for target_size in args.target_sizes:
                        for splits in args.splits:
                            for exp_i in range(args.experiments):
                                # Determine OLH setting
                                if protocol == "OLH_User":
                                    base_protocol = "OLH"
                                    olh_setting = "user"
                                elif protocol == "OLH_Server":
                                    base_protocol = "OLH"
                                    olh_setting = "server"
                                else:
                                    base_protocol = protocol
                                    olh_setting = "server"

                                # Deterministic seed per config
                                config_idx = len(tasks) + 1
                                seed = args.seed + config_idx * 1000 + exp_i

                                tasks.append({
                                    'epsilon': epsilon,
                                    'domain': domain,
                                    'n': n,
                                    'protocol': base_protocol,
                                    'protocol_label': protocol,
                                    'ratio': ratio,
                                    'target_set_size': target_size,
                                    'splits': splits,
                                    'dataset_type': dataset_type,
                                    'h_ao': 1,
                                    'seed': seed,
                                    'inner_processors': args.inner_processors,
                                    'olh_setting': olh_setting,
                                    'variance_threshold': args.variance_threshold,
                                    'max_pca_dim': args.max_pca_dim,
                                    'exp_i': exp_i,
                                    'desc': (
                                        f"ε={epsilon}, {dataset_type}, {protocol}, "
                                        f"ratio={ratio}, target={target_size}, "
                                        f"splits={splits}, exp={exp_i + 1}"
                                    ),
                                })

    return tasks



def run_one_task(task: dict) -> dict:
    """
    Execute one experiment configuration:
      1. Generate perturbed data
      2. Apply PCA heuristic
      3. Build result dict with DataFrame chunk
    """
    try:
        support_list, labels = generate_perturbed_data(
            epsilon=task['epsilon'],
            domain=task['domain'],
            n=task['n'],
            protocol=task['protocol'],
            ratio=task['ratio'],
            target_set_size=task['target_set_size'],
            splits=task['splits'],
            dataset_type=task['dataset_type'],
            h_ao=task['h_ao'],
            seed=task['seed'],
            processors=task['inner_processors'],
            olh_setting=task['olh_setting'],
        )

        pca_features, pca_dim, explained_var = apply_pca_heuristic(
            support_list,
            variance_threshold=task['variance_threshold'],
            max_dim=task['max_pca_dim'],
        )

        num_users = len(labels)
        num_attackers = int(labels.sum())

        pc_cols = [f'pc_{i}' for i in range(pca_dim)]
        df_pca = pd.DataFrame(pca_features, columns=pc_cols)

        df_pca['pca_dim'] = pca_dim
        df_pca['pca_variance_explained'] = explained_var
        df_pca['target_set_size'] = task['target_set_size']
        df_pca['attacker_ratio'] = task['ratio']
        df_pca['protocol'] = task['protocol_label']
        df_pca['splits'] = task['splits']
        df_pca['epsilon'] = task['epsilon']
        df_pca['dataset_type'] = task['dataset_type']
        df_pca['label'] = labels

        return {
            'ok': True,
            'df': df_pca,
            'pca_dim': pca_dim,
            'explained_var': explained_var,
            'num_users': num_users,
            'num_attackers': num_attackers,
            'desc': task['desc'],
        }

    except Exception as e:
        return {
            'ok': False,
            'desc': task['desc'],
            'error': str(e),
            'traceback': traceback.format_exc(),
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate LDP attack detection dataset with PCA features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output CSV file path'
    )

    parser.add_argument(
        '--protocols',
        nargs='+',
        default=['OUE', 'OLH'],
        choices=['OUE', 'OLH', 'OLH_User', 'OLH_Server', 'HST_User', 'HST_Server'],
        help='LDP protocols to use'
    )

    parser.add_argument(
        '--epsilons',
        nargs='+',
        type=float,
        default=DEFAULT_EPSILONS,
        help='Privacy parameters (epsilon values)'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['zipf', 'emoji', 'fire'],
        choices=['zipf', 'emoji', 'fire'],
        help='Dataset types to generate'
    )

    parser.add_argument(
        '--ratios',
        nargs='+',
        type=float,
        default=DEFAULT_RATIOS,
        help='Attacker ratios'
    )

    parser.add_argument(
        '--target-sizes',
        nargs='+',
        type=int,
        default=DEFAULT_TARGET_SIZES,
        help='Target set sizes'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        type=int,
        default=DEFAULT_SPLITS,
        help='Split values'
    )

    parser.add_argument(
        '--experiments',
        type=int,
        default=5,
        help='Number of experiments per configuration'
    )

    parser.add_argument(
        '--full-scale',
        action='store_true',
        help='Use full-scale dataset sizes (100k+ users)'
    )

    parser.add_argument(
        '--n',
        type=int,
        default=None,
        help='Override number of users (applies to all datasets)'
    )

    parser.add_argument(
        '--domain',
        type=int,
        default=None,
        help='Override domain size (applies to all datasets)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed'
    )

    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing output file instead of overwriting'
    )


    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=DEFAULT_VARIANCE_THRESHOLD,
        help='Minimum cumulative explained variance for PCA component selection'
    )

    parser.add_argument(
        '--max-pca-dim',
        type=int,
        default=DEFAULT_MAX_PCA_DIM,
        help='Hard cap on number of PCA components (None = no cap)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of outer ProcessPoolExecutor workers (for OUE/HST tasks)'
    )

    parser.add_argument(
        '--inner-processors',
        type=int,
        default=4,
        help='Number of inner parallel processes per task (for OUE perturbation / OLH hashing)'
    )

    return parser.parse_args()



def main():
    """Main entry point."""
    args = parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    if os.path.exists(args.output) and not args.append:
        print(f"Output file already exists, removing: {args.output}")
        os.remove(args.output)

    tasks = build_tasks(args)

    # Separate OLH tasks (run sequentially) from non-OLH tasks (run in parallel)
    olh_protocols = {'OLH', 'OLH_User', 'OLH_Server'}
    parallel_tasks = [t for t in tasks if t['protocol_label'] not in olh_protocols]
    sequential_tasks = [t for t in tasks if t['protocol_label'] in olh_protocols]

    total_configs = (
        len(args.epsilons) *
        len(args.datasets) *
        len(set(args.protocols)) *
        len(args.ratios) *
        len(args.target_sizes) *
        len(args.splits)
    )
    total_runs = len(tasks)

    print("=" * 80)
    print("LDP Attack Detection PCA Dataset Generator (Parallel)")
    print("=" * 80)
    print(f"Output: {args.output}")
    print(f"Protocols: {args.protocols}")
    print(f"Epsilons: {args.epsilons}")
    print(f"Datasets: {args.datasets}")
    print(f"Ratios: {args.ratios}")
    print(f"Target sizes: {args.target_sizes}")
    print(f"Splits: {args.splits}")
    print(f"Experiments per config: {args.experiments}")
    print(f"Total configurations: {total_configs}")
    print(f"Total experiment runs: {total_runs}")
    print(f"  Parallel tasks (OUE/HST): {len(parallel_tasks)}")
    print(f"  Sequential tasks (OLH):   {len(sequential_tasks)}")
    print(f"Outer workers: {args.workers}")
    print(f"Inner processors per task: {args.inner_processors}")
    print(f"Variance threshold: {args.variance_threshold}")
    print(f"Max PCA dim: {args.max_pca_dim or 'None (no cap)'}")
    print("=" * 80)

    total_users = 0
    total_attackers = 0
    num_success = 0
    num_failed = 0
    all_pca_dims = []
    all_results = []  # Collect all successful result DataFrames

    def _handle_result(result):
        """Process a single task result: accumulate stats and collect DataFrame."""
        nonlocal total_users, total_attackers, num_success, num_failed

        if result['ok']:
            all_results.append(result['df'])

            total_users += result['num_users']
            total_attackers += result['num_attackers']
            all_pca_dims.append(result['pca_dim'])
            num_success += 1

            print(
                f'[DONE] {result["desc"]} | '
                f'users={result["num_users"]}, attackers={result["num_attackers"]}, '
                f'pca_dim={result["pca_dim"]}, var={result["explained_var"]:.4f}'
            )
        else:
            num_failed += 1
            print(f'[FAIL] {result["desc"]} | error={result["error"]}')
            print(result['traceback'])

    if parallel_tasks:
        print(f"\n--- Phase 1: Processing {len(parallel_tasks)} OUE/HST tasks in parallel ---")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_one_task, task): task
                for task in parallel_tasks
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel (OUE/HST)"):
                result = future.result()
                _handle_result(result)

    if sequential_tasks:
        print(f"\n--- Phase 2: Processing {len(sequential_tasks)} OLH tasks sequentially ---")
        for task in tqdm(sequential_tasks, desc="Sequential (OLH)"):
            result = run_one_task(task)
            _handle_result(result)

    # --- Pad and write ---
    if all_results:
        max_pca_dim = max(all_pca_dims)
        print(f"\nPadding all chunks to {max_pca_dim} PCA columns...")

        meta_columns = [
            'pca_dim', 'pca_variance_explained',
            'target_set_size', 'attacker_ratio', 'protocol',
            'splits', 'epsilon', 'dataset_type', 'label'
        ]
        all_pc_cols = [f'pc_{i}' for i in range(max_pca_dim)]
        final_columns = all_pc_cols + meta_columns

        # Pad each DataFrame to uniform width and concatenate
        padded_dfs = []
        for df_chunk in all_results:
            # Add any missing pc columns as zeros
            for col in all_pc_cols:
                if col not in df_chunk.columns:
                    df_chunk[col] = 0.0
            padded_dfs.append(df_chunk[final_columns])

        df_final = pd.concat(padded_dfs, ignore_index=True)

        # Write (or append) to CSV
        if args.append and os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            df_final.to_csv(args.output, mode='a', header=False, index=False)
        else:
            df_final.to_csv(args.output, index=False)

        print(f"Wrote {len(df_final):,} rows to {args.output}")

    print("\n" + "=" * 80)
    print("PCA Dataset Generation Complete")
    print("=" * 80)
    print(f"Successful runs: {num_success}")
    print(f"Failed runs: {num_failed}")
    print(f"Total users written: {total_users:,}")
    print(f"Total attackers written: {total_attackers:,}")

    if all_pca_dims:
        print(f"PCA dimensions: min={min(all_pca_dims)}, max={max(all_pca_dims)}, "
              f"mean={np.mean(all_pca_dims):.1f}")

    if total_users > 0:
        benign = total_users - total_attackers
        print(f"Benign: {benign:,} ({100.0 * benign / total_users:.2f}%)")
        print(f"Attackers: {total_attackers:,} ({100.0 * total_attackers / total_users:.2f}%)")
        print(f"Saved to: {args.output}")
        print(f"File size: {os.path.getsize(args.output) / (1024 * 1024):.2f} MB")
    else:
        print("ERROR: No data generated.")
        sys.exit(1)



if __name__ == '__main__':
    main()
