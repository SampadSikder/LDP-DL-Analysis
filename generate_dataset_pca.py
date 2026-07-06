#!/usr/bin/env python
"""
Usage:
    python generate_dataset_pca.py --output dataset_pca.pt
    python generate_dataset_pca.py --output dataset_pca.pt --protocols OUE --epsilons 0.5 1.0 --experiments 3
"""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
import torch

from config import (
    DEFAULT_EPSILONS,
    DEFAULT_RATIOS,
    DEFAULT_TARGET_SIZES,
    DEFAULT_SPLITS,
    DEFAULT_SEED,
    DEFAULT_PCA_DIM,
    DEFAULT_KNN_K,
    DATASET_CONFIGS,
    DATASET_CONFIGS_FULL,
)
from attacker_detector.data.generators import (
    generate_perturbed_data,
    build_graph_data,
)


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
                            if splits > target_size:
                                continue
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
                                    'pca_dim': args.pca_dim,
                                    'knn_k': args.knn_k,
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
      2. Build graph data (PCA, kNN, centralities, densities)
      3. Return result dict with PyG Data object
    """
    try:
        support_list, labels, real_dist, estimate_dist_all = generate_perturbed_data(
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

        metadata = {
            'protocol': task['protocol_label'],
            'dataset_type': task['dataset_type'],
            'ratio': task['ratio'],
            'target_set_size': task['target_set_size'],
            'splits': task['splits'],
        }

        #Final Feature vector
        data_obj = build_graph_data(
            support_list=support_list,
            labels=labels,
            epsilon=task['epsilon'],
            pca_dim=task['pca_dim'],
            knn_k=task['knn_k'],
            metadata=metadata,
            real_dist=real_dist,
            estimate_dist_all=estimate_dist_all,
        )

        return {
            'ok': True,
            'data': data_obj,
            'pca_dim': task['pca_dim'],
            'explained_var': data_obj.pca_variance_explained,
            'num_users': len(labels),
            'num_attackers': int(labels.sum()),
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
        description='Generate LDP attack detection dataset with Graph features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output .pt file path for the graph dataset'
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
        '--pca-dim',
        type=int,
        default=DEFAULT_PCA_DIM,
        help='Number of PCA components to retain'
    )

    parser.add_argument(
        '--knn-k',
        type=int,
        default=DEFAULT_KNN_K,
        help='Number of neighbors for kNN graph'
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
    print("LDP Attack Detection Graph Dataset Generator")
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
    print(f"PCA Dimension: {args.pca_dim}")
    print(f"kNN Neighbors: {args.knn_k}")
    print("=" * 80)

    total_users = 0
    total_attackers = 0
    num_success = 0
    num_failed = 0
    all_graphs = []

    def _handle_result(result):
        nonlocal total_users, total_attackers, num_success, num_failed

        if result['ok']:
            all_graphs.append(result['data'])
            total_users += result['num_users']
            total_attackers += result['num_attackers']
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

    # Save interval: persist accumulated graphs to disk every N completed tasks
    SAVE_EVERY = 500

    def _save_to_disk():
        """Append accumulated graphs to the output .pt file and free memory."""
        nonlocal all_graphs
        if not all_graphs:
            return 0
        # Load existing graphs from the output file if present
        if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
            try:
                existing = torch.load(args.output, weights_only=False)
                if isinstance(existing, list):
                    all_graphs = existing + all_graphs
            except Exception:
                pass
        torch.save(all_graphs, args.output)
        saved_count = len(all_graphs)
        all_graphs = []
        return saved_count

    if parallel_tasks:
        # Disable inner multiprocessing for parallel tasks to avoid nested.
        for t in parallel_tasks:
            t['inner_processors'] = 1

        print(f"\n--- Phase 1: Processing {len(parallel_tasks)} OUE/HST tasks in parallel ---")

        # Process in batches to limit memory usage
        batch_size = max(SAVE_EVERY, args.workers * 50)
        for batch_start in range(0, len(parallel_tasks), batch_size):
            batch = parallel_tasks[batch_start:batch_start + batch_size]
            batch_end = min(batch_start + batch_size, len(parallel_tasks))
            print(f"\n  Batch [{batch_start+1}-{batch_end}] of {len(parallel_tasks)}")

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_one_task, task): task
                    for task in batch
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Parallel (OUE/HST) [{batch_start+1}-{batch_end}]"
                ):
                    try:
                        result = future.result()
                        _handle_result(result)
                    except Exception as e:
                        task_info = futures[future]
                        num_failed += 1
                        print(f'[CRASH] {task_info["desc"]} | Worker killed: {e}')

                    # Periodic save to disk to free memory
                    if len(all_graphs) >= SAVE_EVERY:
                        n_saved = _save_to_disk()
                        print(f"  [SAVED] {n_saved} graphs to {args.output}")

            if all_graphs:
                n_saved = _save_to_disk()
                print(f"  [SAVED] {n_saved} graphs to {args.output}")

    if sequential_tasks:
        print(f"\n--- Phase 2: Processing {len(sequential_tasks)} OLH tasks sequentially ---")
        for task in tqdm(sequential_tasks, desc="Sequential (OLH)"):
            result = run_one_task(task)
            _handle_result(result)

            if len(all_graphs) >= SAVE_EVERY:
                n_saved = _save_to_disk()
                print(f"  [SAVED] {n_saved} graphs to {args.output}")

    # Final save for any remaining graphs in memory
    if all_graphs:
        n_saved = _save_to_disk()
        print(f"Saved final {n_saved} graphs to {args.output}")

    if os.path.exists(args.output) and os.path.getsize(args.output) > 0:
        final_graphs = torch.load(args.output, weights_only=False)
        total_saved = len(final_graphs) if isinstance(final_graphs, list) else 0
        del final_graphs
    else:
        total_saved = 0

    print("\n" + "=" * 80)
    print("Graph Dataset Generation Complete")
    print("=" * 80)
    print(f"Successful runs: {num_success}")
    print(f"Failed runs: {num_failed}")
    print(f"Total graphs saved: {total_saved}")
    print(f"Total users: {total_users:,}")
    print(f"Total attackers: {total_attackers:,}")

    if total_users > 0:
        benign = total_users - total_attackers
        print(f"Benign: {benign:,} ({100.0 * benign / total_users:.2f}%)")
        print(f"Attackers: {total_attackers:,} ({100.0 * total_attackers / total_users:.2f}%)")
        print(f"Saved to: {args.output}")
        print(f"File size: {os.path.getsize(args.output) / (1024 * 1024):.2f} MB")
    else:
        print("WARNING: No data generated.")
        sys.exit(1)


if __name__ == '__main__':
    main()
