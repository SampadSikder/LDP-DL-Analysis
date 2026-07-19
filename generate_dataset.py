#!/usr/bin/env python

import argparse
import json
import math
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from config import (
    DEFAULT_EPSILONS,
    DEFAULT_RATIOS,
    DEFAULT_TARGET_SIZES,
    DEFAULT_SPLITS,
    DEFAULT_SEED,
    DATASET_CONFIGS,
    DATASET_CONFIGS_FULL
)
from attacker_detector.data.generators import (
    generate_perturbed_data,
    extract_user_level_features,
    FEATURE_NAMES,
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
                                    'inner_processors': 1 if protocol.startswith('OLH') else args.inner_processors,
                                    'olh_setting': olh_setting,
                                    'robust_iterations': args.robust_iterations,
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
      1. Generate perturbed data via generate_perturbed_data()
      2. Extract features
      3. Return result dict with features, labels, and metadata
    """
    try:
        support_list, labels, real_dist, estimate_dist, one_list = generate_perturbed_data(
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

        features, feat_meta = extract_user_level_features(
            support_list=support_list,
            one_list=one_list,
            epsilon=task['epsilon'],
            protocol=task['protocol_label'],
            domain=task['domain'],
            n=task['n'],
            robust_iterations=task['robust_iterations'],
        )

        config_summary = [
            task['target_set_size'], task['ratio'], task['protocol_label'],
            task['splits'], task['epsilon'], task['dataset_type']
        ]

        p_label = task['protocol_label']
        if p_label.startswith('HST'):
            support_tensor_cast = support_list.astype(np.float32)
        else:
            support_tensor_cast = support_list.astype(np.uint8)

        graph_meta = {
            'pi_hat': feat_meta['pi_hat'].astype(np.float32),
            'pi_true': real_dist.astype(np.float32),
            'p': feat_meta['p'],
            'q': feat_meta['q'],
            'epsilon': task['epsilon'],
            'protocol': p_label,
            'support_tensor': support_tensor_cast,
        }

        return {
            'ok': True,
            'features': features,
            'labels': labels,
            'config_summary': config_summary,
            'graph_meta': graph_meta,
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


def append_metadata_to_zip(metadata_path, all_metadata, start_idx):
    import io
    import zipfile
    with zipfile.ZipFile(metadata_path, mode='a', compression=zipfile.ZIP_DEFLATED) as zipf:
        for idx, gm in enumerate(all_metadata):
            g_idx = start_idx + idx
            for key, val in gm.items():
                buf = io.BytesIO()
                np.save(buf, val)
                zipf.writestr(f"graph_{g_idx:06d}_{key}.npy", buf.getvalue())


def flush_to_disk(
    all_features,
    all_labels,
    all_configs,
    all_metadata,
    features_bin_path,
    labels_bin_path,
    config_bin_path,
    metadata_path,
    start_graph_idx,
):
    if not all_features:
        return

    batch_features = np.vstack(all_features).astype(np.float32)
    batch_labels = np.hstack(all_labels).astype(np.float32)

    with open(features_bin_path, 'ab') as f_bin:
        f_bin.write(batch_features.tobytes())
    with open(labels_bin_path, 'ab') as l_bin:
        l_bin.write(batch_labels.tobytes())

    # Expand per-graph config to per-user rows and flush
    tiled = []
    for cfg_sum, n_users in all_configs:
        tiled.append(np.tile(cfg_sum, (n_users, 1)))
    batch_config = np.vstack(tiled)  # object array — use pickle/npy format
    buf = batch_config.tobytes()  # won't work for object — use np.save
    import io
    buf = io.BytesIO()
    np.save(buf, batch_config)
    with open(config_bin_path, 'ab') as c_bin:
        length = len(buf.getvalue())
        c_bin.write(length.to_bytes(8, 'little'))  # prefix with chunk length
        c_bin.write(buf.getvalue())

    append_metadata_to_zip(metadata_path, all_metadata, start_graph_idx)

    n_saved = len(batch_labels)
    all_features.clear()
    all_labels.clear()
    all_configs.clear()
    all_metadata.clear()
    del batch_features, batch_labels

    print(f"  [FLUSHED] {n_saved} users to temp binary files")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate LDP attack detection training dataset (v2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory path'
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
        help='Override number of users'
    )

    parser.add_argument(
        '--domain',
        type=int,
        default=None,
        help='Override domain size'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='Random seed'
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

    parser.add_argument(
        '--robust-iterations',
        type=int,
        default=2,
        help='Number of iterative robust re-estimation rounds for positional features'
    )

    parser.add_argument(
        '--save-every',
        type=int,
        default=20,
        help='Flush accumulated results to disk every N completed tasks'
    )

    return parser.parse_args()



def main():
    """Main entry point."""
    args = parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    tasks = build_tasks(args)

    # Separate OLH tasks (sequential) from non-OLH tasks (parallel)
    olh_protocols = {'OLH', 'OLH_User', 'OLH_Server'}
    parallel_tasks = [t for t in tasks if t['protocol_label'] not in olh_protocols]
    sequential_tasks = [t for t in tasks if t['protocol_label'] in olh_protocols]

    total_runs = len(tasks)

    print("=" * 80)
    print("LDP Attack Detection Dataset Generator")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Protocols: {args.protocols}")
    print(f"Epsilons: {args.epsilons}")
    print(f"Datasets: {args.datasets}")
    print(f"Ratios: {args.ratios}")
    print(f"Target sizes: {args.target_sizes}")
    print(f"Splits: {args.splits}")
    print(f"Experiments per config: {args.experiments}")
    print(f"Total experiment runs: {total_runs}")
    print(f"  Parallel tasks (OUE/HST): {len(parallel_tasks)}")
    print(f"  Sequential tasks (OLH):   {len(sequential_tasks)}")
    print(f"Outer workers: {args.workers}")
    print(f"Inner processors per task: {args.inner_processors}")
    print(f"Robust iterations: {args.robust_iterations}")
    print(f"Feature count: {len(FEATURE_NAMES)}")
    print(f"Features: {FEATURE_NAMES}")
    print("=" * 80)

    # --- Accumulation state ---
    all_features = []
    all_labels = []
    all_configs = []
    all_metadata = []
    total_users = 0
    total_attackers = 0
    num_success = 0
    num_failed = 0
    graph_index = 0
    SAVE_EVERY = args.save_every

    features_path = os.path.join(output_dir, 'features.npy')
    labels_path = os.path.join(output_dir, 'labels.npy')
    config_path = os.path.join(output_dir, 'config.npy')
    metadata_path = os.path.join(output_dir, 'metadata.npz')

    # Clear any previous partial output
    features_bin_path = os.path.join(output_dir, 'features.bin')
    labels_bin_path = os.path.join(output_dir, 'labels.bin')
    config_bin_path = os.path.join(output_dir, 'config.bin')
    for p_clear in [features_path, labels_path, config_path, metadata_path,
                    features_bin_path, labels_bin_path, config_bin_path]:
        if os.path.exists(p_clear):
            os.remove(p_clear)

    def _handle_result(result):
        nonlocal total_users, total_attackers, num_success, num_failed, graph_index

        if result['ok']:
            all_features.append(result['features'])
            all_labels.append(result['labels'])
            all_configs.append((result['config_summary'], result['num_users']))
            total_users += result['num_users']
            total_attackers += result['num_attackers']
            num_success += 1

            # Save per-graph metadata
            gm = result['graph_meta']
            all_metadata.append({
                'pi_hat': gm['pi_hat'],
                'pi_true': gm['pi_true'],
                'p': np.float64(gm['p']),
                'q': np.float64(gm['q']),
                'epsilon': np.float64(gm['epsilon']),
                'protocol': str(gm['protocol']),
                'support_tensor': gm['support_tensor'],
            })
            graph_index += 1

            print(
                f'[DONE] {result["desc"]} | '
                f'users={result["num_users"]}, attackers={result["num_attackers"]}'
            )
        else:
            num_failed += 1
            print(f'[FAIL] {result["desc"]} | error={result["error"]}')
            print(result['traceback'])

    if parallel_tasks:
        print(f"\n--- Phase 1: {len(parallel_tasks)} OUE/HST tasks in parallel ---")

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
                    desc=f"Parallel [{batch_start+1}-{batch_end}]"
                ):
                    try:
                        result = future.result()
                        _handle_result(result)
                    except Exception as e:
                        task_info = futures[future]
                        num_failed += 1
                        print(f'[CRASH] {task_info["desc"]} | Worker killed: {e}')

                    if len(all_features) >= SAVE_EVERY:
                        flush_to_disk(
                            all_features, all_labels, all_configs, all_metadata,
                            features_bin_path, labels_bin_path, config_bin_path, metadata_path,
                            graph_index - len(all_metadata)
                        )

            if all_features:
                flush_to_disk(
                    all_features, all_labels, all_configs, all_metadata,
                    features_bin_path, labels_bin_path, config_bin_path, metadata_path,
                    graph_index - len(all_metadata)
                )

    if sequential_tasks:
        print(f"\n--- Phase 2: Processing {len(sequential_tasks)} tasks sequentially ---")
        for task in tqdm(sequential_tasks, desc="Sequential (OLH)"):
            result = run_one_task(task)
            _handle_result(result)

            if len(all_features) >= SAVE_EVERY:
                flush_to_disk(
                    all_features, all_labels, all_configs, all_metadata,
                    features_bin_path, labels_bin_path, config_bin_path, metadata_path,
                    graph_index - len(all_metadata)
                )

    if all_features:
        flush_to_disk(
            all_features, all_labels, all_configs, all_metadata,
            features_bin_path, labels_bin_path, config_bin_path, metadata_path,
            graph_index - len(all_metadata)
        )

    # Reconstruct features, labels, and configs
    if os.path.exists(features_bin_path):
        print("\nComputing global z-score normalization statistics...")
        norm_path = os.path.join(output_dir, 'norm_stats.json')
        features_all = np.fromfile(features_bin_path, dtype=np.float32).reshape(total_users, 18).astype(np.float64)
        feat_mean = np.mean(features_all, axis=0)
        feat_std = np.std(features_all, axis=0)
        near_zero = feat_std < 1e-8
        feat_std[near_zero] = 1.0  

        # Normalize and save features.npy
        features_normed = (features_all - feat_mean) / feat_std
        np.save(features_path, features_normed.astype(np.float32))

        # Reconstruct labels.npy
        labels_all = np.fromfile(labels_bin_path, dtype=np.float32)
        np.save(labels_path, labels_all)

        # Save normalization stats
        norm_stats = {
            'feature_names': FEATURE_NAMES,
            'mean': feat_mean.tolist(),
            'std': feat_std.tolist(),
            'near_zero_variance_columns': [
                FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if near_zero[i]
            ],
        }
        with open(norm_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        print(f"  Saved normalization stats to {norm_path}")
        if any(near_zero):
            print(f"  WARNING: Near-zero-variance columns: {norm_stats['near_zero_variance_columns']}")

        del features_all, features_normed, labels_all

        if os.path.exists(features_bin_path):
            os.remove(features_bin_path)
        if os.path.exists(labels_bin_path):
            os.remove(labels_bin_path)

    if os.path.exists(config_bin_path):
        print("\nReconstructing config.npy from binary chunks...")
        import io
        chunks = []
        with open(config_bin_path, 'rb') as c_bin:
            while True:
                length_bytes = c_bin.read(8)
                if not length_bytes or len(length_bytes) < 8:
                    break
                length = int.from_bytes(length_bytes, 'little')
                chunk_bytes = c_bin.read(length)
                chunk = np.load(io.BytesIO(chunk_bytes), allow_pickle=True)
                chunks.append(chunk)
        if chunks:
            config_all = np.vstack(chunks)
            np.save(config_path, config_all)
            print(f"  Saved config.npy: {config_all.shape}")
            del config_all, chunks
        os.remove(config_bin_path)

    print("\n" + "=" * 80)
    print("Dataset Generation Complete!")
    print("=" * 80)
    print(f"Successful runs: {num_success}")
    print(f"Failed runs: {num_failed}")
    print(f"Total graphs: {graph_index}")
    print(f"Total users: {total_users:,}")
    print(f"Total attackers: {total_attackers:,}")

    if total_users > 0:
        benign = total_users - total_attackers
        print(f"Benign: {benign:,} ({100.0 * benign / total_users:.2f}%)")
        print(f"Attackers: {total_attackers:,} ({100.0 * total_attackers / total_users:.2f}%)")

    print(f"\nOutput directory: {output_dir}")
    for fname in ['features.npy', 'labels.npy', 'config.npy', 'norm_stats.json', 'metadata.npz']:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fname}: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
