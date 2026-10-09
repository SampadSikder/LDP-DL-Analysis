#!/usr/bin/env python

import argparse
import json
import math
import os
import sys
import traceback
import zlib
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
    extract_user_level_features_diffstats_style,
    FEATURE_SETS,
    DEFAULT_BLOCK_CANDIDATES,
    feature_names_for,
)


OLH_PROTOCOLS = {'OLH', 'OLH_User', 'OLH_Server'}

# --attack -> the generator's h_ao switch. HST_Server ignores it and always runs
# MGA-A: the server fixes the count of +1s, so there is no count to shape.
#   mga-a       h_ao=0: every fake user reports exactly the expected number of
#               1s -- the paper's MGA-A (Sec. 3.2 / 4.1.2), what Diffstats is
#               scored against in its Figure 3.
#   apa         h_ao=2: the paper's optimal APA (Sec. 4.1.4) for OUE and
#               HST_User -- exactly floor(m * P(X=k)) fake users report k 1s,
#               no jitter, so the count histogram matches genuine users.
#               OLH_Server (not in the paper): server-side APA, the same
#               histogram reached by choosing among each fake user's g buckets
#               (attacks.choose_server_apa_buckets).
#   apa-approx  h_ao=1: the legacy setting. OUE / OLH_User draw each fake
#               user's count from the genuine distribution (APA-like) with
#               +-10 jitter; HST_User jitters the MGA-A count by +-10.
ATTACK_H_AO = {'mga-a': 0, 'apa-approx': 1, 'apa': 2}


def _resolve_protocol(protocol: str):
    """protocol label -> (base protocol passed to the generator, OLH setting)."""
    if protocol == "OLH_User":
        return "OLH", "user"
    if protocol == "OLH_Server":
        return "OLH", "server"
    return protocol, "server"


def _make_task(args, *, epsilon, dataset_type, domain, n, protocol, ratio,
               target_size, splits, exp_i, seed, holdout):
    base_protocol, olh_setting = _resolve_protocol(protocol)
    return {
        'epsilon': epsilon,
        'domain': domain,
        'n': n,
        'protocol': base_protocol,
        'protocol_label': protocol,
        'ratio': ratio,
        'target_set_size': target_size,
        'splits': splits,
        'dataset_type': dataset_type,
        'h_ao': ATTACK_H_AO[args.attack],
        'seed': seed,
        'inner_processors': args.inner_processors,  # not nested so use multi core
        'olh_setting': olh_setting,
        'exp_i': exp_i,
        'holdout': holdout,
        'feature_set': args.feature_set,
        'block_candidates': args.block_candidates,
        'desc': (
            f"ε={epsilon}, {dataset_type}, {protocol}, "
            f"ratio={ratio}, target={target_size}, "
            f"splits={splits}, exp={exp_i + 1}"
            + (", holdout" if holdout else "")
        ),
    }


def oat_configs(args) -> list:
    """(epsilon, ratio, target_size, splits) points of a one-at-a-time design.
    """
    d = (args.default_epsilon, args.default_ratio,
         args.default_target_size, args.default_splits)
    points, seen = [], set()

    def add(point):
        key = (round(float(point[0]), 9), round(float(point[1]), 9),
               int(point[2]), int(point[3]))
        if key in seen:
            return
        if key[3] > key[2]:
            print(f"  [WARN] skipping OAT point {key}: splits > target_set_size")
            return
        seen.add(key)
        points.append(key)

    for eps in args.epsilons:
        add((eps, d[1], d[2], d[3]))
    for ratio in args.ratios:
        add((d[0], ratio, d[2], d[3]))
    for target in args.target_sizes:
        add((d[0], d[1], target, d[3]))
    for splits in args.splits:
        add((d[0], d[1], d[2], splits))

    center = (round(float(d[0]), 9), round(float(d[1]), 9), int(d[2]), int(d[3]))
    if center not in seen:
        print(f"  [WARN] the default point {center} is not in any sweep list, so the "
              f"sweeps share no common point. Add each default to its own list.")
    return points


def oat_replicates(args, ratio: float, n: int) -> int:
    """Training replicates for one OAT config.
    attackers: ceil(A / (ratio * n)).
    """
    if args.balance_attackers:
        return max(1, math.ceil(args.balance_attackers / (ratio * n)))
    return args.experiments


def _oat_seed(base: int, dataset_type, protocol, epsilon, ratio, target, splits, rep) -> int:
    key = f"{dataset_type}|{protocol}|{epsilon!r}|{ratio!r}|{target}|{splits}|{rep}"
    return (base + zlib.crc32(key.encode())) % (2 ** 32 - 1)


def build_tasks(args) -> list:
    """Build list of task dicts for all experiment configurations."""
    configs = DATASET_CONFIGS_FULL if args.full_scale else DATASET_CONFIGS
    tasks = []

    if args.design == 'oat':
        points = oat_configs(args)
        for dataset_type in args.datasets:
            dataset_config = configs[dataset_type]
            domain = args.domain if args.domain else dataset_config['domain']
            n = args.n if args.n else dataset_config['n']

            for protocol in args.protocols:
                for epsilon, ratio, target_size, splits in points:
                    n_train = oat_replicates(args, ratio, n)
                    for exp_i in range(n_train + args.holdout_replicates):
                        tasks.append(_make_task(
                            args,
                            epsilon=epsilon, dataset_type=dataset_type,
                            domain=domain, n=n, protocol=protocol, ratio=ratio,
                            target_size=target_size, splits=splits, exp_i=exp_i,
                            seed=_oat_seed(args.seed, dataset_type, protocol,
                                           epsilon, ratio, target_size, splits, exp_i),
                            holdout=int(exp_i >= n_train),
                        ))
        return tasks

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
                                config_idx = len(tasks) + 1
                                tasks.append(_make_task(
                                    args,
                                    epsilon=epsilon, dataset_type=dataset_type,
                                    domain=domain, n=n, protocol=protocol,
                                    ratio=ratio, target_size=target_size,
                                    splits=splits, exp_i=exp_i,
                                    seed=args.seed + config_idx * 1000 + exp_i,
                                    holdout=0,
                                ))

    return tasks



def run_one_task(task: dict) -> dict:
    """
    Execute one experiment configuration:
      1. Generate perturbed data via generate_perturbed_data()
      2. Extract features
      3. Return result dict with features, labels and the per-row config summary
    """
    try:
        support_list, labels, _real_dist, _estimate_dist, one_list = generate_perturbed_data(
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

        features = extract_user_level_features_diffstats_style(
            support_list=support_list,
            one_list=one_list,
            epsilon=task['epsilon'],
            protocol=task['protocol_label'],
            domain=task['domain'],
            n=task['n'],
            feature_set=task['feature_set'],
            block_candidates=task['block_candidates'],
        )

        # experiment_id is appended by _handle_result (main process), which is
        # the only place the final sequential index across all tasks is known.
        config_summary = [
            task['target_set_size'], task['ratio'], task['protocol_label'],
            task['splits'], task['epsilon'], task['dataset_type'], task['n'],
        ]

        return {
            'ok': True,
            'features': features,
            'labels': labels,
            'config_summary': config_summary,
            'holdout': task['holdout'],
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


def write_design(args, tasks: list, output_dir: str) -> None:
    runs = {}
    for t in tasks:
        key = (t['dataset_type'], t['protocol_label'], t['epsilon'], t['ratio'],
               t['target_set_size'], t['splits'])
        entry = runs.setdefault(key, {'train': 0, 'holdout': 0})
        entry['holdout' if t['holdout'] else 'train'] += 1

    design = {
        'design': args.design,
        'attack': args.attack,
        'feature_set': args.feature_set,
        'feature_names': feature_names_for(args.feature_set),
        'block_candidates': (args.block_candidates
                             if args.feature_set == 'v2' else None),
        'protocols': list(args.protocols),
        'datasets': list(args.datasets),
        'n': args.n,
        'sweeps': {
            'epsilon': [float(v) for v in args.epsilons],
            'attacker_ratio': [float(v) for v in args.ratios],
            'target_set_size': [int(v) for v in args.target_sizes],
            'splits': [int(v) for v in args.splits],
        },
        'runs': [
            {'dataset_type': k[0], 'protocol': k[1], 'epsilon': k[2],
             'attacker_ratio': k[3], 'target_set_size': k[4], 'splits': k[5], **v}
            for k, v in runs.items()
        ],
    }
    if args.design == 'oat':
        design['defaults'] = {
            'epsilon': float(args.default_epsilon),
            'attacker_ratio': float(args.default_ratio),
            'target_set_size': int(args.default_target_size),
            'splits': int(args.default_splits),
        }
        design['balance_attackers'] = args.balance_attackers
        design['holdout_replicates'] = args.holdout_replicates

    with open(os.path.join(output_dir, 'design.json'), 'w') as f:
        json.dump(design, f, indent=2)


def _config_chunks(config_bin_path, headers_only=False):
    """Yield the length-prefixed .npy chunks of config.bin -- as (shape, dtype)
    from the header alone, or as loaded arrays."""
    import io
    with open(config_bin_path, 'rb') as c_bin:
        while True:
            length_bytes = c_bin.read(8)
            if len(length_bytes) < 8:
                return
            length = int.from_bytes(length_bytes, 'little')
            start = c_bin.tell()
            if headers_only:
                version = np.lib.format.read_magic(c_bin)
                read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                               else np.lib.format.read_array_header_2_0)
                shape, _, dtype = read_header(c_bin)
                yield shape, dtype
                c_bin.seek(start + length)
            else:
                yield np.load(io.BytesIO(c_bin.read(length)), allow_pickle=True)


def assemble_config(config_bin_path, config_path):
    """config.bin -> config.npy, streaming into a memory-mapped output so only
    one chunk is ever in RAM (np.vstack of all chunks needed ~2x the file).

    Pass 1 reads only the chunk headers to size the output; the dtype is the
    widest chunk's, which is what np.vstack would have produced, so the file
    is the same. Returns the shape, or None if there were no chunks."""
    metas = list(_config_chunks(config_bin_path, headers_only=True))
    if not metas:
        return None
    rows = sum(shape[0] for shape, _ in metas)
    cols = metas[0][0][1]
    dtype = np.result_type(*[dt for _, dt in metas])
    out = np.lib.format.open_memmap(config_path, mode='w+', dtype=dtype, shape=(rows, cols))
    row = 0
    for chunk in _config_chunks(config_bin_path):
        out[row:row + len(chunk)] = chunk
        row += len(chunk)
    out.flush()
    del out
    return (rows, cols)


def flush_to_disk(
    all_features,
    all_labels,
    all_configs,
    features_bin_path,
    labels_bin_path,
    config_bin_path,
):
    if not all_features:
        return

    batch_features = np.vstack(all_features).astype(np.float32)
    batch_labels = np.hstack(all_labels).astype(np.float32)

    with open(features_bin_path, 'ab') as f_bin:
        f_bin.write(batch_features.tobytes())
    with open(labels_bin_path, 'ab') as l_bin:
        l_bin.write(batch_labels.tobytes())

    # Expand per-experiment config to per-user rows and flush. row_in_experiment
    # (0..n_users-1) is the one column that varies within an experiment, so it's
    tiled = []
    for cfg_sum, n_users, holdout in all_configs:
        tiled_rows = np.tile(np.array([str(v) for v in cfg_sum]), (n_users, 1))
        # astype(str) on ints is always U21; size it to the largest index instead
        row_in_experiment = np.arange(n_users).astype(
            f'<U{len(str(max(n_users - 1, 0)))}').reshape(-1, 1)
        holdout_col = np.full((n_users, 1), str(int(holdout)))
        tiled.append(np.hstack([tiled_rows, row_in_experiment, holdout_col]))
    batch_config = np.vstack(tiled)
    import io
    buf = io.BytesIO()
    np.save(buf, batch_config)
    with open(config_bin_path, 'ab') as c_bin:
        length = len(buf.getvalue())
        c_bin.write(length.to_bytes(8, 'little'))  # prefix with chunk length
        c_bin.write(buf.getvalue())

    n_saved = len(batch_labels)
    all_features.clear()
    all_labels.clear()
    all_configs.clear()
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
        '--save-every',
        type=int,
        default=20,
        help='Flush accumulated results to disk every N completed tasks'
    )

    parser.add_argument(
        '--attack',
        choices=sorted(ATTACK_H_AO),
        default='apa-approx',
        help='mga-a = the paper\'s MGA-A (fixed count of 1s per fake user); '
             'apa = the paper\'s optimal APA (OUE, HST_User); '
             'apa-approx = legacy h_ao=1 behaviour (default, reproduces existing data)'
    )

    parser.add_argument(
        '--feature-set',
        choices=sorted(FEATURE_SETS),
        default='v1',
        help='v1 = the original 16 features; v2 = v1 minus 4 redundant analytic '
             'k-features plus 2 target-block features (14 columns)'
    )

    parser.add_argument(
        '--block-candidates',
        type=int,
        default=DEFAULT_BLOCK_CANDIDATES,
        help='v2 only: highest-z items searched for the coordinated target block'
    )

    parser.add_argument(
        '--tasks-per-worker',
        type=int,
        default=50,
        help='Tasks each worker process runs before the pool is recreated. Large n '
             '(hundreds of thousands of users): use 1-2 so worker memory is released '
             'between tasks'
    )

    parser.add_argument(
        '--olh-parallel',
        action='store_true',
        help='Run OLH tasks in the outer process pool with --inner-processors 1'
    )

    oat = parser.add_argument_group(
        'one-at-a-time design',
        'With --design oat, each of --epsilons / --ratios / --target-sizes / '
        '--splits is swept on its own while the other parameters sit at their '
        '--default-* values, instead of taking the full cross product.'
    )
    oat.add_argument('--design', choices=['grid', 'oat'], default='grid',
                     help='grid = full cross product (original behaviour); '
                          'oat = one-at-a-time sweeps around the defaults')
    oat.add_argument('--default-epsilon', type=float, default=1.0)
    oat.add_argument('--default-ratio', type=float, default=0.05)
    oat.add_argument('--default-target-size', type=int, default=10)
    oat.add_argument('--default-splits', type=int, default=4)
    oat.add_argument(
        '--balance-attackers', type=int, default=None,
        help='Training replicates per config = ceil(A / (ratio * n)), so every '
             'config supplies at least A attackers. Overrides --experiments.'
    )
    oat.add_argument(
        '--holdout-replicates', type=int, default=0,
        help='Extra replicates per config marked holdout=1 (config column 9); '
             'main.py never trains on them'
    )

    args = parser.parse_args()
    if args.design == 'grid' and (args.balance_attackers or args.holdout_replicates):
        parser.error('--balance-attackers / --holdout-replicates require --design oat')
    if args.balance_attackers is not None and args.balance_attackers < 1:
        parser.error('--balance-attackers must be a positive integer')
    if args.holdout_replicates < 0:
        parser.error('--holdout-replicates must be >= 0')
    if args.tasks_per_worker < 1:
        parser.error('--tasks-per-worker must be >= 1')
    if args.block_candidates < 2:
        parser.error('--block-candidates must be >= 2')
    return args



def main():
    """Main entry point."""
    args = parse_args()
    feature_names = feature_names_for(args.feature_set)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    tasks = build_tasks(args)

    if args.olh_parallel:
        for t in tasks:
            if t['protocol_label'] in OLH_PROTOCOLS:
                t['inner_processors'] = 1
        parallel_tasks, sequential_tasks = tasks, []
    else:
        parallel_tasks = [t for t in tasks if t['protocol_label'] not in OLH_PROTOCOLS]
        sequential_tasks = [t for t in tasks if t['protocol_label'] in OLH_PROTOCOLS]

    total_runs = len(tasks)
    write_design(args, tasks, output_dir)

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
    print(f"Attack: {args.attack} (h_ao={ATTACK_H_AO[args.attack]})")
    print(f"Design: {args.design}")
    if args.design == 'oat':
        print(f"  Defaults: eps={args.default_epsilon}, ratio={args.default_ratio}, "
              f"target={args.default_target_size}, splits={args.default_splits}")
        print(f"  Configs per (dataset, protocol): {len(oat_configs(args))}")
        print(f"  Balance attackers: {args.balance_attackers or 'off'}  "
              f"Holdout replicates: {args.holdout_replicates}")
        n_hold = sum(t['holdout'] for t in tasks)
        print(f"  Train runs: {total_runs - n_hold}  Holdout runs: {n_hold}")
    else:
        print(f"Experiments per config: {args.experiments}")
    print(f"Total experiment runs: {total_runs}")
    print(f"  Parallel tasks:   {len(parallel_tasks)}")
    print(f"  Sequential tasks: {len(sequential_tasks)}")
    print(f"Outer workers: {args.workers}")
    print(f"Inner processors per task: {args.inner_processors}")
    print(f"Feature set: {args.feature_set}"
          + (f" (block candidates: {args.block_candidates})" if args.feature_set == 'v2' else ""))
    print(f"Feature count: {len(feature_names)}")
    print(f"Features: {feature_names}")
    print("=" * 80)

    # --- Accumulation state ---
    all_features = []
    all_labels = []
    all_configs = []
    total_users = 0
    total_attackers = 0
    num_success = 0
    num_failed = 0
    graph_index = 0
    SAVE_EVERY = args.save_every

    features_path = os.path.join(output_dir, 'features.npy')
    labels_path = os.path.join(output_dir, 'labels.npy')
    config_path = os.path.join(output_dir, 'config.npy')
    stale_metadata_path = os.path.join(output_dir, 'metadata.npz')

    features_bin_path = os.path.join(output_dir, 'features.bin')
    labels_bin_path = os.path.join(output_dir, 'labels.bin')
    config_bin_path = os.path.join(output_dir, 'config.bin')
    for p_clear in [features_path, labels_path, config_path, stale_metadata_path,
                    features_bin_path, labels_bin_path, config_bin_path]:
        if os.path.exists(p_clear):
            os.remove(p_clear)

    def _handle_result(result):
        nonlocal total_users, total_attackers, num_success, num_failed, graph_index

        if result['ok']:
            # experiment_id is assigned here, in the main process, because this
            # is the only place the final sequential index across all (possibly
            # out-of-order-completing) tasks is known.
            experiment_id = graph_index
            config_summary = list(result['config_summary']) + [experiment_id]

            all_features.append(result['features'])
            all_labels.append(result['labels'])
            all_configs.append((config_summary, result['num_users'], result['holdout']))
            total_users += result['num_users']
            total_attackers += result['num_attackers']
            num_success += 1
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

        # A fresh process pool per batch: worker memory is returned to the OS
        # every --tasks-per-worker tasks instead of ratcheting up for the run.
        batch_size = max(1, args.workers * args.tasks_per_worker)
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
                    # pop so the finished future -- and the result it holds -- can
                    # be freed now, not when the whole batch ends
                    task_info = futures.pop(future)
                    try:
                        result = future.result()
                        _handle_result(result)
                    except Exception as e:
                        num_failed += 1
                        print(f'[CRASH] {task_info["desc"]} | Worker killed: {e}')
                    result = None

                    if len(all_features) >= SAVE_EVERY:
                        flush_to_disk(
                            all_features, all_labels, all_configs,
                            features_bin_path, labels_bin_path, config_bin_path,
                        )

            if all_features:
                flush_to_disk(
                    all_features, all_labels, all_configs,
                    features_bin_path, labels_bin_path, config_bin_path,
                )

    if sequential_tasks:
        print(f"\n--- Phase 2: Processing {len(sequential_tasks)} tasks sequentially ---")
        for task in tqdm(sequential_tasks, desc="Sequential (OLH)"):
            result = run_one_task(task)
            _handle_result(result)

            if len(all_features) >= SAVE_EVERY:
                flush_to_disk(
                    all_features, all_labels, all_configs,
                    features_bin_path, labels_bin_path, config_bin_path,
                )

    if all_features:
        flush_to_disk(
            all_features, all_labels, all_configs,
            features_bin_path, labels_bin_path, config_bin_path,
        )

    # Reconstruct features, labels, and configs
    if os.path.exists(features_bin_path):
        print("\nComputing global z-score normalization statistics...")
        norm_path = os.path.join(output_dir, 'norm_stats.json')
        features_all = np.fromfile(features_bin_path, dtype=np.float32).reshape(
            total_users, len(feature_names)
        ).astype(np.float64)
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
            'feature_names': feature_names,
            'mean': feat_mean.tolist(),
            'std': feat_std.tolist(),
            'near_zero_variance_columns': [
                feature_names[i] for i in range(len(feature_names)) if near_zero[i]
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
        shape = assemble_config(config_bin_path, config_path)
        if shape is not None:
            print(f"  Saved config.npy: {shape}")
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
    for fname in ['features.npy', 'labels.npy', 'config.npy', 'norm_stats.json', 'design.json']:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fname}: {size_mb:.2f} MB")


if __name__ == '__main__':
    main()
