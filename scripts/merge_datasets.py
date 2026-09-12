import argparse
import json
import os
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacker_detector.data.dataset import (  # noqa: E402
    _CONF_EXPERIMENT_ID,
)

CHUNK_ROWS_DEFAULT = 2_000_000


def load_norm_stats(data_dir: str) -> dict:
    with open(os.path.join(data_dir, 'norm_stats.json')) as f:
        return json.load(f)


def validate_inputs(dirs: list) -> list:
    feature_names_ref = None
    for d in dirs:
        for fname in ['features.npy', 'labels.npy', 'config.npy', 'norm_stats.json']:
            if not os.path.exists(os.path.join(d, fname)):
                raise FileNotFoundError(
                    f"{d} is missing {fname} -- run finalize_dataset.py on it first"
                )
        stats = load_norm_stats(d)
        if feature_names_ref is None:
            feature_names_ref = stats['feature_names']
        elif stats['feature_names'] != feature_names_ref:
            raise ValueError(
                f"{d} has different feature_names than {dirs[0]} -- "
                f"these datasets were generated with different feature extraction code "
                f"and cannot be merged."
            )
    return feature_names_ref


def compute_global_mean_std(dirs: list, feature_names: list, chunk_rows: int):
    n_features = len(feature_names)
    total_sum = np.zeros(n_features, dtype=np.float64)
    total_sumsq = np.zeros(n_features, dtype=np.float64)
    total_n = 0

    for d in dirs:
        stats = load_norm_stats(d)
        mean = np.array(stats['mean'], dtype=np.float64)
        std = np.array(stats['std'], dtype=np.float64)
        feats = np.load(os.path.join(d, 'features.npy'), mmap_mode='r')
        n_rows = feats.shape[0]
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            chunk = np.asarray(feats[start:end], dtype=np.float64)
            raw = chunk * std + mean
            total_sum += raw.sum(axis=0)
            total_sumsq += (raw ** 2).sum(axis=0)
        total_n += n_rows
        print(f"  scanned {d}: {n_rows:,} rows")

    mean = total_sum / total_n
    var = np.clip(total_sumsq / total_n - mean ** 2, 0, None)
    std = np.sqrt(var)
    near_zero = std < 1e-8
    std[near_zero] = 1.0
    return mean, std, near_zero, total_n


def write_merged_features(dirs: list, out_path: str, mean, std, total_n: int, n_features: int, chunk_rows: int):
    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(total_n, n_features))
    row_offset = 0
    for d in dirs:
        stats = load_norm_stats(d)
        d_mean = np.array(stats['mean'], dtype=np.float64)
        d_std = np.array(stats['std'], dtype=np.float64)
        feats = np.load(os.path.join(d, 'features.npy'), mmap_mode='r')
        n_rows = feats.shape[0]
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            chunk = np.asarray(feats[start:end], dtype=np.float64)
            raw = chunk * d_std + d_mean
            normed = (raw - mean) / std
            out[row_offset + start: row_offset + end] = normed.astype(np.float32)
        row_offset += n_rows
    out.flush()
    del out


def merge_labels_and_config(dirs: list):
    all_labels = []
    all_configs = []
    experiment_offsets = []  # offset applied to each dir's experiment_id / graph index
    cumulative_offset = 0

    for d in dirs:
        labels = np.load(os.path.join(d, 'labels.npy'))
        config = np.load(os.path.join(d, 'config.npy'), allow_pickle=True)

        experiment_offsets.append(cumulative_offset)

        if config.shape[1] > _CONF_EXPERIMENT_ID:
            config = config.copy()
            config[:, _CONF_EXPERIMENT_ID] = (
                config[:, _CONF_EXPERIMENT_ID].astype(np.int64) + cumulative_offset
            )
            n_experiments = int(config[:, _CONF_EXPERIMENT_ID].astype(np.int64).max()) + 1 - cumulative_offset
            cumulative_offset += n_experiments
        else:
            print(f"  [WARN] {d}: config.npy has no experiment_id column; "
                  f"metadata.npz merging will not be offset-correct for it")

        all_labels.append(labels)
        all_configs.append(config)
        print(f"  loaded {d}: {len(labels):,} rows")

    merged_labels = np.concatenate(all_labels)
    merged_config = np.vstack(all_configs)
    return merged_labels, merged_config, experiment_offsets


def merge_metadata(dirs: list, experiment_offsets: list, out_path: str):
    paths = [os.path.join(d, 'metadata.npz') for d in dirs]
    if not all(os.path.exists(p) for p in paths):
        print("  [SKIP] Not all input directories have metadata.npz -- skipping metadata merge")
        return

    pattern = re.compile(r'^graph_(\d+)_(.+)$')
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
        for path, offset in zip(paths, experiment_offsets):
            with zipfile.ZipFile(path, 'r') as zin:
                for name in zin.namelist():
                    m = pattern.match(name)
                    if not m:
                        zout.writestr(name, zin.read(name))
                        continue
                    idx = int(m.group(1)) + offset
                    new_name = f"graph_{idx:06d}_{m.group(2)}"
                    zout.writestr(new_name, zin.read(name))
    print(f"  Saved {out_path}")


def merge(dirs: list, output_dir: str, chunk_rows: int):
    os.makedirs(output_dir, exist_ok=True)

    print("Validating inputs...")
    feature_names = validate_inputs(dirs)
    n_features = len(feature_names)

    print("\nPass 1/2: computing global mean/std over de-normalized features...")
    mean, std, near_zero, total_n = compute_global_mean_std(dirs, feature_names, chunk_rows)
    print(f"  Total rows: {total_n:,}")

    print("\nPass 2/2: writing merged, re-normalized features.npy...")
    write_merged_features(
        dirs, os.path.join(output_dir, 'features.npy'), mean, std, total_n, n_features, chunk_rows
    )

    print("\nMerging labels.npy and config.npy...")
    merged_labels, merged_config, experiment_offsets = merge_labels_and_config(dirs)
    np.save(os.path.join(output_dir, 'labels.npy'), merged_labels)
    np.save(os.path.join(output_dir, 'config.npy'), merged_config)
    print(f"  labels.npy: {merged_labels.shape}, config.npy: {merged_config.shape}")

    norm_stats = {
        'feature_names': feature_names,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'near_zero_variance_columns': [feature_names[i] for i in range(n_features) if near_zero[i]],
    }
    with open(os.path.join(output_dir, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f, indent=2)

    print("\nMerging metadata.npz...")
    merge_metadata(dirs, experiment_offsets, os.path.join(output_dir, 'metadata.npz'))

    print(f"\nDone. Merged dataset written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dirs', nargs='+', help='Two or more finalized dataset directories to merge')
    parser.add_argument('--output', '-o', required=True, help='Output directory for the merged dataset')
    parser.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS_DEFAULT,
                         help='Rows per streaming chunk when processing features.npy')
    args = parser.parse_args()

    if len(args.data_dirs) < 2:
        parser.error("Provide at least two dataset directories to merge")

    merge(args.data_dirs, args.output, args.chunk_rows)


if __name__ == '__main__':
    main()
