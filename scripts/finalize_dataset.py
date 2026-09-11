
import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacker_detector.data.generators import FEATURE_NAMES  # noqa: E402


def finalize(data_dir: str):
    features_bin_path = os.path.join(data_dir, 'features.bin')
    labels_bin_path = os.path.join(data_dir, 'labels.bin')
    config_bin_path = os.path.join(data_dir, 'config.bin')

    features_path = os.path.join(data_dir, 'features.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    config_path = os.path.join(data_dir, 'config.npy')
    norm_path = os.path.join(data_dir, 'norm_stats.json')

    if not os.path.exists(labels_bin_path) or not os.path.exists(features_bin_path):
        raise FileNotFoundError(f"features.bin / labels.bin not found in {data_dir}")

    n_features = len(FEATURE_NAMES)

    # --- labels ---
    labels_all = np.fromfile(labels_bin_path, dtype=np.float32)
    total_users = len(labels_all)
    print(f"labels.bin: {total_users:,} users")

    # --- features ---
    raw = np.fromfile(features_bin_path, dtype=np.float32)
    n_feature_rows = len(raw) // n_features
    if len(raw) % n_features != 0:
        print(f"  [WARN] features.bin size not a multiple of {n_features} features; "
              f"dropping {len(raw) % n_features} trailing floats")
    if n_feature_rows != total_users:
        print(f"  [WARN] features.bin has {n_feature_rows:,} rows but labels.bin has "
              f"{total_users:,}; truncating both to {min(n_feature_rows, total_users):,} "
              f"(the interrupted run likely stopped mid-flush)")
        total_users = min(n_feature_rows, total_users)
        labels_all = labels_all[:total_users]

    print("Computing global z-score normalization statistics...")
    features_all = raw[:total_users * n_features].reshape(total_users, n_features).astype(np.float64)
    del raw

    feat_mean = np.mean(features_all, axis=0)
    feat_std = np.std(features_all, axis=0)
    near_zero = feat_std < 1e-8
    feat_std[near_zero] = 1.0

    features_normed = (features_all - feat_mean) / feat_std
    np.save(features_path, features_normed.astype(np.float32))
    del features_all, features_normed

    np.save(labels_path, labels_all)

    norm_stats = {
        'feature_names': FEATURE_NAMES,
        'mean': feat_mean.tolist(),
        'std': feat_std.tolist(),
        'near_zero_variance_columns': [
            FEATURE_NAMES[i] for i in range(n_features) if near_zero[i]
        ],
    }
    with open(norm_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved {features_path}, {labels_path}, {norm_path}")
    if any(near_zero):
        print(f"  WARNING: Near-zero-variance columns: {norm_stats['near_zero_variance_columns']}")

    # --- config ---
    if os.path.exists(config_bin_path):
        print("\nReconstructing config.npy from binary chunks...")
        chunks = []
        rows_read = 0
        with open(config_bin_path, 'rb') as c_bin:
            while True:
                length_bytes = c_bin.read(8)
                if not length_bytes or len(length_bytes) < 8:
                    break
                length = int.from_bytes(length_bytes, 'little')
                chunk_bytes = c_bin.read(length)
                if len(chunk_bytes) < length:
                    print("  [WARN] Final config.bin chunk is truncated (interrupted mid-write); dropping it")
                    break
                try:
                    chunk = np.load(io.BytesIO(chunk_bytes), allow_pickle=True)
                except Exception as e:
                    print(f"  [WARN] Could not parse a config.bin chunk ({e}); dropping remaining data")
                    break
                chunks.append(chunk)
                rows_read += chunk.shape[0]

        if chunks:
            config_all = np.vstack(chunks)
            del chunks
            if len(config_all) != total_users:
                print(f"  [WARN] config.bin has {len(config_all):,} rows but expected {total_users:,}; "
                      f"truncating to {min(len(config_all), total_users):,}")
                n = min(len(config_all), total_users)
                config_all = config_all[:n]
            np.save(config_path, config_all)
            print(f"  Saved config.npy: {config_all.shape}")
        else:
            print("  [WARN] No usable config.bin chunks found; config.npy not written")
    else:
        print("\n[WARN] No config.bin found; config.npy not written "
              "(sensitivity analysis by epsilon/ratio/dataset_type will be unavailable)")

    print("\nDone. You can now delete the .bin files if you want to reclaim disk space:")
    print(f"  rm {features_bin_path} {labels_bin_path} {config_bin_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', help='Directory containing features.bin/labels.bin/config.bin from an interrupted run')
    args = parser.parse_args()
    finalize(args.data_dir)


if __name__ == '__main__':
    main()
