import argparse
import os
import shutil
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacker_detector.data.dataset import (  # noqa: E402
    _CONF_TARGET_SET_SIZE,
    _CONF_ATTACKER_RATIO,
    _CONF_PROTOCOL,
    _CONF_SPLITS,
    _CONF_EPSILON,
    _CONF_DATASET_TYPE,
)

CHUNK_ROWS_DEFAULT = 5_000_000
BATCH_ROWS_DEFAULT = 2_000_000
STRATA_COLS = [_CONF_EPSILON, _CONF_DATASET_TYPE, _CONF_ATTACKER_RATIO, _CONF_TARGET_SET_SIZE, _CONF_SPLITS]


def build_group_index(config_mm: np.ndarray, protocols: list, chunk_rows: int):
    """Scan config.npy once, return ({"<protocol>::<stratum_key>": index array}, {protocol: total_matched})."""
    groups = defaultdict(list)
    totals = defaultdict(int)
    protocol_set = set(protocols)
    n_rows = config_mm.shape[0]

    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        chunk = np.asarray(config_mm[start:end])  # materialize this chunk only
        mask = np.isin(chunk[:, _CONF_PROTOCOL], list(protocol_set))
        if not mask.any():
            continue
        local_idx = np.nonzero(mask)[0]
        sub = chunk[local_idx]
        global_idx = local_idx + start

        keys = sub[:, STRATA_COLS[0]]
        for col in STRATA_COLS[1:]:
            keys = np.char.add(np.char.add(keys, '|'), sub[:, col])
        combined_keys = np.char.add(np.char.add(sub[:, _CONF_PROTOCOL], '::'), keys)

        for key in np.unique(combined_keys):
            idxs = global_idx[combined_keys == key]
            groups[key].append(idxs)
            totals[key.split('::', 1)[0]] += len(idxs)

        print(f"  scanned {end:,}/{n_rows:,} rows, matched so far: {sum(totals.values()):,}")

    groups = {k: np.concatenate(v) for k, v in groups.items()}
    return groups, totals


def choose_sample_indices(groups: dict, totals: dict, protocols: list, n_target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for protocol in protocols:
        total_matched = totals.get(protocol, 0)
        if total_matched == 0:
            print(f"  [WARN] No rows found for protocol={protocol}; skipping")
            continue
        if total_matched < n_target:
            print(f"  [WARN] Only {total_matched:,} rows available for {protocol}, "
                  f"less than requested {n_target:,}; using all of them")

        proto_prefix = protocol + '::'
        for key, idx in groups.items():
            if not key.startswith(proto_prefix):
                continue
            target = round(n_target * len(idx) / total_matched)
            target = min(target, len(idx))
            if target == len(idx):
                chosen.append(idx)
            else:
                chosen.append(rng.choice(idx, size=target, replace=False))
        print(f"  {protocol}: selected up to {n_target:,} (of {total_matched:,} available)")

    result = np.concatenate(chosen)
    result.sort()
    return result


def write_sampled_dataset(data_dir: str, output_dir: str, idx: np.ndarray, batch_rows: int):
    os.makedirs(output_dir, exist_ok=True)

    features_mm = np.load(os.path.join(data_dir, 'features.npy'), mmap_mode='r')
    config_mm = np.load(os.path.join(data_dir, 'config.npy'), mmap_mode='r')
    n_features = features_mm.shape[1]
    n_cols = config_mm.shape[1]

    out_features = np.lib.format.open_memmap(
        os.path.join(output_dir, 'features.npy'), mode='w+', dtype=np.float32, shape=(len(idx), n_features)
    )
    out_config = np.lib.format.open_memmap(
        os.path.join(output_dir, 'config.npy'), mode='w+', dtype=config_mm.dtype, shape=(len(idx), n_cols)
    )
    for start in range(0, len(idx), batch_rows):
        end = min(start + batch_rows, len(idx))
        batch_idx = idx[start:end]
        out_features[start:end] = features_mm[batch_idx]
        out_config[start:end] = config_mm[batch_idx]
        print(f"  wrote {end:,}/{len(idx):,} sampled rows")
    out_features.flush()
    out_config.flush()
    del out_features, out_config

    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    np.save(os.path.join(output_dir, 'labels.npy'), labels[idx])

    shutil.copy(os.path.join(data_dir, 'norm_stats.json'), os.path.join(output_dir, 'norm_stats.json'))

    meta_src = os.path.join(data_dir, 'metadata.npz')
    if os.path.exists(meta_src):
        shutil.copy(meta_src, os.path.join(output_dir, 'metadata.npz'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', help='Finalized source dataset directory')
    parser.add_argument('--protocol', required=True, nargs='+',
                         choices=['OUE', 'OLH_Server', 'OLH_User', 'HST_Server', 'HST_User'],
                         help='Protocol(s) to filter and sample. With multiple protocols, --n rows are '
                              'sampled independently from EACH one (e.g. 3 protocols + --n 10000000 = '
                              '30,000,000 rows total in one output directory).')
    parser.add_argument('--n', type=int, required=True, dest='n_target',
                         help='Target number of sampled rows PER protocol')
    parser.add_argument('--output', '-o', required=True, help='Output directory for the sampled dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk-rows', type=int, default=CHUNK_ROWS_DEFAULT,
                         help='Rows per streaming chunk when scanning config.npy')
    parser.add_argument('--batch-rows', type=int, default=BATCH_ROWS_DEFAULT,
                         help='Rows per batch when writing the sampled output')
    args = parser.parse_args()

    config_mm = np.load(os.path.join(args.data_dir, 'config.npy'), mmap_mode='r')

    print(f"Scanning config.npy for protocol(s)={args.protocol}...")
    groups, totals = build_group_index(config_mm, args.protocol, args.chunk_rows)
    for protocol in args.protocol:
        print(f"  {protocol}: {totals.get(protocol, 0):,} matching rows")

    if not totals:
        raise ValueError(f"No rows found for protocol(s)={args.protocol} in {args.data_dir}")

    idx = choose_sample_indices(groups, totals, args.protocol, args.n_target, args.seed)
    print(f"Selected {len(idx):,} rows total")

    print(f"Writing sampled dataset to {args.output}...")
    write_sampled_dataset(args.data_dir, args.output, idx, args.batch_rows)
    print("Done.")


if __name__ == '__main__':
    main()
