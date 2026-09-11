import argparse
import contextlib
import io
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_CONFIG_COLUMNS  # noqa: E402

COLS_V1 = list(DATASET_CONFIG_COLUMNS)                       # 6 cols (legacy datasets)
COLS_V2 = COLS_V1 + ['n_users', 'experiment_id', 'row_in_experiment']  # 9 cols (v2 datasets)

CANDIDATE_DIRS = [
    'outputs/dataset.csv',
    'outputs/dataset_v2',
    'outputs/dataset_diffstats',
    'outputs/diffstats_style',
]


class Tee:
    """Write to multiple streams at once (e.g. stdout + a report file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def human_size(num_bytes: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_value_counts(series: pd.Series, name: str):
    counts = series.value_counts()
    total = counts.sum()
    print(f"\n{name}:")
    for val, cnt in counts.items():
        print(f"  {val:<15} {cnt:>12,}  ({100.0 * cnt / total:5.1f}%)")


def normalize_protocol_family(protocol: str) -> str:
    """Collapse OLH_Server/OLH_User -> OLH, HST_Server/HST_User -> HST, etc."""
    return str(protocol).split('_')[0]


def load_npy_config_df(data_dir: str) -> pd.DataFrame:
    config = np.load(os.path.join(data_dir, 'config.npy'), allow_pickle=True)
    n_cols = config.shape[1]
    if n_cols == len(COLS_V1):
        cols = COLS_V1
    elif n_cols == len(COLS_V2):
        cols = COLS_V2
    else:
        cols = [f'col_{i}' for i in range(n_cols)]
        print(f"  [WARN] Unexpected config.npy shape {config.shape}; using generic column names.")

    df = pd.DataFrame(config, columns=cols)
    labels = np.load(os.path.join(data_dir, 'labels.npy'))
    df['label'] = labels.astype(int)
    return df


def iter_bin_config_chunks(config_bin_path: str):
    """Yield object-array chunks from a length-prefixed config.bin file."""
    with open(config_bin_path, 'rb') as f:
        while True:
            length_bytes = f.read(8)
            if not length_bytes or len(length_bytes) < 8:
                break
            length = int.from_bytes(length_bytes, 'little')
            chunk_bytes = f.read(length)
            chunk = np.load(io.BytesIO(chunk_bytes), allow_pickle=True)
            yield chunk


def summarize_bin_dataset(data_dir: str):
    """Stream-count an in-progress (.bin) dataset without materializing it."""
    config_bin = os.path.join(data_dir, 'config.bin')
    labels_bin = os.path.join(data_dir, 'labels.bin')

    print("  [Streaming .bin dataset — generation likely incomplete/in-progress]")

    if os.path.exists(labels_bin):
        labels = np.fromfile(labels_bin, dtype=np.float32)
        print(f"\n  Total user-rows so far (from labels.bin): {len(labels):,}")
        vals, counts = np.unique(labels, return_counts=True)
        print("\n  label:")
        for v, c in zip(vals, counts):
            tag = 'attacker' if v == 1 else 'benign'
            print(f"    {v:.0f} ({tag:<8}) {c:>12,}  ({100.0*c/len(labels):5.1f}%)")

    if not os.path.exists(config_bin):
        print("  [No config.bin found]")
        return

    n_cols = None
    column_counters = None
    total_rows = 0
    for chunk in iter_bin_config_chunks(config_bin):
        if n_cols is None:
            n_cols = chunk.shape[1]
            cols = COLS_V1 if n_cols == len(COLS_V1) else (
                COLS_V2 if n_cols == len(COLS_V2) else [f'col_{i}' for i in range(n_cols)]
            )
            column_counters = {c: Counter() for c in cols}
        total_rows += chunk.shape[0]
        for i, c in enumerate(cols):
            if c in ('n_users', 'experiment_id', 'row_in_experiment'):
                continue  # high-cardinality, not useful to tally
            vals, counts = np.unique(chunk[:, i], return_counts=True)
            column_counters[c].update(dict(zip(vals.tolist(), counts.tolist())))

    print(f"\n  Total user-rows so far (from config.bin): {total_rows:,}")
    for c, counter in column_counters.items():
        if not counter:
            continue
        print(f"\n  {c}:")
        for val, cnt in sorted(counter.items(), key=lambda kv: -kv[1]):
            print(f"    {val:<15} {cnt:>12,}  ({100.0 * cnt / total_rows:5.1f}%)")

        if c == 'protocol':
            families = Counter()
            for val, cnt in counter.items():
                families[normalize_protocol_family(val)] += cnt
            print(f"\n  protocol (grouped by mechanism):")
            for val, cnt in sorted(families.items(), key=lambda kv: -kv[1]):
                print(f"    {val:<15} {cnt:>12,}  ({100.0 * cnt / total_rows:5.1f}%)")


def summarize_npy_dataset(data_dir: str):
    df = load_npy_config_df(data_dir)
    n_features = None
    features_path = os.path.join(data_dir, 'features.npy')
    if os.path.exists(features_path):
        n_features = np.load(features_path, mmap_mode='r').shape[1]

    print(f"\n  Total samples: {len(df):,}")
    if n_features is not None:
        print(f"  Feature count: {n_features}")

    print_value_counts(df['label'].map({0: 'benign', 1: 'attacker'}), 'label')

    if 'dataset_type' in df.columns:
        print_value_counts(df['dataset_type'], 'dataset_type (zipf / emoji / fire)')

    if 'protocol' in df.columns:
        print_value_counts(df['protocol'], 'protocol (raw)')
        families = df['protocol'].map(normalize_protocol_family)
        print_value_counts(families, 'protocol (grouped by mechanism: OLH / OUE / HST)')

    if 'epsilon' in df.columns:
        print_value_counts(df['epsilon'].astype(float), 'epsilon')

    if 'splits' in df.columns:
        print_value_counts(df['splits'].astype(int), 'splits')

    if 'target_set_size' in df.columns:
        print_value_counts(df['target_set_size'].astype(int), 'target_set_size')

    if 'attacker_ratio' in df.columns:
        print_value_counts(df['attacker_ratio'].astype(float), 'attacker_ratio')

    if 'dataset_type' in df.columns:
        print("\nAttacker rate by dataset_type:")
        rate = df.groupby('dataset_type')['label'].mean()
        for k, v in rate.items():
            print(f"  {k:<10} {100.0*v:5.2f}% attackers")

    if 'dataset_type' in df.columns and 'protocol' in df.columns:
        print("\nSample counts: dataset_type x protocol")
        ct = pd.crosstab(df['dataset_type'], df['protocol'])
        print(ct.to_string())


def analyze_one(data_dir: str, use_bin: bool, output_dir: str = None):
    report_file = None
    tee_ctx = contextlib.nullcontext()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = data_dir.strip('/').replace('/', '_')
        report_path = os.path.join(output_dir, f"{safe_name}_analysis.txt")
        report_file = open(report_path, 'w')
        tee_ctx = contextlib.redirect_stdout(Tee(sys.stdout, report_file))

    with tee_ctx:
        print_header(f"Dataset: {data_dir}")
        if not os.path.isdir(data_dir):
            print("  [Not found]")
        else:
            size = dir_size(data_dir)
            print(f"  On-disk size: {human_size(size)}")

            has_npy = os.path.exists(os.path.join(data_dir, 'config.npy'))
            has_bin = os.path.exists(os.path.join(data_dir, 'config.bin'))

            if has_npy:
                summarize_npy_dataset(data_dir)
            elif has_bin:
                if use_bin:
                    summarize_bin_dataset(data_dir)
                else:
                    print(
                        "  [Only .bin chunks found — generation appears incomplete/in-progress.\n"
                        "   Re-run with --bin to stream-count it (can be slow for large files).]"
                    )
            else:
                print("  [No config.npy or config.bin found — nothing to analyze]")

    if report_file:
        report_file.close()
        print(f"  [Report written to {report_path}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', nargs='?', default='outputs/dataset.csv',
                         help='Dataset directory to analyze (default: outputs/dataset.csv)')
    parser.add_argument('--all', action='store_true', help='Scan all known dataset directories under outputs/')
    parser.add_argument('--bin', action='store_true',
                         help='Stream-count in-progress .bin datasets (slow for very large files)')
    parser.add_argument('-o', '--output-dir', default=None,
                         help='Directory to write a text report to (one file per dataset), in addition to printing')
    args = parser.parse_args()

    if args.all:
        for d in CANDIDATE_DIRS:
            analyze_one(d, use_bin=args.bin, output_dir=args.output_dir)
    else:
        analyze_one(args.data_dir, use_bin=args.bin, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
