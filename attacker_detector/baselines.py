"""FIAD / Diffstats baselines, scored on fresh LDP runs of a dataset's configs.

main.py --baseline fiad diffstats -d <dataset> ... dispatches here.

Both detectors work on a whole run's (n, d) support matrix, which datasets do
not keep, so every config of the dataset (design.json) is simulated again with
the same generator, attack and n. Runs are seeded like the dataset's held-out
replicates, so the users' true items and the target set match those runs; the
perturbation and attack randomness is fresh.

Writes, per method, to <output-dir>/<method>/:
  baseline_runs.csv             one row per simulated run (appended as each run
                                finishes; a rerun skips runs already there)
  sensitivity_test_results.csv  same columns and slices as main.py's DL file
  test_results.csv              the overall row
"""

import csv
import json
import os
import random
import time

import numpy as np
import pandas as pd

from config import DATASET_CONFIGS
from generate_dataset import ATTACK_H_AO, _oat_seed, _resolve_protocol
from attacker_detector.analysis.sensitivity import sensitivity_table_from_counts
from attacker_detector.data.dataset import (
    _CONF_TARGET_SET_SIZE,
    _CONF_ATTACKER_RATIO,
    _CONF_PROTOCOL,
    _CONF_SPLITS,
    _CONF_EPSILON,
    _CONF_DATASET_TYPE,
    _CONF_N,
)
from attacker_detector.data.generators import generate_perturbed_data
from attacker_detector.DiffStats import diffstats_detect
from attacker_detector.FIAD import fiad_detect


BASELINES = {
    'fiad': lambda support, one_list, run, n_jobs: fiad_detect(
        support, run['epsilon'], run['protocol'], n_jobs=n_jobs),
    'diffstats': lambda support, one_list, run, n_jobs: diffstats_detect(
        support, one_list, run['epsilon'], run['domain'], run['protocol']),
}

KEY_COLS = ['dataset_type', 'protocol', 'epsilon', 'attacker_ratio',
            'target_set_size', 'splits', 'replicate']
RUN_COLS = KEY_COLS + ['n', 'seed', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall',
                       'F1_Score', 'FPR', 'FNR', 'flagged', 'seconds']
_CONFIG_BLOCK = 1_000_000


def _config_groups_from_npy(data_path: str) -> list:
    """Distinct (target, ratio, protocol, splits, epsilon, dataset, n) rows of
    config.npy, read in blocks through a memory map."""
    cfg = np.load(os.path.join(data_path, 'config.npy'), mmap_mode='r')
    seen = []
    for start in range(0, len(cfg), _CONFIG_BLOCK):
        block = pd.DataFrame(np.asarray(cfg[start:start + _CONFIG_BLOCK, :_CONF_N + 1]))
        seen.append(block.drop_duplicates())
    groups = pd.concat(seen).drop_duplicates().to_numpy()

    def _num(s):
        return int(s) if s.lstrip('-').isdigit() else float(s)

    return [{
        'dataset_type': c[_CONF_DATASET_TYPE], 'protocol': c[_CONF_PROTOCOL],
        'epsilon': _num(c[_CONF_EPSILON]), 'attacker_ratio': _num(c[_CONF_ATTACKER_RATIO]),
        'target_set_size': _num(c[_CONF_TARGET_SET_SIZE]), 'splits': _num(c[_CONF_SPLITS]),
        'n': int(c[_CONF_N]), 'train': 0, 'holdout': 0,
    } for c in groups]


def plan_runs(args) -> tuple:
    """(runs, design, attack): every LDP run to simulate, the dataset's
    design.json (None without one) and the attack it was generated with."""
    design_path = os.path.join(args.data_path, 'design.json')
    design = None
    if os.path.exists(design_path):
        with open(design_path) as f:
            design = json.load(f)
        groups = [dict(g) for g in design['runs']]
        if design.get('n') is None:
            # generated without --n: each dataset's n is in config.npy
            n_by_dataset = {g['dataset_type']: g['n'] for g in _config_groups_from_npy(args.data_path)}
            for g in groups:
                g['n'] = n_by_dataset[g['dataset_type']]
        else:
            for g in groups:
                g['n'] = int(design['n'])
        attack = design.get('attack')
    else:
        print(f"  [WARN] no design.json in {args.data_path}; configs come from config.npy")
        groups = _config_groups_from_npy(args.data_path)
        attack = None
    if attack is None:
        attack = 'apa-approx'
        print(f"  [WARN] the dataset does not record its attack; assuming the generator "
              f"default --attack {attack}")

    datasets = {d for d in (args.dataset, args.test_dataset, args.eval_dataset) if d}
    runs = []
    for g in groups:
        if args.protocol and g['protocol'] != args.protocol:
            continue
        if datasets and g['dataset_type'] not in datasets:
            continue
        replicates = args.baseline_runs or max(1, int(g.get('holdout', 0)))
        for j in range(replicates):
            # the dataset's held-out replicates are numbered train, train + 1, ...
            rep = int(g.get('train', 0)) + j
            runs.append({
                **{k: g[k] for k in KEY_COLS[:-1]},
                'replicate': rep,
                'n': g['n'],
                'domain': DATASET_CONFIGS[g['dataset_type']]['domain'],
                'h_ao': ATTACK_H_AO[attack],
                'seed': _oat_seed(args.seed, g['dataset_type'], g['protocol'], g['epsilon'],
                                  g['attacker_ratio'], g['target_set_size'], g['splits'], rep),
            })
    return runs, design, attack


def _key(row) -> tuple:
    return tuple(str(row[k]) for k in KEY_COLS)


def _score(flagged: np.ndarray, labels: np.ndarray) -> dict:
    pred = np.zeros(len(labels), dtype=bool)
    pred[flagged] = True
    y = labels.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    tn = len(y) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': precision, 'Recall': recall,
        'F1_Score': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'FPR': fp / (fp + tn) if fp + tn else 0.0,
        'FNR': fn / (tp + fn) if tp + fn else 0.0,
        'flagged': int(pred.sum()),
    }


def _read_runs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=RUN_COLS)
    return pd.read_csv(path, dtype=str)


def _append_run(path: str, row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=RUN_COLS, extrasaction='ignore')
        if new:
            writer.writeheader()
        writer.writerow(row)


def _write_tables(method: str, runs_path: str, planned: set, oat_defaults, out_dir: str):
    done = _read_runs(runs_path)
    done = done[[_key(r) in planned for _, r in done.iterrows()]]
    if done.empty:
        print(f"  [WARN] {method}: no finished runs to tabulate")
        return None

    run_config = np.empty((len(done), _CONF_N + 1), dtype=object)
    for col, idx in (('target_set_size', _CONF_TARGET_SET_SIZE),
                     ('attacker_ratio', _CONF_ATTACKER_RATIO),
                     ('protocol', _CONF_PROTOCOL), ('splits', _CONF_SPLITS),
                     ('epsilon', _CONF_EPSILON), ('dataset_type', _CONF_DATASET_TYPE),
                     ('n', _CONF_N)):
        run_config[:, idx] = done[col].to_numpy()
    run_config = run_config.astype(str)
    counts = {c: done[c].astype(np.int64).to_numpy() for c in ('TP', 'FP', 'FN', 'TN')}

    table = sensitivity_table_from_counts(run_config, counts['TP'], counts['FP'],
                                          counts['FN'], counts['TN'], oat_defaults)
    table.to_csv(os.path.join(out_dir, 'sensitivity_test_results.csv'), index=False)

    overall = table.iloc[0]
    pd.DataFrame([{
        'label': f'Test ({method})',
        'n_samples': int(overall['Count']),
        **{m: overall[m] for m in ('Accuracy', 'Precision', 'Recall', 'F1_Score')},
    }]).to_csv(os.path.join(out_dir, 'test_results.csv'), index=False)
    return table


def run_baselines(args) -> None:
    methods = list(dict.fromkeys(args.baseline))
    runs, design, attack = plan_runs(args)
    if not runs:
        raise ValueError(f"No configs in {args.data_path} for protocol={args.protocol!r}, "
                         f"dataset={args.dataset or args.test_dataset!r}")
    oat_defaults = design['defaults'] if design and design.get('design') == 'oat' else None

    out_dirs = {m: os.path.join(args.output_dir, m) for m in methods}
    runs_paths = {m: os.path.join(out_dirs[m], 'baseline_runs.csv') for m in methods}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)
    done = {m: {_key(r) for _, r in _read_runs(runs_paths[m]).iterrows()} for m in methods}

    print("=" * 70)
    print(f"Baselines: {', '.join(methods)}  (attack={attack}, h_ao={ATTACK_H_AO[attack]})")
    print(f"  {len(runs)} LDP runs from {args.data_path}"
          + (f", one-at-a-time defaults {oat_defaults}" if oat_defaults else ""))
    print(f"  Output: {args.output_dir}")
    print("=" * 70)

    for i, run in enumerate(runs, 1):
        key = _key(run)
        todo = [m for m in methods if key not in done[m]]
        desc = (f"[{i}/{len(runs)}] {run['dataset_type']} {run['protocol']} "
                f"eps={run['epsilon']} beta={run['attacker_ratio']} r={run['target_set_size']} "
                f"r'={run['splits']} rep={run['replicate']} n={run['n']:,}")
        if not todo:
            print(f"{desc}  (done)")
            continue

        t0 = time.time()
        base_protocol, olh_setting = _resolve_protocol(run['protocol'])
        random.seed(run['seed'])
        support, labels, _, _, one_list = generate_perturbed_data(
            epsilon=run['epsilon'], domain=run['domain'], n=run['n'],
            protocol=base_protocol, ratio=run['attacker_ratio'],
            target_set_size=run['target_set_size'], splits=run['splits'],
            dataset_type=run['dataset_type'], h_ao=run['h_ao'], seed=run['seed'],
            processors=args.baseline_processors, olh_setting=olh_setting,
        )
        # Both detectors read support as present / absent (HST: y * s > 0).
        support = support > 0
        print(f"{desc}  simulated in {time.time() - t0:.0f}s")

        for method in todo:
            t0 = time.time()
            flagged = BASELINES[method](support, one_list, run, args.baseline_processors)
            row = {**run, **_score(flagged, labels), 'seconds': round(time.time() - t0, 2)}
            _append_run(runs_paths[method], row)
            done[method].add(key)
            print(f"    {method:9} F1={row['F1_Score']:.4f}  P={row['Precision']:.4f}  "
                  f"R={row['Recall']:.4f}  flagged={row['flagged']:,}  ({row['seconds']:.0f}s)")
        del support, labels, one_list

    planned = {_key(r) for r in runs}
    for method in methods:
        table = _write_tables(method, runs_paths[method], planned, oat_defaults, out_dirs[method])
        if table is not None:
            print(f"\n{method} sensitivity ({out_dirs[method]}/sensitivity_test_results.csv):")
            print(table.to_string(index=False))
    print("\nDone!")
