"""Reference-beta ("beta-fixed") variant of the frequency-oracle detector.

main.py --source-beta B --val-data <dir> ... dispatches here. Two things differ from the
standard one-at-a-time run, nothing else (model, features, loss, optimizer, seeds are the same):

1. Which LDP runs feed the model. Optimization and validation rows come only from complete runs
   generated at attacker share B (`source_beta`). The other shares of a sweep are never trained on;
   they stay in the held-out test runs, so the sweep measures transfer to other shares.
   Runs are assigned to optimization or validation before any row is picked: optimization uses the
   training runs of --data-path, validation the runs of --val-data (extra runs of the same configs,
   generate_dataset.py --replicate-offset ... --role validation). Test = the held-out runs of
   --data-path, unchanged.
2. The decision threshold. After the usual early stopping (validation F1 at p > 0.5 on the balanced
   validation sample, restored best epoch), one threshold is chosen on the validation scores: the
   one with the best attacker-class F1 when the balanced validation sample is re-weighted to the
   natural share `threshold_reference_beta`. It is saved next to the checkpoint and used unchanged
   for every held-out evaluation (the 60,000-row test sample, every held-out row, IGR).

Names used in configs and metadata:
  source_beta               attacker share of a complete generated run used for training / validation
  training_class_share      attacker share of the sampled rows the model sees: 0.5
  threshold_reference_beta  the share the threshold-selection metric stands for
  test_beta                 attacker share of a held-out test run

Scores are logits; a user is flagged when logit > threshold (never >=), here and in IGR.

Files written to --output-dir (besides main.py's usual ones):
  variant.json         identity of the run (every input and setting, hashed) and its status
  split_manifest.json  eligible configs, quotas, the runs behind optimization / validation / test
  scaler.json          global z-score of the 4 v3 features, fitted on the optimization rows only
  threshold.json       the frozen threshold and how it was selected
"""

import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd

VARIANT = 'beta_fixed'
SCORE_SCALE = 'logit'
DECISION_OPERATOR = '>'
OBJECTIVE = 'attacker-class F1 on the validation sample re-weighted to threshold_reference_beta'
SEARCH = ('every distinct validation-score boundary (midpoint between neighbouring distinct scores), '
          'plus flag-all (-inf) and flag-none (+inf); ties on F1 -> the higher threshold')
CHECKPOINT_CRITERION = ('early stopping: epoch with the best validation F1 at p > 0.5 on the balanced '
                        'validation sample (Trainer.fit_with_validation, unchanged)')
REQUIRED_OUTPUTS = ('model.pt', 'scaler.json', 'threshold.json', 'split_manifest.json',
                    'test_results.csv', 'sensitivity_test_results.csv')


# ------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------

def allocate_evenly(total, k):
    """`total` split over k slots as evenly as integers allow; the first (total mod k) slots
    get one more, so the parts always sum to `total`."""
    if k < 1:
        raise ValueError('nothing to allocate over')
    base, rem = divmod(int(total), k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _json_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _write_json(path, obj):
    tmp = f'{path}.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------
# decision rule and threshold selection
# ------------------------------------------------------------------------------------------

def decide(scores, threshold):
    """The one decision rule: flag when score > threshold. Scores are compared as float64."""
    return np.asarray(scores, dtype=np.float64) > float(threshold)


def validation_weights(labels, reference_beta):
    """(attacker weight, genuine weight) that turn a validation sample with attacker share s into
    one with share reference_beta: beta / s and (1 - beta) / (1 - s)."""
    share = float(np.mean(np.asarray(labels).astype(bool)))
    if not 0.0 < share < 1.0:
        raise ValueError(f'validation sample needs both classes (attacker share {share})')
    if not 0.0 < reference_beta < 1.0:
        raise ValueError(f'reference beta must be in (0, 1), got {reference_beta}')
    return reference_beta / share, (1.0 - reference_beta) / (1.0 - share), share


def weighted_metrics(scores, labels, threshold, w_pos=1.0, w_neg=1.0):
    """Attacker-class metrics at one threshold with per-row weights w_pos (attackers) and w_neg
    (genuine): F1 = 2 TP / (2 TP + FP + FN). Zero denominators give 0."""
    y = np.asarray(labels).astype(bool)
    flag = decide(scores, threshold)
    tp = w_pos * float((flag & y).sum())
    fp = w_neg * float((flag & ~y).sum())
    fn = w_pos * float((~flag & y).sum())
    tn = w_neg * float((~flag & ~y).sum())
    return {
        'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0,
        'precision': tp / (tp + fp) if tp + fp > 0 else 0.0,
        'recall': tp / (tp + fn) if tp + fn > 0 else 0.0,
        'fpr': fp / (fp + tn) if fp + tn > 0 else 0.0,
        'flagged_share': (tp + fp) / (tp + fp + fn + tn),
    }


def select_threshold(scores, labels, reference_beta):
    """Threshold with the best weighted attacker-class F1 on the validation scores.

    Candidates are all operating points the scores allow: flag none (+inf), then for each distinct
    score value v (highest first) "flag every row with score >= v", written as the midpoint between
    v and the next lower distinct value (-inf below the lowest = flag all). Rows with tied scores
    are always flagged together. Among equal F1 the higher threshold wins.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    if len(s) != len(y) or len(s) == 0:
        raise ValueError('scores and labels must be non-empty and the same length')
    if not np.isfinite(s).all():
        raise ValueError('validation scores must be finite')
    w_pos, w_neg, share = validation_weights(y, reference_beta)

    order = np.argsort(-s, kind='stable')
    s_sorted, y_sorted = s[order], y[order]
    ctp = np.cumsum(np.where(y_sorted, w_pos, 0.0))
    cfp = np.cumsum(np.where(y_sorted, 0.0, w_neg))
    last = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])   # last row of each distinct score
    distinct = s_sorted[last]                                           # descending
    lower = np.r_[distinct[1:], -np.inf]
    thresholds = np.r_[np.inf, np.where(np.isfinite(lower), (distinct + lower) / 2.0, -np.inf)]
    tp = np.r_[0.0, ctp[last]]
    fp = np.r_[0.0, cfp[last]]
    fn = ctp[-1] - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.argmax(f1))                    # first maximum = highest threshold
    thr = float(thresholds[best])
    at = weighted_metrics(s, y, thr, w_pos, w_neg)
    half = weighted_metrics(s, y, 0.0, w_pos, w_neg)
    return {
        'threshold': thr,
        'threshold_probability': _sigmoid(thr),
        'operating_point': 'flag_none' if thr == np.inf else 'flag_all' if thr == -np.inf else 'interior',
        'validation_f1': at['f1'], 'validation_precision': at['precision'],
        'validation_recall': at['recall'], 'validation_fpr': at['fpr'],
        'validation_flagged_share': at['flagged_share'],
        'validation_f1_at_p0.5': half['f1'], 'validation_precision_at_p0.5': half['precision'],
        'validation_recall_at_p0.5': half['recall'], 'validation_fpr_at_p0.5': half['fpr'],
        'candidates': int(len(thresholds)),
        'attacker_weight': w_pos, 'genuine_weight': w_neg,
        'validation_rows': int(len(y)), 'validation_attacker_share': share,
    }


def _sigmoid(x):
    if x == np.inf:
        return 1.0
    if x == -np.inf:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def save_threshold(model_dir, record):
    rec = dict(record)
    thr = float(rec['threshold'])
    rec['threshold'] = thr if math.isfinite(thr) else ('inf' if thr > 0 else '-inf')   # plain JSON
    _write_json(os.path.join(model_dir, 'threshold.json'), rec)


def load_threshold(model_dir):
    """The frozen threshold record of a model directory. No fallback: a model of this variant
    without its threshold cannot be evaluated."""
    path = os.path.join(model_dir, 'threshold.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} is missing. A {VARIANT} model is only evaluated at its validation-selected '
            f'threshold; there is no 0.5 fallback. Re-run main.py --source-beta for this model.')
    with open(path) as f:
        rec = json.load(f)
    if rec.get('decision_operator') != DECISION_OPERATOR or rec.get('score_scale') != SCORE_SCALE:
        raise ValueError(f'{path}: unexpected decision rule {rec.get("score_scale")} '
                         f'{rec.get("decision_operator")}')
    rec['threshold'] = float(rec['threshold'])
    return rec


def load_scaler(model_dir):
    path = os.path.join(model_dir, 'scaler.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} is missing; a {VARIANT} model needs the scaler fitted on its '
                                f'optimization rows')
    with open(path) as f:
        return json.load(f)


def is_variant_dir(model_dir):
    return os.path.exists(os.path.join(model_dir, 'variant.json'))


def model_logits(model, X, device, batch=32_768):
    import torch
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for a in range(0, len(X), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[a:a + batch], dtype=np.float32)).to(device)
            out[a:a + batch] = model(xb).reshape(-1).cpu().numpy()
    return out


# ------------------------------------------------------------------------------------------
# split: runs first, then rows
# ------------------------------------------------------------------------------------------

_KEY = ['dataset_type', 'protocol', 'epsilon', 'attacker_ratio', 'target_set_size', 'splits']


def _frame(ds, idx):
    return pd.DataFrame({
        'dataset_type': ds.dataset_types[idx], 'protocol': ds.protocols[idx],
        'epsilon': np.round(ds.epsilons[idx], 9), 'attacker_ratio': np.round(ds.attacker_ratios[idx], 9),
        'target_set_size': ds.target_set_sizes[idx].astype(int), 'splits': ds.splits[idx].astype(int),
        'run': ds.experiment_ids[idx], 'y': ds.labels[idx].astype(int), 'idx': idx,
    })


def _runs_of(frame):
    g = frame.groupby(_KEY + ['run'], sort=True).agg(rows=('y', 'size'), attackers=('y', 'sum')).reset_index()
    return g


def _sample_config(frame, quota, rng, what, cfg):
    """`quota` attackers and `quota` genuine rows of one config, drawn from all of its runs in the
    pool. More runs only supply the quota; the config's weight is the quota."""
    out = []
    for cls, name in ((1, 'attackers'), (0, 'genuine users')):
        rows = np.sort(frame.loc[frame.y == cls, 'idx'].to_numpy())
        if len(rows) < quota:
            per_run = max(1, len(rows) // max(1, frame.run.nunique()))
            raise ValueError(
                f'{what}: config {cfg} has {len(rows):,} {name} in {frame.run.nunique()} run(s) but '
                f'needs {quota:,}. Generate at least {math.ceil(quota / per_run)} complete run(s) of '
                f'this config for this split.')
        out.append(rng.choice(rows, size=quota, replace=False))
    return np.concatenate(out)


def build_split(ds, ds_val, *, source_beta, train_datasets, test_dataset, train_samples=500_000,
                val_size=0.1, test_samples=60_000, seed=42):
    """Optimization / validation / test rows of the reference-beta design.

    ds      the main one-at-a-time dataset, v3 features before the global z-score, already filtered
            to one protocol (and one dataset for in-distribution models)
    ds_val  the validation source runs (same filtering); every run there is at source_beta

    Returns the split dict main.py's code expects (features scaled with the optimization-row
    scaler; ds.features is replaced by the scaled array), plus 'manifest' and 'scaler'.
    """
    from sklearn.model_selection import train_test_split
    from .data.dataset import _balanced_per_group, _config_groups

    train_datasets = list(train_datasets)
    if test_dataset in train_datasets and len(train_datasets) > 1:
        raise ValueError('the test dataset of a cross-dataset model cannot be a source dataset')
    y = ds.labels.astype(int)
    hold = ds.holdout
    src = np.isclose(ds.attacker_ratios, source_beta, atol=1e-9)
    in_train_ds = np.isin(ds.dataset_types, train_datasets)

    pool = np.where((hold == 0) & in_train_ds & src)[0]
    held = np.where((hold == 1) & (ds.dataset_types == test_dataset))[0]
    if len(pool) == 0:
        raise ValueError(f'no training runs at source beta {source_beta} for datasets {train_datasets}')
    if len(held) == 0:
        raise ValueError(f'no held-out runs for test dataset {test_dataset!r}')

    # ---- validation runs: a separate directory of runs, all at source beta
    if not np.isclose(ds_val.attacker_ratios, source_beta, atol=1e-9).all():
        raise ValueError(f'--val-data holds runs at another attacker share than {source_beta}')
    if (ds_val.holdout != 0).any():
        raise ValueError('--val-data must not hold held-out (test) runs')
    val_pool = np.where(np.isin(ds_val.dataset_types, train_datasets))[0]
    if len(val_pool) == 0:
        raise ValueError(f'--val-data has no runs for datasets {train_datasets}')

    opt_f, val_f = _frame(ds, pool), _frame(ds_val, val_pool)
    configs = sorted(map(tuple, opt_f[_KEY].drop_duplicates().to_numpy().tolist()))
    val_configs = sorted(map(tuple, val_f[_KEY].drop_duplicates().to_numpy().tolist()))
    if configs != val_configs:
        missing = sorted(set(configs) - set(val_configs))
        extra = sorted(set(val_configs) - set(configs))
        raise ValueError(f'--val-data does not cover the eligible configs: missing {missing}, '
                         f'not in the training data {extra}')

    val_total = int(round(train_samples * val_size))
    opt_total = train_samples - val_total
    if opt_total % 2 or val_total % 2:
        raise ValueError(f'optimization ({opt_total}) and validation ({val_total}) budgets must be even')
    k = len(configs)
    opt_quota = allocate_evenly(opt_total // 2, k)
    val_quota = allocate_evenly(val_total // 2, k)

    rng_opt, rng_val = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
    opt_by_cfg = {c: g for c, g in opt_f.groupby(_KEY, sort=True)}
    val_by_cfg = {c: g for c, g in val_f.groupby(_KEY, sort=True)}
    tr_parts, va_parts, cfg_rows = [], [], []
    for c, q_opt, q_val in zip(configs, opt_quota, val_quota):
        tr_parts.append(_sample_config(opt_by_cfg[c], q_opt, rng_opt, 'optimization', c))
        va_parts.append(_sample_config(val_by_cfg[c], q_val, rng_val, 'validation', c))
        cfg_rows.append({
            **dict(zip(_KEY, c)), 'source_beta': float(c[3]),
            'optimization_rows_per_class': q_opt, 'validation_rows_per_class': q_val,
            'optimization_runs': _runs_of(opt_by_cfg[c])[['run', 'rows', 'attackers']].to_dict('records'),
            'validation_runs': _runs_of(val_by_cfg[c])[['run', 'rows', 'attackers']].to_dict('records'),
        })
    tr_idx = rng_opt.permutation(np.concatenate(tr_parts))
    va_idx = rng_val.permutation(np.concatenate(va_parts))

    # ---- held-out test: the same rows the standard run takes (same call, same seed)
    held_groups = _config_groups(ds, held)
    held_strata = held_groups * 2 + y[held]
    if test_samples >= len(held):
        test_idx = held
    else:
        test_idx, _ = train_test_split(held, train_size=test_samples, random_state=seed,
                                       stratify=held_strata)
    per_class_held = int(np.bincount(held_strata).max())
    sens_bal, _ = _balanced_per_group(held, held_groups, y[held], per_class_held,
                                      np.random.default_rng(seed))

    # ---- global scaler: optimization rows only, then applied unchanged everywhere
    raw_opt = np.asarray(ds.features[tr_idx], dtype=np.float64)
    mean, std = raw_opt.mean(axis=0), raw_opt.std(axis=0)
    std[std < 1e-12] = 1.0
    scaler = {'feature_names': list(ds.feature_names), 'mean': mean.tolist(), 'std': std.tolist(),
              'fitted_on': 'optimization rows', 'rows': int(len(tr_idx))}

    def scale(a):
        return ((np.asarray(a, dtype=np.float64) - mean) / std).astype(np.float32)

    X_val = scale(ds_val.features[va_idx])
    y_val = ds_val.labels[va_idx]
    ds.features = scale(ds.features)                  # held-out rows are read from ds later
    X_train, y_train = ds.features[tr_idx], ds.labels[tr_idx]
    X_test, y_test = ds.features[test_idx], ds.labels[test_idx]

    other = (hold == 0) & in_train_ds & ~src
    test_f = _frame(ds, held)
    manifest = {
        'variant': VARIANT,
        'source_beta': float(source_beta), 'training_class_share': 0.5,
        'train_datasets': train_datasets, 'test_dataset': test_dataset,
        'protocols': sorted(set(ds.protocols[pool].tolist())),
        'excluded_training_betas': sorted(set(np.round(ds.attacker_ratios[other], 9).tolist())),
        'excluded_training_runs': int(_runs_of(_frame(ds, np.where(other)[0])).shape[0]) if other.any() else 0,
        'test_betas': sorted(set(test_f.attacker_ratio.tolist())),
        'budget': {'development_rows': int(train_samples), 'optimization_rows': int(len(tr_idx)),
                   'optimization_attackers': int(y_train.sum()),
                   'validation_rows': int(len(va_idx)), 'validation_attackers': int(y_val.sum())},
        'eligible_configs': cfg_rows,
        'optimization_run_count': int(sum(len(c['optimization_runs']) for c in cfg_rows)),
        'validation_run_count': int(sum(len(c['validation_runs']) for c in cfg_rows)),
        'test_runs': _runs_of(test_f).to_dict('records'),
        'test_sample_rows': int(len(test_idx)), 'heldout_rows': int(len(held)),
        'split_unit': 'complete LDP run (dataset_type, experiment_id); optimization runs come from '
                      '--data-path, validation runs from --val-data, test runs are the held-out runs '
                      'of --data-path',
    }
    print(f'Reference-beta split (source beta {source_beta:g}): {k} eligible configs over '
          f'{"+".join(train_datasets)}')
    print(f'  optimization {len(tr_idx):,} rows ({int(y_train.sum()):,} attackers) from '
          f'{manifest["optimization_run_count"]} runs; validation {len(va_idx):,} rows '
          f'({int(y_val.sum()):,} attackers) from {manifest["validation_run_count"]} separate runs')
    print(f'  excluded from training: beta {manifest["excluded_training_betas"]} '
          f'({manifest["excluded_training_runs"]} runs)')
    print(f'  test: {len(test_idx):,} sampled rows at natural prevalence '
          f'({100 * y_test.mean():.2f}% attackers); {len(held):,} held-out rows, '
          f'beta {manifest["test_betas"]}')
    return {
        'X_train': X_train, 'y_train': y_train, 'train_indices': tr_idx,
        'X_val': X_val, 'y_val': y_val, 'val_indices': va_idx,
        'X_test': X_test, 'y_test': y_test, 'test_indices': test_idx,
        'sens_indices': held, 'sens_balanced_indices': np.sort(sens_bal),
        'scaler': scaler, 'manifest': manifest,
    }


# ------------------------------------------------------------------------------------------
# run identity (resume, and no mixing with other variants)
# ------------------------------------------------------------------------------------------

_HP = ('epochs', 'patience', 'batch_size', 'lr', 'dropout', 'pos_weight', 'balanced_batches',
       'hidden_sizes', 'd_token', 'n_heads', 'n_layers', 'ffn_d_multiplier', 'attention_dropout',
       'residual_dropout')


def _data_identity(path):
    out = {'path': os.path.normpath(path)}
    for name in ('design.json', 'norm_stats.json'):
        p = os.path.join(path, name)
        out[name] = file_sha256(p) if os.path.exists(p) else None
    return out


def run_identity(args):
    ident = {
        'variant': VARIANT, 'objective': OBJECTIVE, 'search': SEARCH,
        'score_scale': SCORE_SCALE, 'decision_operator': DECISION_OPERATOR,
        'data': _data_identity(args.data_path), 'val_data': _data_identity(args.val_data),
        'feature_set': args.feature_set, 'protocol': args.protocol, 'dataset': args.dataset,
        'training_method': args.training_method,
        'train_dataset': list(args.train_dataset) if args.train_dataset else None,
        'test_dataset': args.test_dataset,
        'source_beta': float(args.source_beta),
        'threshold_reference_beta': float(args.threshold_reference_beta),
        'training_class_share': 0.5,
        'train_samples': args.train_samples, 'val_size': args.val_size,
        'test_samples': args.test_samples, 'seed': args.seed, 'model': args.model,
        'hyperparameters': {k: getattr(args, k, None) for k in _HP},
    }
    return ident, _json_sha(ident)


def check_args(args, parser_error):
    if args.val_data is None:
        parser_error('--source-beta needs --val-data (the directory of validation source runs)')
    if not args.output_dir:
        parser_error('--source-beta needs --output-dir')
    if args.feature_set != 'v3':
        parser_error('--source-beta needs --feature-set v3 (the global scaler is fitted on the '
                     'optimization rows; stored features are already scaled on every row)')
    if args.k_folds != 0:
        parser_error('--source-beta needs --k-folds 0')
    if not args.val_size > 0:
        parser_error('--source-beta needs --val-size > 0')
    if args.training_method == 'three-way':
        parser_error('--source-beta does not support --training-method three-way')
    if args.training_method == 'none' and not args.dataset:
        parser_error('--source-beta needs --dataset (in-distribution) or --training-method cross')
    if not args.protocol:
        parser_error('--source-beta needs --protocol')
    if not 0.0 < args.source_beta < 1.0:
        parser_error('--source-beta must be in (0, 1)')
    if os.path.abspath(args.val_data) == os.path.abspath(args.data_path):
        parser_error('--val-data must be a different directory than --data-path: validation rows '
                     'come from other runs than optimization rows')


def _check_val_design(design, design_val, source_beta):
    """The validation directory holds other runs than the main one: same generator settings,
    replicate indices beyond every replicate of the main dataset."""
    if not design or not design_val:
        raise ValueError('both --data-path and --val-data need a design.json (generate_dataset.py --design oat)')
    if design.get('design') != 'oat' or design_val.get('design') != 'oat':
        raise ValueError('the reference-beta variant needs one-at-a-time datasets')
    if design_val.get('role') != 'validation':
        raise ValueError("--val-data's design.json does not say role 'validation' "
                         '(generate_dataset.py --role validation --replicate-offset ...)')
    for key in ('attack', 'feature_set', 'block_candidates', 'n'):
        if design.get(key) != design_val.get(key):
            raise ValueError(f'--val-data was generated with another {key}: '
                             f'{design_val.get(key)!r} vs {design.get(key)!r}')
    if design.get('seed') is not None and design.get('seed') != design_val.get('seed'):
        raise ValueError('--val-data was generated with another base seed')
    used = max(int(r.get('train', 0)) + int(r.get('holdout', 0)) for r in design['runs'])
    used += int(design.get('replicate_offset') or 0)
    offset = int(design_val.get('replicate_offset') or 0)
    if offset < used:
        raise ValueError(f'--val-data replicate offset {offset} overlaps the main dataset '
                         f'(replicates 0..{used - 1}); its runs could repeat training or test runs')
    betas = {round(float(r['attacker_ratio']), 9) for r in design_val['runs']}
    if betas != {round(float(source_beta), 9)}:
        raise ValueError(f'--val-data holds runs at beta {sorted(betas)}, expected only {source_beta}')


def validation_generation_args(design, source_beta):
    """generate_dataset.py arguments for the validation runs of a main dataset: every config of the
    design that sits at source_beta, one new run each."""
    d = design['defaults']
    if not math.isclose(float(d['attacker_ratio']), float(source_beta), abs_tol=1e-9):
        raise ValueError(f"source beta {source_beta} is not the design's default share {d['attacker_ratio']}")
    s = design['sweeps']
    used = max(int(r.get('train', 0)) + int(r.get('holdout', 0)) for r in design['runs'])
    out = ['--design', 'oat', '--attack', design['attack'], '--datasets', *design['datasets'],
           '--epsilons', *[f'{v:g}' for v in s['epsilon']], '--ratios', f'{float(source_beta):g}',
           '--target-sizes', *[str(v) for v in s['target_set_size']],
           '--splits', *[str(v) for v in s['splits']],
           '--default-epsilon', f"{d['epsilon']:g}", '--default-ratio', f'{float(source_beta):g}',
           '--default-target-size', str(d['target_set_size']), '--default-splits', str(d['splits']),
           '--experiments', '1', '--holdout-replicates', '0',
           '--replicate-offset', str(max(1000, used)), '--role', 'validation',
           '--feature-set', design['feature_set']]
    if design.get('block_candidates'):
        out += ['--block-candidates', str(design['block_candidates'])]
    if design.get('n'):
        out += ['--n', str(design['n'])]
    if design.get('seed') is not None:
        out += ['--seed', str(design['seed'])]
    return out


# ------------------------------------------------------------------------------------------
# the run
# ------------------------------------------------------------------------------------------

def _complete(out_dir):
    return all(os.path.exists(os.path.join(out_dir, f)) for f in REQUIRED_OUTPUTS)


def run(args):
    import torch
    from .analysis import plot_sensitivity_metric, run_sensitivity_analysis
    from .data import load_npy_dataset
    from .data.dataset import apply_feature_set_v3
    from .models import FT_TRANSFORMER_HP_KEYS, get_model
    from .training import Trainer

    out = args.output_dir
    identity, identity_sha = run_identity(args)
    variant_path = os.path.join(out, 'variant.json')
    if os.path.exists(variant_path):
        with open(variant_path) as f:
            prev = json.load(f)
        if prev.get('identity_sha256') != identity_sha:
            raise SystemExit(f'{out} holds a {VARIANT} run with other settings or data '
                             f'({prev.get("identity_sha256", "?")[:12]} vs {identity_sha[:12]}); '
                             f'refusing to overwrite it. Use another --output-dir.')
        if prev.get('status') == 'complete' and _complete(out):
            print(f'{out}: complete ({identity_sha[:12]}); nothing to do')
            return
    elif os.path.isdir(out) and any(os.path.exists(os.path.join(out, f))
                                    for f in ('model.pt', 'sensitivity_test_results.csv')):
        raise SystemExit(f'{out} holds results of another experiment (no variant.json); refusing to '
                         f'overwrite them. Use another --output-dir.')
    os.makedirs(out, exist_ok=True)
    started = time.strftime('%F %T')
    _write_json(variant_path, {'variant': VARIANT, 'status': 'running', 'identity_sha256': identity_sha,
                               'identity': identity, 'started': started})

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'\nReference-beta variant: source beta {args.source_beta:g}, threshold reference beta '
          f'{args.threshold_reference_beta:g}, training class share 0.5')

    def load(path):
        d = load_npy_dataset(path)
        if not d.is_oat:
            raise ValueError(f'{path} is not a one-at-a-time dataset')
        apply_feature_set_v3(d, normalize=False)     # per-run statistics on complete runs, before filtering
        idx = np.where(d.protocols == args.protocol)[0]
        if args.dataset:
            idx = idx[d.dataset_types[idx] == args.dataset]
        if len(idx) == 0:
            raise ValueError(f'{path}: no rows for protocol={args.protocol!r}, dataset={args.dataset!r}')
        if len(idx) < len(d):
            d.features, d.labels, d.config = d.features[idx], d.labels[idx], d.config[idx]
        return d

    ds = load(args.data_path)
    ds_val = load(args.val_data)
    _check_val_design(ds.design, ds_val.design, args.source_beta)

    cross = args.training_method == 'cross'
    train_datasets = list(args.train_dataset) if cross else [args.dataset]
    test_dataset = args.test_dataset if cross else args.dataset
    split = build_split(ds, ds_val, source_beta=args.source_beta, train_datasets=train_datasets,
                        test_dataset=test_dataset, train_samples=args.train_samples,
                        val_size=args.val_size, test_samples=args.test_samples, seed=args.seed)
    del ds_val
    _write_json(os.path.join(out, 'scaler.json'), split['scaler'])
    scaler_sha = file_sha256(os.path.join(out, 'scaler.json'))
    manifest = dict(split['manifest'], threshold_reference_beta=float(args.threshold_reference_beta),
                    seed=args.seed, identity_sha256=identity_sha, scaler_sha256=scaler_sha,
                    data_path=identity['data'], val_data=identity['val_data'])
    _write_json(os.path.join(out, 'split_manifest.json'), manifest)

    # ---- training: same model, loss, optimizer, early stopping as the standard run
    n_features = ds.features.shape[1]
    model_kwargs = {'dropout_rate': args.dropout}
    if args.model == 'mlp' and args.hidden_sizes is not None:
        model_kwargs['hidden_sizes'] = args.hidden_sizes
    if args.model == 'ft_transformer':
        model_kwargs.update({k: getattr(args, k) for k in FT_TRANSFORMER_HP_KEYS
                             if getattr(args, k) is not None})
    model = get_model(args.model, input_dim=n_features, **model_kwargs)
    print(model)
    pos_weight = None if args.pos_weight == 'auto' else float(args.pos_weight)
    trainer = Trainer(model, device, learning_rate=args.lr, model_type=args.model,
                      epochs=args.epochs, pos_weight=pos_weight)
    print('Training with early stopping on the validation runs...')
    train_result = trainer.fit_with_validation(
        split['X_train'], split['y_train'], split['X_val'], split['y_val'],
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        balanced_batches=args.balanced_batches)
    pd.DataFrame(train_result['history']).to_csv(os.path.join(out, 'training_history.csv'), index=False)
    model_path = os.path.join(out, 'model.pt')
    trainer.save(model_path)
    model_sha = file_sha256(model_path)

    # ---- threshold: one pass over the validation rows, then frozen
    val_scores = model_logits(model, split['X_val'], device)
    sel = select_threshold(val_scores, split['y_val'], args.threshold_reference_beta)
    save_threshold(out, {
        'variant': VARIANT,
        'model': {'type': args.model, 'output_dir': os.path.normpath(out),
                  'checkpoint': 'model.pt', 'checkpoint_sha256': model_sha,
                  'scaler': 'scaler.json', 'scaler_sha256': scaler_sha,
                  'identity_sha256': identity_sha},
        'protocol': args.protocol, 'training_method': args.training_method,
        'train_datasets': train_datasets, 'test_dataset': test_dataset,
        'source_beta': float(args.source_beta),
        'threshold_reference_beta': float(args.threshold_reference_beta),
        'training_class_share': 0.5, 'seed': args.seed,
        'eligible_configs': len(manifest['eligible_configs']),
        'optimization_runs': manifest['optimization_run_count'],
        'validation_runs': manifest['validation_run_count'],
        'split_manifest': 'split_manifest.json',
        'checkpoint_selection': CHECKPOINT_CRITERION,
        'best_epoch': train_result['best_epoch'], 'best_val_f1_at_p0.5': train_result['best_val_f1'],
        'objective': OBJECTIVE,
        'validation_weighting': ('per-row weights beta / s for attackers and (1 - beta) / (1 - s) for '
                                 'genuine users, s = attacker share of the validation sample; applied '
                                 'to the validation metric only, never to the training loss'),
        'search': SEARCH, 'score_scale': SCORE_SCALE, 'decision_operator': DECISION_OPERATOR,
        **sel,
    })

    # ---- evaluation: always through the stored threshold
    frozen = load_threshold(out)
    thr = frozen['threshold']
    print(f'\nFrozen threshold: logit > {thr:.6g} (p > {frozen["threshold_probability"]:.6g}); '
          f'validation F1 {frozen["validation_f1"]:.4f} at beta {args.threshold_reference_beta:g} '
          f'(precision {frozen["validation_precision"]:.4f}, recall {frozen["validation_recall"]:.4f}, '
          f'FPR {frozen["validation_fpr"]:.5f}); at p > 0.5 the same metric is '
          f'{frozen["validation_f1_at_p0.5"]:.4f}')
    stamp = {'Threshold': thr, 'Model_SHA256': model_sha[:16]}

    test_label = f'Test ({test_dataset})' if cross else 'Test'
    test_metrics = trainer.evaluate(split['X_test'], split['y_test'], label=test_label, logit_threshold=thr)
    pd.DataFrame([{'label': test_label, 'n_samples': len(split['y_test']), **test_metrics, **stamp,
                   'Threshold_Probability': frozen['threshold_probability'],
                   'Source_Beta': float(args.source_beta)}]
                 ).to_csv(os.path.join(out, 'test_results.csv'), index=False)

    oat_defaults = ds.design['defaults']
    sens = split['sens_indices']
    print(f'\nRunning Sensitivity Analysis on {len(sens):,} held-out rows...')
    sensitivity_df = run_sensitivity_analysis(
        model, ds.features[sens], ds.labels[sens], device, config_array=ds.config[sens],
        batch_size=4096, oat_defaults=oat_defaults, logit_threshold=thr).assign(**stamp)
    print(sensitivity_df.to_string(index=False))
    sensitivity_df.to_csv(os.path.join(out, 'sensitivity_test_results.csv'), index=False)

    if args.save_predictions:
        sens_logits = model_logits(model, ds.features[sens], device)
        np.savez_compressed(
            os.path.join(out, 'test_predictions.npz'),
            prob=(1.0 / (1.0 + np.exp(-sens_logits.astype(np.float64)))).astype(np.float32),
            logit=sens_logits, label=ds.labels[sens].astype(np.int8),
            epsilon=ds.epsilons[sens].astype(np.float32),
            attacker_ratio=ds.attacker_ratios[sens].astype(np.float32),
            splits=ds.splits[sens].astype(np.int8), dataset=ds.dataset_types[sens].astype('U5'),
            threshold=np.float64(thr))

    bal = split['sens_balanced_indices']
    run_sensitivity_analysis(
        model, ds.features[bal], ds.labels[bal], device, config_array=ds.config[bal], batch_size=4096,
        oat_defaults=oat_defaults, logit_threshold=thr).assign(**stamp).to_csv(
            os.path.join(out, 'sensitivity_test_balanced.csv'), index=False)

    if not args.no_plot:
        for metric in ('F1_Score', 'Accuracy', 'Precision', 'Recall'):
            plot_sensitivity_metric(sensitivity_df, metric=metric,
                                    save_path=os.path.join(out, f'sensitivity_test_{metric.lower()}.png'))

    _write_json(variant_path, {'variant': VARIANT, 'status': 'complete', 'identity_sha256': identity_sha,
                               'identity': identity, 'started': started,
                               'finished': time.strftime('%F %T'), 'model_sha256': model_sha,
                               'scaler_sha256': scaler_sha, 'threshold': frozen['threshold'] if
                               math.isfinite(frozen['threshold']) else str(frozen['threshold'])})
    print('\nDone!')


if __name__ == '__main__':      # python -m attacker_detector.reference_beta gen-args <data dir> <beta>
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == 'gen-args':
        with open(os.path.join(sys.argv[2], 'design.json')) as fh:
            print(' '.join(validation_generation_args(json.load(fh), float(sys.argv[3]))))
    else:
        sys.exit('usage: python -m attacker_detector.reference_beta gen-args <data dir> <source beta>')
