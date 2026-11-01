"""Train once per (protocol, epsilon, training-source set), freeze, evaluate every destination.

"""

import json
import os
import time
import traceback

import numpy as np
import pandas as pd
import psutil
import torch

from .metrics import classification
from .methods import MethodUnavailable, TrainingRound, make_method
from .methods.checkpoint import save_checkpoint
from .sources import SOURCES
from .store import RoundStore


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, default=float)


def rounds_from_store(store: RoundStore, manifest: dict, split: str, protocol: str, eps: float, sources,
                      allowed=None):
    """Rounds of one shard group. `allowed` (round ids of the current config) drops rounds
    left by other configs or older spec schemas, so coverage is exactly what is configured."""
    out = []
    for src in sources:
        for rid, reports, groups, is_fake in store.load(split, protocol, eps, src):
            if allowed is not None and rid not in allowed:
                continue
            rec = manifest[rid]
            obs = {'reports': reports, 'groups': groups, 'n_g1': int(np.sum(groups == 1)),
                   'n_g2': int(np.sum(groups == 2)), 'protocol': protocol, 'epsilon': eps,
                   'domain': tuple(rec['domain']), 'square_mode': rec['square_mode']}
            out.append(TrainingRound(rid, obs, is_fake, rec['labels'], rec['meta'], rec['population_id']))
    return out


def _row(method, seed, tag, r: TrainingRound, rec, out):
    lab = r.labels
    row = {'partition': tag, 'method': method.name, 'kind': method.kind, 'seed': seed, 'round_id': r.round_id,
           'base_round_id': rec['base_round_id'], 'population_id': r.population_id, 'source': rec['source'],
           'protocol': rec['protocol'], 'epsilon': rec['epsilon'], 'beta': rec['beta'],
           'realized_beta': r.meta['realized_beta'], 'attack': rec['attack'], 'target': rec['target'],
           'total_size': rec['total_size'], 'spread_ratio': rec['spread_ratio'], 'status': rec['status'],
           'mean_hat_raw': out.mean_raw, 'var_hat_raw': out.var_raw, 'mean_hat': out.mean, 'var_hat': out.var,
           'genuine_mean': lab['genuine_mean'], 'genuine_variance': lab['genuine_variance'],
           'standard_mean': lab['standard_mean'], 'standard_variance': lab['standard_variance'],
           'target_mean': r.meta.get('target_mean', np.nan), 'target_variance': r.meta.get('target_variance', np.nan),
           'runtime': out.runtime, 'fallback': ','.join(map(str, out.fallback)),
           'budget_mode': rec.get('budget_mode', 'beta'), 'budget_factor': rec.get('budget_factor'),
           'below_budget': rec.get('below_budget', 'skip'), 'n_fake': r.meta.get('n_fake')}
    b = r.meta.get('budget', {})           # evaluation metadata only; never a model input
    row.update(m_min=b.get('m_min'), beta_min=b.get('beta_min'), oracle_m_min=b.get('oracle_m_min'),
               oracle_beta_min=b.get('oracle_beta_min'), budget_ratio=b.get('budget_ratio'),
               oracle_budget_ratio=b.get('oracle_budget_ratio'),
               theory_status=r.meta.get('theory', {}).get('status'),
               construction_status=r.meta.get('construction', {}).get('status'),
               realized_target_error_mean=r.meta.get('realized', {}).get('target_error_mean'),
               realized_target_error_variance=r.meta.get('realized', {}).get('target_error_variance'))
    row['sq_err_mean'] = (out.mean - lab['genuine_mean']) ** 2
    row['sq_err_var'] = (out.var - lab['genuine_variance']) ** 2
    row['sq_err_mean_raw'] = (out.mean_raw - lab['genuine_mean']) ** 2
    row['sq_err_var_raw'] = (out.var_raw - lab['genuine_variance']) ** 2
    if out.removed is not None:
        c = classification(r.is_fake, out.removed, out.scores)
        row.update({f'clf_{k}': v for k, v in c.items()})
        row.update({k: out.info.get(k) for k in ('removed_frac', 'retained_g1', 'retained_g2')})
    elif out.scores is not None and 0 < r.is_fake.sum() < len(r.is_fake):
        from sklearn.metrics import average_precision_score
        row['clf_pr_auc'] = float(average_precision_score(r.is_fake, out.scores))
    return row


def run_unit(store: RoundStore, manifest: dict, protocol: str, eps: float, train_sources, test_sources,
             method_name: str, seed: int, cfg: dict, out_dir: str, tag: str, allowed=None):
    """Fit one method on one training-source set and write per-round results for every destination."""
    key = f"{tag}__{protocol}_eps{eps:g}__{'+'.join(train_sources)}__{method_name}__s{seed}"
    done_path = os.path.join(out_dir, 'units', key + '.done.json')
    if os.path.exists(done_path):
        return 'skipped'
    os.makedirs(os.path.dirname(done_path), exist_ok=True)
    try:
        method = make_method(method_name, seed, cfg)
    except MethodUnavailable as e:
        _write_json(done_path, {'status': 'unavailable', 'reason': str(e)})
        return 'unavailable'
    train = rounds_from_store(store, manifest, 'train', protocol, eps, train_sources, allowed)
    val = rounds_from_store(store, manifest, 'val', protocol, eps, train_sources, allowed)
    proc = psutil.Process()
    t0 = time.perf_counter()
    try:
        method.fit(train, val)
    except Exception:
        _write_json(done_path, {'status': 'failed', 'stage': 'fit', 'error': traceback.format_exc()})
        return 'failed'
    fit_time = time.perf_counter() - t0
    save_checkpoint(method, os.path.join(out_dir, 'checkpoints', key))
    rows = []
    for dest in test_sources:
        test = rounds_from_store(store, manifest, 'test', protocol, eps, [dest], allowed)
        labels = [{'is_fake': r.is_fake} for r in test] if method.needs_labels else None
        outs = method.predict_many([r.observation for r in test], labels)
        rows += [_row(method, seed, tag, r, manifest[r.round_id], out) for r, out in zip(test, outs)]
    df = pd.DataFrame(rows)
    df['train_sources'] = '+'.join(train_sources)
    df['fit_time'] = fit_time
    df['peak_rss_mb'] = proc.memory_info().rss / 2 ** 20
    res_path = os.path.join(out_dir, 'per_round', key + '.csv')
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    df.to_csv(res_path, index=False)
    _write_json(done_path, {'status': 'completed', 'rows': len(df), 'fit_time': fit_time,
                            'n_train_rounds': len(train), 'n_val_rounds': len(val)})
    return 'completed'


def parse_unit_key(key: str) -> dict:
    tag, prot_eps, train, method, seed = key.split('__')
    protocol, eps = prot_eps.split('_eps')
    return {'tag': tag, 'protocol': protocol, 'train_epsilon': float(eps), 'train_sources': train,
            'method': method, 'seed': int(seed[1:])}


def evaluate_frozen(ckpt_root: str, store: RoundStore, manifest: dict, test_cfg: dict, out_dir: str,
                    allowed=None, methods=None, device: str = 'cpu'):
    """Score every saved checkpoint on a different config's test rounds (same protocol), with no
    refitting. Rows record the checkpoint's training epsilon and sources, so shifts in epsilon,
    spread, attack or population are explicit."""
    from .methods.checkpoint import load_checkpoint
    done = 0
    for key in sorted(os.listdir(ckpt_root)):
        info = parse_unit_key(key)
        if methods and info['method'] not in methods:
            continue
        for eps in test_cfg['epsilon']:
            if info['protocol'] not in test_cfg['protocols']:
                continue
            res_path = os.path.join(out_dir, 'per_round', f"frozen__{key}__test_eps{float(eps):g}.csv")
            if os.path.exists(res_path):
                continue
            method = load_checkpoint(os.path.join(ckpt_root, key), device)
            rows = []
            for dest in test_cfg['sources']:
                test = rounds_from_store(store, manifest, 'test', info['protocol'], float(eps), [dest], allowed)
                labels = [{'is_fake': r.is_fake} for r in test] if method.needs_labels else None
                outs = method.predict_many([r.observation for r in test], labels)
                rows += [_row(method, info['seed'], 'frozen-' + info['tag'], r, manifest[r.round_id], o)
                         for r, o in zip(test, outs)]
            df = pd.DataFrame(rows)
            df['train_sources'] = info['train_sources']
            df['train_epsilon'] = info['train_epsilon']
            df['checkpoint'] = key
            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            df.to_csv(res_path, index=False)
            done += 1
    return done


def coverage_table(manifest: dict) -> pd.DataFrame:
    """One row per spec with its generation status (infeasible / failed rounds stay visible)."""
    rows = [{k: rec[k] for k in ('round_id', 'split', 'source', 'protocol', 'epsilon', 'beta', 'attack', 'target',
                                  'status', 'population_id')} for rec in manifest.values()]
    return pd.DataFrame(rows)


def load_results(out_dir: str) -> pd.DataFrame:
    d = os.path.join(out_dir, 'per_round')
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.csv')) if os.path.isdir(d) else []
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _cluster_bootstrap(df, col, clusters='population_id', n_boot=500, seed=0):
    """95% CI of a mean, resampling parent populations (rounds are not independent)."""
    g = df.groupby(clusters)[col].agg(['sum', 'count'])
    if len(g) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    s, c = g['sum'].to_numpy()[idx].sum(1), g['count'].to_numpy()[idx].sum(1)
    return tuple(np.percentile(s / c, [2.5, 97.5]))


def aggregate(results: pd.DataFrame, by=('partition', 'train_sources', 'source', 'protocol', 'epsilon',
                                         'beta', 'attack', 'method'), n_boot: int = 500) -> pd.DataFrame:
    r = results.copy()
    r['std_sq_err_mean'] = (r['standard_mean'] - r['genuine_mean']) ** 2
    r['std_sq_err_var'] = (r['standard_variance'] - r['genuine_variance']) ** 2
    r['worse'] = (r['sq_err_mean'] + r['sq_err_var']) > (r['std_sq_err_mean'] + r['std_sq_err_var'])
    rows = []
    for key, g in r.groupby(list(by), dropna=False):
        row = dict(zip(by, key))
        row.update(n_rounds=len(g), n_populations=g['population_id'].nunique(), n_seeds=g['seed'].nunique(),
                   mse_mean=g['sq_err_mean'].mean(), mse_var=g['sq_err_var'].mean(),
                   mse_mean_before=g['std_sq_err_mean'].mean(), mse_var_before=g['std_sq_err_var'].mean(),
                   median_abs_err_mean=np.sqrt(g['sq_err_mean']).median(),
                   p90_abs_err_mean=np.sqrt(g['sq_err_mean']).quantile(0.9),
                   median_abs_err_var=np.sqrt(g['sq_err_var']).median(),
                   p90_abs_err_var=np.sqrt(g['sq_err_var']).quantile(0.9),
                   frac_worse=g['worse'].mean(), runtime_mean=g['runtime'].mean(),
                   fallback_rate=(g['fallback'].fillna('').astype(str) != '').mean())
        row['ag_mean'] = row['mse_mean_before'] - row['mse_mean']
        row['ag_var'] = row['mse_var_before'] - row['mse_var']
        row['mse_mean_ci'] = _cluster_bootstrap(g, 'sq_err_mean', n_boot=n_boot)
        row['mse_var_ci'] = _cluster_bootstrap(g, 'sq_err_var', n_boot=n_boot)
        for c in ('clf_precision', 'clf_recall', 'clf_f1', 'clf_pr_auc', 'clf_genuine_removal_rate'):
            if c in g:
                row[c] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)
